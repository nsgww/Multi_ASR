from __future__ import annotations

import asyncio
import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from multi_stream_asr.asr.offline.shared_server import SharedQwenServerManager
from multi_stream_asr.config import AppConfig, LaneConfig
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.session.shared_models import (
    EventCallback,
    SharedModelManager,
    SingleProcessPipeline,
)
from multi_stream_asr.web.event_bus import FrontendEventBus

_logger = logging.getLogger(__name__)


@dataclass
class Session:
    lane_id: int
    controller: SingleProcessPipeline
    on_event: EventCallback
    started_at: float = field(default_factory=time.monotonic)
    started_event: asyncio.Event = field(default_factory=asyncio.Event)
    finished_event: asyncio.Event = field(default_factory=asyncio.Event)


class SessionManager:
    """Manages in-process ASR sessions backed by shared model instances.

    - Loads the shared Qwen server and the shared Zipformer recognizer on startup.
    - Each WebSocket connection gets a lightweight ``SingleProcessPipeline`` that
      reuses the shared recognizers instead of spawning a child process.
    """

    def __init__(
        self,
        config: AppConfig,
        event_bus: FrontendEventBus,
        *,
        fake: bool = False,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._fake = fake
        self._state = PipelineState.UNLOADED
        self._shared_manager: SharedQwenServerManager | None = None
        self._shared_models = SharedModelManager(config, fake=fake)
        self._sessions: dict[int, Session] = {}
        self._available_lanes: list[int] = list(
            range(config.max_concurrent_sessions)
        )
        self._lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._lifecycle_timeout_seconds = (
            config.shared_qwen.request_timeout_ms / 1000.0 + 5.0
        )

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    @property
    def kws_loaded(self) -> bool:
        return self._shared_models.kws_loaded

    def set_hotwords(self, words: list[str]) -> None:
        self._shared_models.set_hotwords(words)

    def reload_kws_keywords(self) -> None:
        self._shared_models.reload_kws_keywords()

    async def load_models(self) -> None:
        async with self._lifecycle_lock:
            if self._state is not PipelineState.UNLOADED:
                raise RuntimeError(f"cannot load models from {self._state.value}")
            self._state = PipelineState.LOADING
            try:
                if not self._fake:
                    self._shared_manager = SharedQwenServerManager(self._config)
                    await asyncio.to_thread(self._shared_manager.start)
                await self._shared_models.load()
                self._state = PipelineState.LOADED
                _logger.info("models loaded; ready for sessions")
            except Exception:
                self._state = PipelineState.FATAL
                raise

    async def unload_models(self) -> None:
        async with self._lifecycle_lock:
            if self._state is PipelineState.UNLOADED:
                return
            self._state = PipelineState.UNLOADING
            for session in list(self._sessions.values()):
                await self._stop_session(session, reason="shutdown")
            self._sessions.clear()
            self._available_lanes = list(
                range(self._config.max_concurrent_sessions)
            )
            await self._shared_models.close()
            if self._shared_manager is not None:
                await asyncio.to_thread(self._shared_manager.stop)
                self._shared_manager = None
            self._state = PipelineState.UNLOADED

    async def create_session(
        self,
        on_event: EventCallback,
        *,
        input_sample_rate: int = 16000,
    ) -> Session:
        async with self._lock:
            if self._state is not PipelineState.LOADED:
                raise RuntimeError(f"service is not loaded: {self._state.value}")
            if not self._available_lanes:
                raise RuntimeError("max concurrent sessions reached")
            lane_id = self._available_lanes.pop(0)
            started_event = asyncio.Event()
            finished_event = asyncio.Event()
            session = Session(
                lane_id=lane_id,
                controller=SingleProcessPipeline(
                    stream_id=lane_id,
                    app_config=self._config,
                    shared_models=self._shared_models,
                    on_event=on_event,
                    started_event=started_event,
                    finished_event=finished_event,
                    input_sample_rate=input_sample_rate,
                ),
                on_event=on_event,
                started_event=started_event,
                finished_event=finished_event,
            )
            self._sessions[lane_id] = session
            return session

    async def start_session(self, session: Session, *, loop_audio: bool = False) -> None:
        run_dir = (
            self._config.output_dir
            / f"session-{int(time.time() * 1000)}-{session.lane_id}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        # Fresh run: restart if a previous task still exists, otherwise start.
        if session.controller.is_alive():
            await session.controller.restart(run_dir)
        else:
            await session.controller.start(run_dir)
        try:
            await asyncio.wait_for(
                session.started_event.wait(),
                timeout=self._lifecycle_timeout_seconds,
            )
        except asyncio.TimeoutError:
            await self._stop_session(session, reason="start_timeout")
            raise RuntimeError("session start timed out")

    async def stop_session(self, session: Session) -> None:
        async with self._lock:
            await self._stop_session(session, reason="user_requested")

    async def _stop_session(self, session: Session, *, reason: str) -> None:
        if session.lane_id not in self._sessions:
            return
        await session.controller.stop()
        self._sessions.pop(session.lane_id, None)
        if session.lane_id not in self._available_lanes:
            self._available_lanes.append(session.lane_id)
            self._available_lanes.sort()

    def push_audio(self, session: Session, data: bytes | np.ndarray | None) -> None:
        if session.controller.audio_queue is None:
            raise RuntimeError("session does not support streaming audio")
        try:
            session.controller.audio_queue.put_nowait(data)
        except queue.Full:
            _logger.warning("audio queue full for lane %d; dropping audio", session.lane_id)
