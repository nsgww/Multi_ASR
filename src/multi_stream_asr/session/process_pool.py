"""Per-lane subprocess session manager for the Web service.

Each lane is a pre-warmed child process (``LaneProcessController(streaming=True)``)
holding its own full copy of the Zipformer/KWS/speaker models plus its own
Qwen3AsrClient. Sessions borrow a lane from the pool; audio is injected through
the lane's multiprocessing audio queue; events come back over the lane's IPC
event queue and are routed to the owning session's ``on_event`` callback.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from multi_stream_asr.asr.offline.shared_server import SharedQwenServerManager
from multi_stream_asr.config import AppConfig, LaneConfig, ZIPFORMER_WARMUP_AUDIO_PATH
from multi_stream_asr.models import EventStatus
from multi_stream_asr.pipeline.ipc import LaneIpcMessage, asr_event_from_payload
from multi_stream_asr.pipeline.process_manager import LaneProcessController
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.web import messages as msg
from multi_stream_asr.web.event_bus import FrontendEventBus
from multi_stream_asr.web.messages import asr_event_to_frontend

_logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class Session:
    lane_id: int
    controller: LaneProcessController
    on_event: EventCallback
    input_sample_rate: int = 16000
    started_at: float = field(default_factory=time.monotonic)
    started_event: asyncio.Event = field(default_factory=asyncio.Event)
    finished_event: asyncio.Event = field(default_factory=asyncio.Event)
    started: bool = False


class ProcessPoolSessionManager:
    """Duck-type compatible with the old in-process ``SessionManager``."""

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
        self._controllers: dict[int, LaneProcessController] = {}
        self._reader_tasks: dict[int, asyncio.Task[None]] = {}
        self._sessions: dict[int, Session] = {}
        self._available_lanes: list[int] = []
        self._model_loaded: set[int] = set()
        self._loaded: set[int] = set()
        self._kws_loaded = False
        self._load_failure: str | None = None
        self._load_event = asyncio.Event()
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
        return self._kws_loaded

    @property
    def alive_lanes(self) -> int:
        return sum(1 for c in self._controllers.values() if c.is_alive())

    def set_hotwords(self, words: list[str]) -> None:
        command = {"type": "set_hotwords", "words": [str(w) for w in words]}
        for controller in self._controllers.values():
            if controller.is_alive():
                controller.send(command)
        _logger.info("hotwords broadcast to %d lanes", len(self._controllers))

    def reload_kws_keywords(self) -> None:
        if not self._kws_loaded:
            raise RuntimeError("KWS 未加载（可能被配置禁用）")
        for controller in self._controllers.values():
            if controller.is_alive():
                controller.send({"type": "reload_kws"})

    def reload_speaker_registry(self) -> None:
        for controller in self._controllers.values():
            if controller.is_alive():
                controller.send({"type": "reload_speaker"})

    async def load_models(self) -> None:
        async with self._lifecycle_lock:
            if self._state is not PipelineState.UNLOADED:
                raise RuntimeError(f"cannot load models from {self._state.value}")
            self._state = PipelineState.LOADING
            self._load_failure = None
            self._load_event.clear()
            try:
                if not self._fake:
                    self._shared_manager = SharedQwenServerManager(self._config)
                    await asyncio.to_thread(self._shared_manager.start)
                lane_count = self._config.max_concurrent_sessions
                self._controllers = {
                    lane_id: LaneProcessController(
                        app_config=self._config,
                        lane=LaneConfig(stream_id=lane_id),
                        fake=self._fake,
                        streaming=True,
                    )
                    for lane_id in range(lane_count)
                }
                for lane_id, controller in self._controllers.items():
                    controller.start()
                    self._reader_tasks[lane_id] = asyncio.create_task(
                        self._read_lane_events(controller),
                        name=f"asr-lane-reader-{lane_id}",
                    )
                await self._wait_for_load(
                    lambda: len(self._model_loaded) == lane_count,
                    phase="loading_models",
                )
                warmup_path = ZIPFORMER_WARMUP_AUDIO_PATH.resolve()
                for controller in self._controllers.values():
                    controller.send(
                        {"type": "warmup", "input_path": str(warmup_path)}
                    )
                await self._wait_for_load(
                    lambda: len(self._loaded) == lane_count,
                    phase="warming_zipformer",
                )
                self._available_lanes = sorted(self._controllers)
                self._state = PipelineState.LOADED
                _logger.info("process pool loaded: %d lanes ready", lane_count)
            except Exception:
                self._state = PipelineState.FATAL
                await self._shutdown_all_lanes()
                if self._shared_manager is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(self._shared_manager.stop)
                    self._shared_manager = None
                raise

    async def _wait_for_load(self, predicate: Callable[[], bool], *, phase: str) -> None:
        deadline = time.monotonic() + self._lifecycle_timeout_seconds
        while not predicate():
            if self._load_failure is not None:
                raise RuntimeError(f"lane load failed during {phase}: {self._load_failure}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"lane load timed out during {phase}")
            self._load_event.clear()
            try:
                await asyncio.wait_for(self._load_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                raise RuntimeError(f"lane load timed out during {phase}") from None

    async def unload_models(self) -> None:
        async with self._lifecycle_lock:
            if self._state is PipelineState.UNLOADED:
                return
            self._state = PipelineState.UNLOADING
            for session in list(self._sessions.values()):
                await self._stop_session(session, reason="shutdown")
            self._sessions.clear()
            self._available_lanes = []
            await self._shutdown_all_lanes()
            if self._shared_manager is not None:
                await asyncio.to_thread(self._shared_manager.stop)
                self._shared_manager = None
            self._state = PipelineState.UNLOADED

    async def _shutdown_all_lanes(self) -> None:
        for controller in self._controllers.values():
            if controller.is_alive():
                with contextlib.suppress(Exception):
                    controller.send({"type": "shutdown"})
        for controller in self._controllers.values():
            with contextlib.suppress(Exception):
                await controller.terminate_then_kill(
                    terminate_timeout=5.0,
                    kill_timeout=2.0,
                )
            with contextlib.suppress(Exception):
                controller.close_process()
            controller.close_queues()
        for task in self._reader_tasks.values():
            task.cancel()
        if self._reader_tasks:
            await asyncio.gather(*self._reader_tasks.values(), return_exceptions=True)
        self._reader_tasks = {}
        self._controllers = {}

    async def create_session(
        self,
        on_event: EventCallback,
        *,
        input_sample_rate: int = 16000,
    ) -> Session:
        async with self._lock:
            if self._state is not PipelineState.LOADED:
                raise RuntimeError(f"service is not loaded: {self._state.value}")
            while self._available_lanes:
                lane_id = self._available_lanes.pop(0)
                controller = self._controllers.get(lane_id)
                if controller is not None and controller.is_alive():
                    break
                _logger.warning("lane %d process is dead; dropping from pool", lane_id)
            else:
                if not self._sessions and self.alive_lanes == 0:
                    raise RuntimeError(
                        "all lane processes are dead; restart the service"
                    )
                raise RuntimeError(
                    f"max concurrent sessions reached "
                    f"(alive lanes: {self.alive_lanes}, active: {len(self._sessions)})"
                )
            session = Session(
                lane_id=lane_id,
                controller=controller,
                on_event=on_event,
                input_sample_rate=input_sample_rate,
            )
            self._sessions[lane_id] = session
            return session

    async def start_session(self, session: Session, *, loop_audio: bool = False) -> None:
        run_dir = (
            self._config.output_dir
            / f"session-{int(time.time() * 1000)}-{session.lane_id}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        session.started_event.clear()
        session.finished_event.clear()
        session.controller.send(
            {
                "type": "start",
                "run_dir": str(run_dir),
                "input_sample_rate": session.input_sample_rate,
                "loop_audio": loop_audio,
            }
        )
        try:
            await asyncio.wait_for(
                session.started_event.wait(),
                timeout=self._lifecycle_timeout_seconds,
            )
        except asyncio.TimeoutError:
            await self._stop_session(session, reason="start_timeout")
            raise RuntimeError("session start timed out") from None
        session.started = True

    async def stop_session(self, session: Session) -> None:
        async with self._lock:
            await self._stop_session(session, reason="user_requested")

    async def _stop_session(self, session: Session, *, reason: str) -> None:
        if session.lane_id not in self._sessions:
            return
        controller = session.controller
        retired = False
        if (
            session.started
            and not session.finished_event.is_set()
            and controller.is_alive()
        ):
            # 先塞结束帧解阻音频 get()，再发 stop；空闲 lane 收到 stop 会退出。
            with contextlib.suppress(Exception):
                self.push_audio(session, None)
            try:
                controller.send({"type": "stop", "reason": reason})
            except Exception:
                _logger.exception("failed to send stop to lane %d", session.lane_id)
            try:
                await asyncio.wait_for(
                    session.finished_event.wait(),
                    timeout=self._lifecycle_timeout_seconds,
                )
            except asyncio.TimeoutError:
                _logger.warning(
                    "lane %d did not finish within timeout on stop (%s); retiring process",
                    session.lane_id,
                    reason,
                )
                await self._retire_stuck_lane(session.lane_id, controller)
                retired = True
                session.finished_event.set()
        session.started = False
        self._sessions.pop(session.lane_id, None)
        if (
            not retired
            and controller.is_alive()
            and session.lane_id not in self._available_lanes
        ):
            self._available_lanes.append(session.lane_id)
            self._available_lanes.sort()

    async def _retire_stuck_lane(
        self, lane_id: int, controller: LaneProcessController
    ) -> None:
        """超时仍未结束的 lane 直接杀掉，避免脏进程被放回池里卡死下次 start。"""
        reader = self._reader_tasks.pop(lane_id, None)
        if reader is not None and not reader.done():
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
        with contextlib.suppress(Exception):
            controller.send({"type": "shutdown"})
        with contextlib.suppress(Exception):
            await controller.terminate_then_kill(
                terminate_timeout=5.0,
                kill_timeout=2.0,
            )
        with contextlib.suppress(Exception):
            controller.close_process()
        with contextlib.suppress(Exception):
            controller.close_queues()
        self._controllers.pop(lane_id, None)
        if lane_id in self._available_lanes:
            self._available_lanes.remove(lane_id)
        _logger.error(
            "lane %d process retired after stop timeout; "
            "pool capacity is now %d until models are reloaded",
            lane_id,
            self.alive_lanes,
        )

    def push_audio(self, session: Session, data: bytes | np.ndarray | None) -> None:
        audio_queue = session.controller.audio_queue
        if audio_queue is None:
            raise RuntimeError("session does not support streaming audio")
        try:
            audio_queue.put_nowait(data)
        except queue.Full:
            _logger.warning("audio queue full for lane %d; dropping audio", session.lane_id)

    # -- lane event reader -------------------------------------------------

    async def _read_lane_events(self, controller: LaneProcessController) -> None:
        lane_id = controller.lane.stream_id
        while True:
            try:
                raw = await asyncio.to_thread(controller.event_queue.get, True, 0.2)
            except queue.Empty:
                if self._state is PipelineState.UNLOADED:
                    return
                if not controller.is_alive():
                    self._handle_lane_death(lane_id, "lane process exited")
                    return
                continue
            except (EOFError, OSError) as exc:
                self._handle_lane_death(lane_id, f"event queue broken: {exc}")
                return
            except asyncio.CancelledError:
                raise
            try:
                message = LaneIpcMessage.from_dict(raw)
            except Exception:
                _logger.exception("lane %d sent a malformed IPC message", lane_id)
                continue
            self._dispatch(lane_id, message)

    def _handle_lane_death(self, lane_id: int, detail: str) -> None:
        if self._state in (PipelineState.UNLOADING, PipelineState.UNLOADED):
            return
        _logger.error("lane %d died: %s", lane_id, detail)
        if self._state is PipelineState.LOADING:
            self._load_failure = f"lane {lane_id}: {detail}"
            self._load_event.set()
        session = self._sessions.get(lane_id)
        if session is not None:
            session.on_event(msg.error_message(f"lane {lane_id} 进程异常退出", lane_id=lane_id))
            session.started_event.set()
            session.finished_event.set()
        if lane_id in self._available_lanes:
            self._available_lanes.remove(lane_id)

    def _dispatch(self, lane_id: int, message: LaneIpcMessage) -> None:
        session = self._sessions.get(lane_id)
        if message.type == "asr_event":
            try:
                event = asr_event_from_payload(message.payload)
            except Exception:
                _logger.exception("lane %d sent a malformed asr_event", lane_id)
                return
            if event.status is EventStatus.LANE_FINISHED:
                if session is not None:
                    session.on_event(msg.finished_message(lane_id))
                    session.finished_event.set()
                return
            frontend = asr_event_to_frontend(event)
            if frontend is not None and session is not None:
                session.on_event(frontend)
            return
        if message.type == "kws_event":
            if session is not None:
                session.on_event(
                    msg.kws_message(keyword=str(message.payload.get("keyword", "")))
                )
            return
        if message.type == "speaker_remap":
            mapping = message.payload.get("mapping")
            if session is not None and isinstance(mapping, dict):
                session.on_event(
                    msg.speaker_remap_message(mapping, lane_id=lane_id)
                )
            return
        if message.type == "lifecycle":
            self._dispatch_lifecycle(lane_id, message, session)
            return
        if message.type == "fatal":
            error = str(message.payload.get("message", "unknown lane fatal"))
            _logger.error("lane %d fatal: %s", lane_id, error)
            if self._state is PipelineState.LOADING:
                self._load_failure = f"lane {lane_id}: {error}"
                self._load_event.set()
            if session is not None:
                session.on_event(msg.error_message(error, lane_id=lane_id))
                session.finished_event.set()
            return
        # metrics_snapshot / metrics_final / paths / qwen_* ：web 会话不关心

    def _dispatch_lifecycle(
        self,
        lane_id: int,
        message: LaneIpcMessage,
        session: Session | None,
    ) -> None:
        state = str(message.payload.get("state", ""))
        if state == "lane_model_loaded":
            self._model_loaded.add(lane_id)
            if message.payload.get("kws_loaded"):
                self._kws_loaded = True
            self._load_event.set()
        elif state == "lane_loaded":
            self._loaded.add(lane_id)
            self._load_event.set()
        elif state == "lane_started":
            if session is not None:
                session.started_event.set()
        elif state == "lane_finished":
            if session is not None:
                session.finished_event.set()
        elif state in ("lane_failed", "lane_stopped"):
            self._load_event.set()
            if session is not None:
                session.finished_event.set()
