from __future__ import annotations

import asyncio
import io
import json
import wave
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from multi_stream_asr.asr.summary.config import SummaryConfig
from multi_stream_asr.config import AppConfig, ModelConfig
from multi_stream_asr.models import ASREvent, EventStatus, ResultType
from multi_stream_asr.session.process_pool import ProcessPoolSessionManager, Session
from multi_stream_asr.web.event_bus import FrontendEventBus
from multi_stream_asr.web.server import create_app


class FakeLaneProcessController:
    def __init__(self) -> None:
        self.event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.command_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._alive = True
        self._exitcode: int | None = None

    def start(self) -> None:
        self._alive = True

    def send(self, command: dict[str, Any]) -> None:
        self.command_queue.put_nowait(command)

    def is_alive(self) -> bool:
        return self._alive

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    async def join(self, timeout: float) -> int | None:
        self._alive = False
        return self._exitcode

    async def terminate_then_kill(self, *, terminate_timeout: float, kill_timeout: float) -> bool:
        self._alive = False
        return True

    def close_queues(self) -> None:
        pass

    def close_process(self) -> None:
        self._alive = False


class FakeSessionManager(ProcessPoolSessionManager):
    def __init__(self, config: AppConfig, event_bus: FrontendEventBus) -> None:
        self._config = config
        self._event_bus = event_bus
        self._state = "loaded"
        self._sessions: dict[int, Session] = {}
        self._controllers: dict[int, Any] = {}
        self._kws_loaded = False
        self._next_lane = 0
        self.calls: list[str] = []

    @property
    def state(self):
        from multi_stream_asr.pipeline.state import PipelineState
        return PipelineState.LOADED

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    async def load_models(self) -> None:
        self.calls.append("load_models")

    async def unload_models(self) -> None:
        self.calls.append("unload_models")

    async def create_session(self, on_event, *, input_sample_rate: int = 16000) -> Session:
        lane_id = self._next_lane
        self._next_lane += 1
        controller = FakeLaneProcessController()
        session = Session(lane_id=lane_id, controller=controller, on_event=on_event)
        self._sessions[lane_id] = session
        return session

    async def start_session(self, session: Session, *, loop_audio: bool = False) -> None:
        self.calls.append(f"start_session:{session.lane_id}")
        session.started_event.set()
        session.on_event({"type": "status", "status": "started", "lane_id": session.lane_id})

    async def stop_session(self, session: Session) -> None:
        self.calls.append(f"stop_session:{session.lane_id}")
        self._sessions.pop(session.lane_id, None)

    def push_audio(self, session: Session, data: bytes | None) -> None:
        if isinstance(session.controller, FakeLaneProcessController):
            session.controller.audio_queue.put_nowait(data)
        if data is None:
            session.finished_event.set()
            session.on_event({"type": "finished", "lane_id": session.lane_id})


def app_config_with_model_files(tmp_path: Path) -> AppConfig:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for filename in (
        "Qwen3-ASR-0.6B-text-q40.gguf",
        "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
        "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
    ):
        (model_dir / filename).write_bytes(b"model")
    models = ModelConfig(
        offline_model_id="qwen3-asr-0.6b-q40",
        offline_model_dir=model_dir,
    )
    return AppConfig(
        lanes=2,
        output_dir=tmp_path / "output",
        models=models,
        web_port=8060,
        max_concurrent_sessions=2,
    )


def write_valid_wav(path: Path, *, duration_ms: int = 200) -> None:
    samples = duration_ms * 16  # 16 samples/ms @ 16kHz
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\0\0" * samples)


def test_health_returns_service_state(tmp_path: Path) -> None:
    config = app_config_with_model_files(tmp_path)
    manager = FakeSessionManager(config, FrontendEventBus())
    app = create_app(
        config=config,
        session_manager=manager,
        summary_config=SummaryConfig(enabled=False),
    )
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "loaded"
    assert payload["active_sessions"] == 0
    assert payload["max_sessions"] == 2


def test_index_returns_api_info(tmp_path: Path) -> None:
    config = app_config_with_model_files(tmp_path)
    app = create_app(
        config=config,
        session_manager=FakeSessionManager(config, FrontendEventBus()),
        summary_config=SummaryConfig(enabled=False),
    )
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "multi-stream-asr"


def test_websocket_stream_protocol(tmp_path: Path) -> None:
    config = app_config_with_model_files(tmp_path)
    manager = FakeSessionManager(config, FrontendEventBus())
    app = create_app(
        config=config,
        session_manager=manager,
        summary_config=SummaryConfig(enabled=False),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "status"
            assert ready["status"] == "ready"
            assert ready["lane_id"] == 0

            ws.send_json({"type": "config", "sample_rate": 16000, "channels": 1, "width": 2})
            config_ok = ws.receive_json()
            assert config_ok["type"] == "config_ok"

            ws.send_json({"type": "start"})
            started = ws.receive_json()
            assert started["type"] == "status"
            assert started["status"] == "started"

            ws.send_bytes(b"\0\0" * 1600)


def test_download_transcript_not_found(tmp_path: Path) -> None:
    config = app_config_with_model_files(tmp_path)
    app = create_app(
        config=config,
        session_manager=FakeSessionManager(config, FrontendEventBus()),
        summary_config=SummaryConfig(enabled=False),
    )
    with TestClient(app) as client:
        response = client.get("/v1/transcripts/missing.txt")
    assert response.status_code == 404


def test_download_transcript_returns_file(tmp_path: Path) -> None:
    config = app_config_with_model_files(tmp_path)
    transcript_dir = config.output_dir / "transcripts"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "test.txt").write_text("hello world", encoding="utf-8")

    app = create_app(
        config=config,
        session_manager=FakeSessionManager(config, FrontendEventBus()),
        summary_config=SummaryConfig(enabled=False),
    )
    with TestClient(app) as client:
        response = client.get("/v1/transcripts/test.txt")
    assert response.status_code == 200
    assert response.text == "hello world"


def test_regenerate_summary_endpoint(tmp_path: Path) -> None:
    config = app_config_with_model_files(tmp_path)
    transcript_dir = config.output_dir / "transcripts"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "test.txt").write_text("hello world", encoding="utf-8")

    app = create_app(
        config=config,
        session_manager=FakeSessionManager(config, FrontendEventBus()),
        summary_config=SummaryConfig(enabled=False),
    )
    with TestClient(app) as client:
        response = client.post("/v1/transcripts/test.txt/summary", json={})
    assert response.status_code == 503
    assert response.json()["provider"] == "disabled"


def test_audio_transcriptions_rejects_when_not_loaded(tmp_path: Path) -> None:
    from multi_stream_asr.pipeline.state import PipelineState

    class UnloadedManager(FakeSessionManager):
        @property
        def state(self):
            return PipelineState.UNLOADED

    config = app_config_with_model_files(tmp_path)
    app = create_app(
        config=config,
        session_manager=UnloadedManager(config, FrontendEventBus()),
        summary_config=SummaryConfig(enabled=False),
    )
    wav_path = tmp_path / "input.wav"
    write_valid_wav(wav_path)
    with TestClient(app) as client:
        with open(wav_path, "rb") as f:
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("input.wav", f, "audio/wav")},
            )
    assert response.status_code == 503
