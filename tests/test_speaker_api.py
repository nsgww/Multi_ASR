from __future__ import annotations

import io
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from multi_stream_asr.asr.summary.config import SummaryConfig
from multi_stream_asr.config import AppConfig, ModelConfig, SpeakerConfig
from multi_stream_asr.web.event_bus import FrontendEventBus
from multi_stream_asr.web.server import create_app

from test_frontend_api import FakeSessionManager


class FakeEmbedder:
    """固定维度假声纹：注册接口只关心它不是 None。"""

    available = True
    last_error = None

    def __init__(self, dim: int = 192) -> None:
        self.dim = dim

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        vec = np.ones(self.dim, dtype=np.float32)
        return vec / np.linalg.norm(vec)


class NoneEmbedder(FakeEmbedder):
    def embed(self, pcm: np.ndarray) -> None:
        return None


class ReloadRecordingManager(FakeSessionManager):
    def __init__(self, config: AppConfig, event_bus: FrontendEventBus) -> None:
        super().__init__(config, event_bus)
        self.speaker_reloads = 0

    def reload_speaker_registry(self) -> None:
        self.speaker_reloads += 1


def _make_config(tmp_path: Path, *, enabled: bool = True) -> AppConfig:
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
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
        speaker=SpeakerConfig(
            enabled=enabled,
            model_path=tmp_path / "no-such-model.onnx",
            registry_dir=tmp_path / "voiceprints",
        ),
    )


def _make_app(config: AppConfig):
    manager = ReloadRecordingManager(config, FrontendEventBus())
    app = create_app(
        config=config,
        session_manager=manager,
        summary_config=SummaryConfig(enabled=False),
    )
    app.state.speaker_embedder = FakeEmbedder()
    return app, manager


def _wav_bytes(duration_ms: int = 500) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x01\x00" * duration_ms * 16)
    return buffer.getvalue()


def test_enroll_list_delete_roundtrip(tmp_path: Path) -> None:
    app, manager = _make_app(_make_config(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/speakers",
            data={"name": "张三"},
            files=[("files", ("a.wav", _wav_bytes(), "audio/wav"))],
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload == {"ok": True, "name": "张三", "enrolled": 1, "total": 1}
        assert manager.speaker_reloads == 1

        listed = client.get("/v1/speakers").json()
        assert listed["enabled"] is True
        assert listed["speakers"] == [{"name": "张三", "count": 1}]

        response = client.delete("/v1/speakers/张三")
        assert response.status_code == 200
        assert manager.speaker_reloads == 2
        assert client.get("/v1/speakers").json()["speakers"] == []


def test_enroll_multiple_files_accumulate(tmp_path: Path) -> None:
    app, _ = _make_app(_make_config(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/speakers",
            data={"name": "李四"},
            files=[
                ("files", ("a.wav", _wav_bytes(), "audio/wav")),
                ("files", ("b.wav", _wav_bytes(), "audio/wav")),
            ],
        )
        assert response.json()["total"] == 2


def test_delete_missing_returns_404(tmp_path: Path) -> None:
    app, _ = _make_app(_make_config(tmp_path))
    with TestClient(app) as client:
        assert client.delete("/v1/speakers/不存在").status_code == 404


def test_enroll_rejects_bad_wav(tmp_path: Path) -> None:
    app, _ = _make_app(_make_config(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/speakers",
            data={"name": "王五"},
            files=[("files", ("bad.wav", b"not a wav", "audio/wav"))],
        )
        assert response.status_code == 400


def test_enroll_rejects_unembeddable_audio(tmp_path: Path) -> None:
    app, _ = _make_app(_make_config(tmp_path))
    app.state.speaker_embedder = NoneEmbedder()
    with TestClient(app) as client:
        response = client.post(
            "/v1/speakers",
            data={"name": "王五"},
            files=[("files", ("a.wav", _wav_bytes(), "audio/wav"))],
        )
        assert response.status_code == 400
        assert "过短" in response.json()["detail"]


def test_speaker_endpoints_503_when_disabled(tmp_path: Path) -> None:
    app, _ = _make_app(_make_config(tmp_path, enabled=False))
    with TestClient(app) as client:
        response = client.post(
            "/v1/speakers",
            data={"name": "张三"},
            files=[("files", ("a.wav", _wav_bytes(), "audio/wav"))],
        )
        assert response.status_code == 503
        assert client.delete("/v1/speakers/张三").status_code == 503
        listed = client.get("/v1/speakers").json()
        assert listed["enabled"] is False
        assert listed["speakers"] == []


def test_registry_persists_across_app_instances(tmp_path: Path) -> None:
    """注册落盘后，重建 app（模拟重启）仍能列出同一批声纹。"""
    config = _make_config(tmp_path)
    app, _ = _make_app(config)
    with TestClient(app) as client:
        client.post(
            "/v1/speakers",
            data={"name": "张三"},
            files=[("files", ("a.wav", _wav_bytes(), "audio/wav"))],
        )

    app2, _ = _make_app(config)
    with TestClient(app2) as client:
        listed = client.get("/v1/speakers").json()
        assert listed["speakers"] == [{"name": "张三", "count": 1}]
