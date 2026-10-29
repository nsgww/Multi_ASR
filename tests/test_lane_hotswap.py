"""通道热改（/api/models/load|unload|validate）接口测试。

Fake 池模拟 ProcessPoolSessionManager 的状态机：load 成功 -> LOADED，
unload -> UNLOADED，可注入加载失败，验证换池语义与 409/422 守卫。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from multi_stream_asr.asr.summary.config import SummaryConfig
from multi_stream_asr.config import AppConfig, ModelConfig
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.session.process_pool import ProcessPoolSessionManager
from multi_stream_asr.web.event_bus import FrontendEventBus
from multi_stream_asr.web.server import create_app


class FakePool(ProcessPoolSessionManager):
    def __init__(self, config: AppConfig, event_bus: FrontendEventBus) -> None:
        self._config = config
        self._bus = event_bus
        self._state = PipelineState.UNLOADED
        self._active_sessions = 0
        self.fail_next_load = False
        self.calls: list[str] = []

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def active_sessions(self) -> int:
        return self._active_sessions

    @property
    def alive_lanes(self) -> int:
        return self._config.lanes if self._state is PipelineState.LOADED else 0

    @property
    def kws_loaded(self) -> bool:
        return False

    async def load_models(self) -> None:
        self.calls.append("load")
        if self.fail_next_load:
            self.fail_next_load = False
            raise RuntimeError("boom")
        self._state = PipelineState.LOADED

    async def unload_models(self) -> None:
        self.calls.append("unload")
        self._state = PipelineState.UNLOADED

    def set_hotwords(self, words: list[str]) -> None:
        self.calls.append("hotwords")

    def reload_kws_keywords(self) -> None:
        self.calls.append("reload_kws")

    def reload_speaker_registry(self) -> None:
        self.calls.append("reload_speaker")


def make_config(tmp_path: Path, lanes: int = 4) -> AppConfig:
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    for filename in (
        "Qwen3-ASR-0.6B-text-q40.gguf",
        "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
        "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
    ):
        (model_dir / filename).write_bytes(b"model")
    return AppConfig(
        lanes=lanes,
        output_dir=tmp_path / "output",
        models=ModelConfig(offline_model_id="qwen3-asr-0.6b-q40", offline_model_dir=model_dir),
        max_concurrent_sessions=lanes,
    )


def make_client(tmp_path: Path, lanes: int = 4, *, fail_next_new_pool: bool = False):
    config = make_config(tmp_path, lanes)
    created: list[FakePool] = []

    def factory(config: AppConfig, bus: FrontendEventBus) -> FakePool:
        pool = FakePool(config, bus)
        # 只让热改时新建的池失败，启动时的第一个池正常加载
        if fail_next_new_pool and len(created) >= 1:
            pool.fail_next_load = True
        created.append(pool)
        return pool

    app = create_app(
        config=config,
        session_manager_factory=factory,
        summary_config=SummaryConfig(enabled=False),
    )
    return app, created


def test_status_payload_reports_lanes_and_supported_lanes(tmp_path: Path) -> None:
    app, created = make_client(tmp_path, lanes=4)
    with TestClient(app) as client:
        payload = client.get("/health").json()
        assert payload["lanes"] == 4
        assert payload["supported_lanes"] == list(range(1, 9))
        assert created[0].state is PipelineState.LOADED


def test_load_same_config_reuses_current_pool(tmp_path: Path) -> None:
    app, created = make_client(tmp_path, lanes=4)
    with TestClient(app) as client:
        response = client.post("/api/models/load", json={})
    assert response.status_code == 200
    assert len(created) == 1
    assert created[0].calls.count("load") == 2  # 启动一次 + 显式一次


def test_load_rebuilds_pool_with_new_lanes(tmp_path: Path) -> None:
    app, created = make_client(tmp_path, lanes=4)
    with TestClient(app) as client:
        unload = client.post("/api/models/unload")
        assert unload.status_code == 200
        assert client.get("/health").json()["state"] == "unloaded"

        response = client.post("/api/models/load", json={"lanes": 6})
        assert response.status_code == 200
        payload = response.json()
        assert len(created) == 2
        assert created[1] is not created[0]
        assert created[1].state is PipelineState.LOADED
        assert payload["lanes"] == 6
        assert payload["alive_lanes"] == 6
        assert created[0].calls[-1] == "unload"
        assert app.state.config.lanes == 6


def test_load_with_changed_lanes_conflicts_while_loaded(tmp_path: Path) -> None:
    app, created = make_client(tmp_path, lanes=4)
    with TestClient(app) as client:
        response = client.post("/api/models/load", json={"lanes": 6})
    assert response.status_code == 409
    assert len(created) == 1
    assert app.state.config.lanes == 4


def test_failed_rebuild_keeps_old_pool(tmp_path: Path) -> None:
    app, created = make_client(tmp_path, lanes=4, fail_next_new_pool=True)
    # 新池加载失败会在服务端抛 RuntimeError：关闭异常上抛拿 500 响应
    with TestClient(app, raise_server_exceptions=False) as client:
        unload = client.post("/api/models/unload")
        assert unload.status_code == 200

        response = client.post("/api/models/load", json={"lanes": 6})
        assert response.status_code == 500
        # 旧池未被替换，配置保持原样
        assert app.state.session_manager is created[0]
        assert app.state.config.lanes == 4
        assert created[0].state is PipelineState.UNLOADED

def test_invalid_lanes_rejected_with_422(tmp_path: Path) -> None:
    app, _ = make_client(tmp_path, lanes=4)
    with TestClient(app) as client:
        for lanes in (0, 9, -1):
            response = client.post("/api/models/load", json={"lanes": lanes})
            assert response.status_code == 422, lanes
        # StrictInt：字符串形式的数字也要拒绝
        response = client.post("/api/models/load", json={"lanes": "6"})
        assert response.status_code == 422
        response = client.post("/api/models/load", json={"offline_model_id": "nope"})
        assert response.status_code == 422


def test_unload_conflicts_with_active_sessions(tmp_path: Path) -> None:
    app, created = make_client(tmp_path, lanes=4)
    with TestClient(app) as client:
        created[0]._active_sessions = 2
        response = client.post("/api/models/unload")
        assert response.status_code == 409
        assert created[0].state is PipelineState.LOADED


def test_validate_models_reports_missing_files(tmp_path: Path) -> None:
    app, _ = make_client(tmp_path, lanes=4)
    with TestClient(app) as client:
        ok = client.post(
            "/api/models/validate",
            json={"lanes": 6, "offline_model_id": "qwen3-asr-0.6b-q40"},
        )
        bad = client.post(
            "/api/models/validate",
            json={"offline_model_dir": str(tmp_path / "missing")},
        )
    assert ok.status_code == 200
    assert ok.json()["valid"] is True
    assert ok.json()["lanes"] == 6
    assert bad.status_code == 200
    assert bad.json()["valid"] is False
    assert bad.json()["missing_files"]
