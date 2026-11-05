"""ProcessPoolSessionManager 单元测试：用假 lane 控制器覆盖池分配/事件路由/广播。"""

from __future__ import annotations

import asyncio
import queue
from pathlib import Path
from typing import Any

import pytest

from multi_stream_asr.config import AppConfig, LaneConfig, SharedQwenConfig
from multi_stream_asr.models import ASREvent, EventStatus
from multi_stream_asr.pipeline.ipc import asr_event_to_payload
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.session import process_pool
from multi_stream_asr.session.process_pool import ProcessPoolSessionManager
from multi_stream_asr.web.event_bus import FrontendEventBus


class FakeLaneController:
    """不落真进程的 LaneProcessController 替身（queue.Queue 版 IPC）。"""

    def __init__(
        self,
        *,
        app_config: AppConfig,
        lane: LaneConfig,
        fake: bool,
        streaming: bool = False,
    ) -> None:
        self.app_config = app_config
        self.lane = lane
        self.fake = fake
        self.streaming = streaming
        self.event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.command_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.audio_queue: queue.Queue[Any] = queue.Queue(maxsize=100)
        self.commands: list[dict[str, Any]] = []
        self._alive = True
        self._seq = 0

    def start(self) -> None:
        self.emit_lifecycle("lane_model_loaded", kws_loaded=True)

    def send(self, command: dict[str, Any]) -> None:
        self.commands.append(dict(command))
        if command["type"] == "warmup":
            self.emit_lifecycle("lane_loaded")

    def emit(self, message_type: str, payload: dict[str, Any]) -> None:
        self._seq += 1
        self.event_queue.put(
            {
                "type": message_type,
                "stream_id": self.lane.stream_id,
                "seq": self._seq,
                "time_ms": 1,
                "payload": payload,
            }
        )

    def emit_lifecycle(self, state: str, **extra: Any) -> None:
        payload = {"state": state, "message": ""}
        payload.update(extra)
        self.emit("lifecycle", payload)

    def emit_asr(self, status: EventStatus, text: str = "你好") -> None:
        event = ASREvent.asr_result(
            stream_id=self.lane.stream_id,
            sentence_id=0,
            revision=0,
            status=status,
            text=text,
            timestamp_ms=1,
        )
        self.emit("asr_event", asr_event_to_payload(event))

    def is_alive(self) -> bool:
        return self._alive

    async def terminate_then_kill(
        self, *, terminate_timeout: float, kill_timeout: float
    ) -> bool:
        self._alive = False
        return True

    def close_queues(self) -> None:
        pass

    def close_process(self) -> None:
        self._alive = False


@pytest.fixture()
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(process_pool, "LaneProcessController", FakeLaneController)
    config = AppConfig(max_concurrent_sessions=2, output_dir=tmp_path / "out")
    return ProcessPoolSessionManager(config, FrontendEventBus(), fake=True)


def controllers_of(manager: ProcessPoolSessionManager) -> dict[int, FakeLaneController]:
    return manager._controllers  # type: ignore[return-value]


async def test_load_models_spawns_and_warms_all_lanes(manager) -> None:
    await manager.load_models()

    assert manager.state is PipelineState.LOADED
    assert manager.kws_loaded is True
    assert len(controllers_of(manager)) == 2
    for controller in controllers_of(manager).values():
        assert any(c["type"] == "warmup" for c in controller.commands)

    await manager.unload_models()
    assert manager.state is PipelineState.UNLOADED


async def test_lane_allocation_reuse_and_cap(manager) -> None:
    await manager.load_models()
    events: list[dict[str, Any]] = []

    s0 = await manager.create_session(events.append)
    s1 = await manager.create_session(events.append)
    assert (s0.lane_id, s1.lane_id) == (0, 1)
    with pytest.raises(RuntimeError, match="max concurrent sessions"):
        await manager.create_session(events.append)

    # 未 start 的会话 stop 不应向 lane 发 stop（空闲 lane 收到 stop 会退出）
    await manager.stop_session(s0)
    assert not any(
        c["type"] == "stop" for c in controllers_of(manager)[0].commands
    )

    s2 = await manager.create_session(events.append)
    assert s2.lane_id == 0  # 归还后取最小空闲号

    await manager.unload_models()


async def _start_and_signal_started(manager, session) -> None:
    """start_session 会等 lane_started；测试里由假控制器异步补发。"""
    controller = controllers_of(manager)[session.lane_id]
    task = asyncio.create_task(manager.start_session(session))
    for _ in range(100):
        if any(c["type"] == "start" for c in controller.commands):
            break
        await asyncio.sleep(0.02)
    controller.emit_lifecycle("lane_started")
    await asyncio.wait_for(task, timeout=5)


async def test_event_routing_to_session(manager) -> None:
    await manager.load_models()
    events: list[dict[str, Any]] = []
    session = await manager.create_session(events.append, input_sample_rate=8000)
    controller = controllers_of(manager)[session.lane_id]

    start_task = asyncio.create_task(manager.start_session(session))
    for _ in range(100):
        if any(c["type"] == "start" for c in controller.commands):
            break
        await asyncio.sleep(0.02)
    start_cmd = next(c for c in controller.commands if c["type"] == "start")
    assert start_cmd["input_sample_rate"] == 8000
    assert "run_dir" in start_cmd
    controller.emit_lifecycle("lane_started")
    await asyncio.wait_for(start_task, timeout=5)

    controller.emit_asr(EventStatus.FINAL, text="最终结果")
    controller.emit("kws_event", {"keyword": "你好小联"})
    controller.emit("speaker_remap", {"mapping": {"S1": "S2"}})
    controller.emit_asr(EventStatus.LANE_FINISHED)

    await asyncio.wait_for(session.finished_event.wait(), timeout=2)
    types = [e["type"] for e in events]
    assert "transcript" in types
    assert "kws" in types
    assert "speaker_remap" in types
    assert types[-1] == "finished"
    transcript = next(e for e in events if e["type"] == "transcript")
    assert transcript["segment"]["text"] == "最终结果"
    assert transcript["lane_id"] == session.lane_id

    await manager.unload_models()


async def test_stop_session_sends_stop_and_releases_lane(manager) -> None:
    await manager.load_models()
    session = await manager.create_session(lambda e: None)
    controller = controllers_of(manager)[session.lane_id]
    await _start_and_signal_started(manager, session)

    stop_task = asyncio.create_task(manager.stop_session(session))
    await asyncio.sleep(0)  # 让 stop 命令先发出
    for _ in range(50):
        if any(c["type"] == "stop" for c in controller.commands):
            break
        await asyncio.sleep(0.02)
    assert any(c["type"] == "stop" for c in controller.commands)
    # stop 前会先塞 None 解阻音频队列
    assert controller.audio_queue.get_nowait() is None
    controller.emit_lifecycle("lane_finished")
    await asyncio.wait_for(stop_task, timeout=2)

    assert manager.active_sessions == 0
    assert session.lane_id in manager._available_lanes

    await manager.unload_models()


async def test_stop_session_timeout_retires_stuck_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_pool, "LaneProcessController", FakeLaneController)
    config = AppConfig(
        max_concurrent_sessions=2,
        output_dir=tmp_path / "out",
        shared_qwen=SharedQwenConfig(request_timeout_ms=20),
    )
    manager = ProcessPoolSessionManager(config, FrontendEventBus(), fake=True)
    manager._lifecycle_timeout_seconds = 0.05
    await manager.load_models()
    session = await manager.create_session(lambda e: None)
    lane_id = session.lane_id
    controller = controllers_of(manager)[lane_id]
    await _start_and_signal_started(manager, session)

    # 故意不发 lane_finished，触发超时强杀
    await asyncio.wait_for(manager.stop_session(session), timeout=5)

    assert manager.active_sessions == 0
    assert lane_id not in manager._available_lanes
    assert lane_id not in manager._controllers
    assert controller.is_alive() is False
    assert manager.alive_lanes == 1

    await manager.unload_models()


async def test_lane_failed_sets_finished_event(manager) -> None:
    await manager.load_models()
    session = await manager.create_session(lambda e: None)
    controller = controllers_of(manager)[session.lane_id]
    await _start_and_signal_started(manager, session)

    controller.emit_lifecycle("lane_failed", message="boom")
    await asyncio.wait_for(session.finished_event.wait(), timeout=2)

    await manager.unload_models()


async def test_set_hotwords_and_reload_kws_broadcast(manager) -> None:
    await manager.load_models()

    manager.set_hotwords(["朗吉", "小联"])
    manager.reload_kws_keywords()

    for controller in controllers_of(manager).values():
        assert {"type": "set_hotwords", "words": ["朗吉", "小联"]} in controller.commands
        assert {"type": "reload_kws"} in controller.commands

    await manager.unload_models()


async def test_reload_speaker_registry_broadcast(manager) -> None:
    await manager.load_models()

    manager.reload_speaker_registry()

    for controller in controllers_of(manager).values():
        assert {"type": "reload_speaker"} in controller.commands

    await manager.unload_models()


async def test_reload_kws_raises_when_not_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoKwsController(FakeLaneController):
        def start(self) -> None:
            self.emit_lifecycle("lane_model_loaded", kws_loaded=False)

    monkeypatch.setattr(process_pool, "LaneProcessController", NoKwsController)
    config = AppConfig(max_concurrent_sessions=1, output_dir=tmp_path / "out")
    manager = ProcessPoolSessionManager(config, FrontendEventBus(), fake=True)
    await manager.load_models()

    assert manager.kws_loaded is False
    with pytest.raises(RuntimeError):
        manager.reload_kws_keywords()

    await manager.unload_models()
