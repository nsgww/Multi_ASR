from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import queue
import threading
import wave
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from multi_stream_asr.asr.offline.shared_server import SHARED_LOG_PATH
from multi_stream_asr.config import (
    AppConfig,
    LaneInputConfig,
    ZIPFORMER_WARMUP_AUDIO_PATH,
)
from multi_stream_asr.models import ASREvent, EventStatus, LaneFinalState, LaneMetrics
from multi_stream_asr.log.qwen_errors import QwenFailureDecision
from multi_stream_asr.log.recorder import RunRecorder
from multi_stream_asr.pipeline.ipc import asr_event_to_payload, lane_metrics_to_payload
from multi_stream_asr.pipeline.process_manager import (
    LaneProcessController,
    ProcessPipelineManager,
)
from multi_stream_asr.pipeline.ipc import make_message
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.web.event_bus import FrontendEventBus


class _ScriptedController:
    instances: list["_ScriptedController"] = []
    actions: list[str] = []
    start_scripts: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    start_command_scripts: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    start_command_exit_codes: dict[int, int] = {}
    silent_exit_codes: dict[int, int] = {}

    def __init__(self, *, app_config: AppConfig, lane: Any, fake: bool) -> None:
        self.app_config = app_config
        self.lane = lane
        self.fake = fake
        self.event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.commands: list[dict[str, Any]] = []
        self._alive = False
        self._exitcode: int | None = None
        self._seq = 0
        self.closed = False
        self.process_closed = False
        type(self).instances.append(self)

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    def start(self) -> None:
        self._alive = True
        type(self).actions.append(f"start:{self.lane.stream_id}")
        if self.lane.stream_id in type(self).silent_exit_codes:
            self._alive = False
            self._exitcode = type(self).silent_exit_codes[self.lane.stream_id]
            return
        script = type(self).start_scripts.get(
            self.lane.stream_id,
            [("lifecycle", {"state": "lane_model_loaded", "message": ""})],
        )
        for message_type, payload in script:
            self.emit(message_type, payload)

    def send(self, command: dict[str, Any]) -> None:
        self.commands.append(command)
        if command["type"] == "warmup":
            self.emit("lifecycle", {"state": "lane_loaded", "message": ""})
            return
        if command["type"] == "start":
            script = type(self).start_command_scripts.get(
                self.lane.stream_id,
                [("lifecycle", {"state": "lane_started", "message": ""})],
            )
            for message_type, payload in script:
                self.emit(message_type, payload)
            if self.lane.stream_id in type(self).start_command_exit_codes:
                self._alive = False
                self._exitcode = type(self).start_command_exit_codes[self.lane.stream_id]
            return
        if command["type"] == "stop":
            self.emit(
                "metrics_final",
                {
                    "lane_metrics": lane_metrics_to_payload(
                        LaneMetrics(
                            stream_id=self.lane.stream_id,
                            final_state=LaneFinalState.STOPPED,
                        )
                    )
                },
            )
            self.emit(
                "paths",
                {
                    "results_path": f"results/lane{self.lane.stream_id}.txt",
                    "model_outputs_path": f"model_outputs/lane{self.lane.stream_id}.jsonl",
                },
            )
            self.emit("lifecycle", {"state": "lane_finished", "message": ""})
            return
        if command["type"] == "shutdown":
            self.emit("lifecycle", {"state": "lane_stopped", "message": ""})
            self._alive = False
            self._exitcode = 0

    def emit(self, message_type: str, payload: dict[str, Any]) -> None:
        self._seq += 1
        self.event_queue.put(
            make_message(
                message_type=message_type,
                stream_id=self.lane.stream_id,
                seq=self._seq,
                payload=payload,
            ).to_dict()
        )

    def is_alive(self) -> bool:
        return self._alive

    async def join(self, timeout: float) -> int | None:
        return self._exitcode

    async def terminate_then_kill(self, *, terminate_timeout: float, kill_timeout: float) -> bool:
        return False

    def close_queues(self) -> None:
        self.closed = True

    def close_process(self) -> None:
        assert not self.is_alive()
        self.process_closed = True


class _NoopSharedQwenServerManager:
    instances: list["_NoopSharedQwenServerManager"] = []

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.alive = False
        self.code: int | None = None
        type(self).instances.append(self)

    @property
    def exit_code(self) -> int | None:
        return self.code

    def start(self) -> None:
        SHARED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        SHARED_LOG_PATH.write_text("", encoding="utf-8")
        self.alive = True

    def stop(self) -> None:
        self.alive = False
        if self.code is None:
            self.code = 0

    def is_alive(self) -> bool:
        return self.alive


class _RecordingQwenErrors:
    def __init__(self, decisions: list[QwenFailureDecision]) -> None:
        self.decisions = iter(decisions)
        self.successes = 0
        self.failures: list[dict[str, Any]] = []
        self.started: list[Path] = []
        self.finished = 0
        self.reset_load = 0
        self.reset_unload = 0

    def reset_for_model_load(self) -> None:
        self.reset_load += 1

    def start_run(self, run_dir: Path) -> None:
        self.started.append(run_dir)

    def record_success(self) -> None:
        self.successes += 1

    def record_failure(self, **kwargs: Any) -> QwenFailureDecision:
        self.failures.append(kwargs)
        return next(self.decisions)

    def finish_run(self) -> None:
        self.finished += 1

    def reset_for_unload(self) -> None:
        self.reset_unload += 1


def _write_valid_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\0\0")


@pytest.fixture(autouse=True)
def reset_scripted_controller() -> None:
    _ScriptedController.instances.clear()
    _ScriptedController.actions.clear()
    _ScriptedController.start_scripts.clear()
    _ScriptedController.start_command_scripts.clear()
    _ScriptedController.start_command_exit_codes.clear()
    _ScriptedController.silent_exit_codes.clear()
    _NoopSharedQwenServerManager.instances.clear()


class _RecordingProcess:
    def __init__(self, *, target: Any, args: tuple[Any, ...], name: str) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.exitcode: int | None = None
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and self.exitcode is None


class _RecordingContext:
    def __init__(self) -> None:
        self.queues: list[object] = []
        self.processes: list[_RecordingProcess] = []

    def Queue(self) -> object:
        queue = object()
        self.queues.append(queue)
        return queue

    def Process(self, *, target: Any, args: tuple[Any, ...], name: str) -> _RecordingProcess:
        process = _RecordingProcess(target=target, args=args, name=name)
        self.processes.append(process)
        return process


def test_lane_process_controller_uses_spawn_and_lane_name(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _RecordingContext()
    requested_methods: list[str] = []

    def get_context(method: str) -> _RecordingContext:
        requested_methods.append(method)
        return context

    monkeypatch.setattr(mp, "get_context", get_context)
    config = AppConfig(lanes=2)
    controller = LaneProcessController(
        app_config=config,
        lane=config.lane_configs[1],
        fake=True,
    )

    controller.start()

    assert requested_methods == ["spawn"]
    assert controller.process is context.processes[0]
    assert controller.process.name == "asr-lane-1"
    assert controller.process.started is True


def test_process_pipeline_manager_is_exported_from_pipeline_package() -> None:
    from multi_stream_asr.pipeline import ProcessPipelineManager as ExportedManager

    assert ExportedManager is ProcessPipelineManager


@pytest.mark.asyncio
async def test_manager_waits_for_all_models_then_all_warmups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class ManualWarmupController(_ScriptedController):
        def start(self) -> None:
            self._alive = True
            type(self).actions.append(f"start:{self.lane.stream_id}")

        def send(self, command: dict[str, Any]) -> None:
            self.commands.append(command)
            if command["type"] in {"stop", "shutdown"}:
                self.emit("lifecycle", {"state": "lane_stopped", "message": ""})
                self._alive = False
                self._exitcode = 0

    async def wait_until(predicate: Any) -> None:
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("condition was not reached")

    monkeypatch.setattr(
        process_manager, "LaneProcessController", ManualWarmupController
    )
    config = AppConfig(
        lanes=2,
        lane_inputs=(
            LaneInputConfig(kind="microphone", microphone_device=3),
            LaneInputConfig(
                kind="local_wav",
                local_wav_path=Path("/must-not-be-used-for-warmup.wav"),
            ),
        ),
    )
    manager = ProcessPipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=1.0,
    )
    load_task = asyncio.create_task(manager.load_models())
    try:
        await wait_until(lambda: len(ManualWarmupController.instances) == 2)
        controllers = sorted(
            ManualWarmupController.instances,
            key=lambda controller: controller.lane.stream_id,
        )
        controllers[0].emit(
            "lifecycle", {"state": "lane_model_loaded", "message": ""}
        )
        await asyncio.sleep(0.02)
        assert all(not controller.commands for controller in controllers)

        controllers[1].emit(
            "lifecycle", {"state": "lane_model_loaded", "message": ""}
        )
        await wait_until(
            lambda: all(
                any(command["type"] == "warmup" for command in controller.commands)
                for controller in controllers
            )
        )
        assert manager.state is PipelineState.LOADING
        assert manager.load_phase == "warming_zipformer"
        assert all(
            controller.commands[-1]
            == {
                "type": "warmup",
                "input_path": str(ZIPFORMER_WARMUP_AUDIO_PATH),
            }
            for controller in controllers
        )

        controllers[0].emit(
            "lifecycle", {"state": "lane_loaded", "message": ""}
        )
        await asyncio.sleep(0.02)
        assert not load_task.done()
        controllers[1].emit(
            "lifecycle", {"state": "lane_loaded", "message": ""}
        )
        await load_task
        assert manager.state is PipelineState.LOADED
        assert manager.load_phase is None
    finally:
        if not load_task.done():
            load_task.cancel()
        with suppress(BaseException):
            await load_task
        if manager.state is not PipelineState.UNLOADED:
            with suppress(BaseException):
                await manager.unload_models()


@pytest.mark.asyncio
async def test_fake_manager_uses_warmup_barrier_without_reading_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager,
        "validate_wav_16k_mono_pcm",
        lambda _path: (_ for _ in ()).throw(AssertionError("fake read warmup wav")),
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.01,
    )

    await manager.load_models()

    assert manager.state is PipelineState.LOADED
    assert all(
        any(command["type"] == "warmup" for command in controller.commands)
        for controller in _ScriptedController.instances
    )
    await manager.unload_models()


@pytest.mark.asyncio
async def test_warmup_fatal_cleans_all_lanes_and_shared_qwen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class WarmupFailingController(_ScriptedController):
        def start(self) -> None:
            self._alive = True
            type(self).actions.append(f"start:{self.lane.stream_id}")
            self.emit(
                "lifecycle", {"state": "lane_model_loaded", "message": ""}
            )

        def send(self, command: dict[str, Any]) -> None:
            self.commands.append(command)
            if command["type"] == "warmup":
                self.emit("fatal", {"message": "zipformer warmup failed"})
                self._alive = False
                self._exitcode = 1
            elif command["type"] in {"stop", "shutdown"}:
                self.emit("lifecycle", {"state": "lane_stopped", "message": ""})
                self._alive = False
                self._exitcode = 0

    monkeypatch.setattr(
        process_manager, "LaneProcessController", WarmupFailingController
    )
    monkeypatch.setattr(
        process_manager,
        "SharedQwenServerManager",
        _NoopSharedQwenServerManager,
    )
    event_bus = FrontendEventBus()
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2),
        event_bus=event_bus,
        fake=False,
        lifecycle_timeout_seconds=1.0,
    )

    with pytest.raises(RuntimeError, match="zipformer warmup failed"):
        await manager.load_models()

    assert manager.state is PipelineState.FATAL
    assert manager.load_phase is None
    assert event_bus.snapshot_results() == {}
    assert _NoopSharedQwenServerManager.instances
    assert not _NoopSharedQwenServerManager.instances[0].is_alive()


@pytest.mark.asyncio
async def test_nonfake_manager_starts_shared_qwen_before_lanes_and_stops_after_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class FakeSharedQwenServerManager:
        def __init__(self, config: AppConfig) -> None:
            self.config = config
            self.alive = False
            self.code: int | None = None

        @property
        def exit_code(self) -> int | None:
            return self.code

        def start(self) -> None:
            SHARED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            SHARED_LOG_PATH.write_text("", encoding="utf-8")
            self.alive = True
            _ScriptedController.actions.append("shared:start")

        def stop(self) -> None:
            self.alive = False
            self.code = 0
            _ScriptedController.actions.append("shared:stop")

        def is_alive(self) -> bool:
            return self.alive

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager,
        "SharedQwenServerManager",
        FakeSharedQwenServerManager,
    )
    config = AppConfig(lanes=2, output_dir=tmp_path)
    manager = ProcessPipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.5,
    )

    await manager.load_models()

    assert manager.state is PipelineState.LOADED
    assert _ScriptedController.actions == ["shared:start", "start:0", "start:1"]

    await manager.unload_models()
    assert _ScriptedController.actions[-1] == "shared:stop"
    assert all(controller.closed for controller in _ScriptedController.instances)


@pytest.mark.asyncio
async def test_shared_qwen_exit_marks_manager_fatal_and_stops_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class FakeSharedQwenServerManager:
        instance: "FakeSharedQwenServerManager | None" = None

        def __init__(self, config: AppConfig) -> None:
            self.alive = False
            self.code: int | None = None
            type(self).instance = self

        @property
        def exit_code(self) -> int | None:
            return self.code

        def start(self) -> None:
            SHARED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            SHARED_LOG_PATH.write_text("", encoding="utf-8")
            self.alive = True

        def stop(self) -> None:
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", FakeSharedQwenServerManager
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.005,
        shared_watch_interval_seconds=0.001,
    )

    await manager.load_models()
    shared = FakeSharedQwenServerManager.instance
    assert shared is not None
    shared.alive = False
    shared.code = 137
    for _ in range(100):
        if manager.state is PipelineState.FATAL:
            break
        await asyncio.sleep(0.001)

    assert manager.state is PipelineState.FATAL
    assert "shared qwen" in str(manager._failure)
    assert "137" in str(manager._failure)
    assert all(
        {"type": "stop", "reason": "sibling_fatal"} in controller.commands
        for controller in _ScriptedController.instances
    )
    await manager.unload_models()
    assert manager.state is PipelineState.UNLOADED


@pytest.mark.asyncio
async def test_shared_qwen_start_failure_does_not_spawn_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class FailingSharedQwenServerManager(_NoopSharedQwenServerManager):
        def start(self) -> None:
            raise RuntimeError("shared startup failed")

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager,
        "SharedQwenServerManager",
        FailingSharedQwenServerManager,
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.2,
    )

    with pytest.raises(RuntimeError, match="shared startup failed"):
        await manager.load_models()

    assert _ScriptedController.instances == []
    assert manager.state is PipelineState.FATAL
    assert FailingSharedQwenServerManager.instances[0].alive is False
    await manager.unload_models()


@pytest.mark.asyncio
async def test_fake_manager_does_not_construct_shared_qwen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    def fail_shared_construction(config: AppConfig) -> None:
        raise AssertionError("fake mode must not construct shared qwen")

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", fail_shared_construction
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.2,
    )

    await manager.load_models()
    assert manager.state is PipelineState.LOADED
    await manager.unload_models()


@pytest.mark.asyncio
async def test_load_fatal_stops_sibling_deduplicates_frontend_fatal_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    fatal_event = ASREvent.asr_result(
        stream_id=0,
        sentence_id=0,
        revision=0,
        status=EventStatus.FATAL_ERROR,
        text="model load failed",
        timestamp_ms=1,
    )
    _ScriptedController.start_scripts[0] = [
        ("asr_event", asr_event_to_payload(fatal_event)),
        (
            "fatal",
            {
                "phase": "load",
                "error_type": "RuntimeError",
                "message": "model load failed",
                "traceback": "trace",
                "exit_code": 1,
            },
        ),
    ]
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    config = AppConfig(lanes=2, output_dir=tmp_path)
    bus = FrontendEventBus()
    manager = ProcessPipelineManager(
        config=config,
        event_bus=bus,
        fake=True,
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.01,
    )

    try:
        with pytest.raises(RuntimeError, match="model load failed"):
            await manager.load_models()

        assert manager.state is PipelineState.FATAL
        sibling = next(item for item in _ScriptedController.instances if item.lane.stream_id == 1)
        assert {"type": "stop", "reason": "sibling_fatal"} in sibling.commands
        fatal_events = []
        while True:
            try:
                event = await asyncio.wait_for(bus.next_event(), timeout=0.02)
            except TimeoutError:
                break
            if event.status is EventStatus.FATAL_ERROR:
                fatal_events.append(event)
        assert len(fatal_events) == 1
        assert all(controller.closed for controller in _ScriptedController.instances)
    finally:
        await manager.unload_models()
    assert manager.state is PipelineState.UNLOADED


@pytest.mark.asyncio
async def test_silent_child_exit_becomes_fatal_with_stream_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    _ScriptedController.silent_exit_codes[0] = 17
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.01,
    )

    try:
        with pytest.raises(RuntimeError, match=r"lane 0.*exit code 17"):
            await manager.load_models()
        assert manager.state is PipelineState.FATAL
        assert manager.lane_metrics[0].final_state is LaneFinalState.FATAL
        assert all(controller.closed for controller in _ScriptedController.instances)
    finally:
        await manager.unload_models()


@pytest.mark.asyncio
async def test_nonfake_silent_child_exit_stops_shared_server_after_all_lanes_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    cleanup_alive_states: list[list[bool]] = []

    class FakeSharedQwenServerManager(_NoopSharedQwenServerManager):
        def stop(self) -> None:
            cleanup_alive_states.append(
                [controller.is_alive() for controller in _ScriptedController.instances]
            )
            super().stop()

    _ScriptedController.silent_exit_codes[0] = -11
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", FakeSharedQwenServerManager
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.01,
    )

    try:
        with pytest.raises(RuntimeError, match=r"lane 0.*exit code -11"):
            await manager.load_models()
    finally:
        await manager.unload_models()

    assert cleanup_alive_states == [[False, False]]


@pytest.mark.asyncio
async def test_metrics_final_wins_and_paths_merge_in_either_order(tmp_path: Path) -> None:
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
    )
    final_lane0 = LaneFinalState.FINISHED
    final_lane1 = LaneFinalState.FATAL

    await manager._handle_message(
        make_message(
            message_type="paths",
            stream_id=1,
            seq=1,
            payload={"model_outputs_path": "model_outputs/lane1.jsonl"},
        )
    )
    await manager._handle_message(
        make_message(
            message_type="metrics_final",
            stream_id=1,
            seq=2,
            payload={
                "lane_metrics": lane_metrics_to_payload(
                    manager.lane_metrics[1].__class__(stream_id=1, processed_frames=9, final_state=final_lane1)
                ),
                "online_frame_latency": dict(_ZERO_LATENCY_FOR_TEST),
                "offline_rtf": dict(_ZERO_RTF_FOR_TEST),
                "event_loop_lag": dict(_ZERO_LATENCY_FOR_TEST),
            },
        )
    )
    await manager._handle_message(
        make_message(
            message_type="metrics_snapshot",
            stream_id=1,
            seq=3,
            payload={
                "lane_metrics": lane_metrics_to_payload(
                    manager.lane_metrics[1].__class__(stream_id=1, processed_frames=999)
                )
            },
        )
    )
    await manager._handle_message(
        make_message(
            message_type="metrics_final",
            stream_id=0,
            seq=1,
            payload={
                "lane_metrics": lane_metrics_to_payload(
                    manager.lane_metrics[0].__class__(stream_id=0, processed_frames=4, final_state=final_lane0)
                ),
                "online_frame_latency": dict(_ZERO_LATENCY_FOR_TEST),
                "offline_rtf": dict(_ZERO_RTF_FOR_TEST),
                "event_loop_lag": dict(_ZERO_LATENCY_FOR_TEST),
            },
        )
    )
    await manager._handle_message(
        make_message(
            message_type="paths",
            stream_id=0,
            seq=2,
            payload={"results_path": "results/lane0.txt"},
        )
    )

    assert [metrics.processed_frames for metrics in manager.lane_metrics] == [4, 9]
    assert [summary["lane_metrics"]["stream_id"] for summary in manager.lane_summaries] == [0, 1]
    assert manager.lane_summaries[0]["results_path"] == "results/lane0.txt"
    assert "model_outputs_path" not in manager.lane_summaries[0]
    assert manager.lane_summaries[1]["model_outputs_path"] == "model_outputs/lane1.jsonl"
    assert "results_path" not in manager.lane_summaries[1]


_ZERO_LATENCY_FOR_TEST = {
    "count": 0,
    "total_ms": 0.0,
    "average_ms": 0.0,
    "max_ms": 0.0,
}
_ZERO_RTF_FOR_TEST = {"count": 0, "total": 0.0, "average": 0.0, "max": 0.0}


@pytest.mark.asyncio
async def test_start_resolves_input_paths_and_loaded_manager_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    relative_target = input_dir / "input.wav"
    _write_valid_wav(relative_target)

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", _NoopSharedQwenServerManager
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, input_dir=input_dir, output_dir=tmp_path / "runs"),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.01,
    )

    await manager.load_models()
    await manager.start_recognition()

    start_commands = [
        next(command for command in controller.commands if command["type"] == "start")
        for controller in _ScriptedController.instances
    ]
    assert [command["input_path"] for command in start_commands] == [
        str(relative_target.resolve()),
        str(relative_target.resolve()),
    ]
    assert all(isinstance(command["run_dir"], str) for command in start_commands)
    assert all(
        set(command) == {"type", "run_dir", "input_kind", "input_path", "loop_audio"}
        for command in start_commands
    )
    assert all(command["input_kind"] == "local_wav" for command in start_commands)
    assert all(command["loop_audio"] is False for command in start_commands)

    await asyncio.wait_for(manager.stop_recognition(), timeout=0.5)
    assert manager.state is PipelineState.LOADED
    await manager.start_recognition()
    await asyncio.wait_for(manager.stop_recognition(), timeout=0.5)
    await manager.unload_models()


@pytest.mark.asyncio
async def test_start_serializes_mixed_local_and_microphone_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    explicit_wav = tmp_path / "explicit.wav"
    _write_valid_wav(explicit_wav)
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", _NoopSharedQwenServerManager
    )
    monkeypatch.setattr(
        process_manager,
        "resolve_microphone_device",
        lambda device: "default-input" if device is None else f"device-{device}",
    )
    manager = ProcessPipelineManager(
        config=AppConfig(
            lanes=2,
            output_dir=tmp_path / "runs",
            lane_inputs=(
                LaneInputConfig(kind="local_wav", local_wav_path=explicit_wav),
                LaneInputConfig(kind="microphone"),
            ),
        ),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.01,
    )

    await manager.load_models()
    await manager.start_recognition()

    commands = [
        next(command for command in controller.commands if command["type"] == "start")
        for controller in _ScriptedController.instances
    ]
    assert commands[0]["input_kind"] == "local_wav"
    assert commands[0]["input_path"] == str(explicit_wav.resolve())
    assert commands[0]["loop_audio"] is False
    assert commands[1] == {
        "type": "start",
        "run_dir": str(manager.run_dir),
        "input_kind": "microphone",
        "microphone_device": None,
        "microphone_device_name": "default-input",
    }

    await manager.stop_recognition()
    await manager.unload_models()


@pytest.mark.asyncio
async def test_start_serializes_only_microphone_input_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    _ScriptedController.start_command_scripts[0] = []
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", _NoopSharedQwenServerManager
    )
    monkeypatch.setattr(
        process_manager,
        "resolve_microphone_device",
        lambda device: "default-input" if device is None else f"device-{device}",
    )
    manager = ProcessPipelineManager(
        config=AppConfig(
            lanes=4,
            output_dir=tmp_path / "runs",
            lane_inputs=(
                LaneInputConfig(kind="microphone", microphone_device=15),
                LaneInputConfig(kind="microphone", microphone_device=16),
                LaneInputConfig(kind="microphone", microphone_device=17),
                LaneInputConfig(kind="microphone", microphone_device=18),
            ),
        ),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.5,
        queue_poll_seconds=0.01,
    )
    await manager.load_models()
    controllers = {
        controller.lane.stream_id: controller
        for controller in _ScriptedController.instances
    }

    start_task = asyncio.create_task(manager.start_recognition())

    def has_start(index: int) -> bool:
        return any(command["type"] == "start" for command in controllers[index].commands)

    async with asyncio.timeout(0.2):
        while not has_start(0):
            await asyncio.sleep(0.001)

    assert has_start(0)
    assert not has_start(1)
    assert not has_start(2)
    assert not has_start(3)

    controllers[0].emit(
        "lifecycle", {"state": "lane_started", "message": ""}
    )
    await asyncio.wait_for(start_task, timeout=0.2)

    assert all(has_start(index) for index in range(4))
    start_commands = {
        controller.lane.stream_id: next(
            command for command in controller.commands if command["type"] == "start"
        )
        for controller in controllers.values()
    }
    for stream_id, device in enumerate((15, 16, 17, 18)):
        assert start_commands[stream_id]["microphone_device"] == device
        assert start_commands[stream_id]["microphone_device_name"] == f"device-{device}"
    await manager.stop_recognition()
    await manager.unload_models()


@pytest.mark.asyncio
async def test_cancelling_start_readiness_wait_cleans_up_partial_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    input_file = tmp_path / "input.wav"
    _write_valid_wav(input_file)
    _ScriptedController.start_command_scripts[0] = []
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", _NoopSharedQwenServerManager
    )
    monkeypatch.setattr(
        process_manager,
        "resolve_microphone_device",
        lambda device: "default-input" if device is None else f"device-{device}",
    )
    manager = ProcessPipelineManager(
        config=AppConfig(
            lanes=2,
            output_dir=tmp_path / "runs",
            lane_inputs=(
                LaneInputConfig(kind="microphone", microphone_device=15),
                LaneInputConfig(kind="local_wav", local_wav_path=input_file),
            ),
        ),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.01,
    )
    await manager.load_models()
    start_task = asyncio.create_task(manager.start_recognition())
    microphone = _ScriptedController.instances[0]
    async with asyncio.timeout(0.2):
        while not any(command["type"] == "start" for command in microphone.commands):
            await asyncio.sleep(0.001)

    start_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await start_task
    assert manager.state is PipelineState.FATAL
    assert all(not controller.is_alive() for controller in _ScriptedController.instances)


@pytest.mark.asyncio
async def test_stop_during_microphone_readiness_returns_to_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    input_file = tmp_path / "input.wav"
    _write_valid_wav(input_file)
    _ScriptedController.start_command_scripts[0] = []
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", _NoopSharedQwenServerManager
    )
    monkeypatch.setattr(
        process_manager,
        "resolve_microphone_device",
        lambda device: "default-input" if device is None else f"device-{device}",
    )
    manager = ProcessPipelineManager(
        config=AppConfig(
            lanes=4,
            output_dir=tmp_path / "runs",
            lane_inputs=(
                LaneInputConfig(kind="microphone", microphone_device=15),
                LaneInputConfig(kind="microphone", microphone_device=16),
                LaneInputConfig(kind="local_wav", local_wav_path=input_file),
                LaneInputConfig(kind="local_wav", local_wav_path=input_file),
            ),
        ),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.5,
        queue_poll_seconds=0.01,
    )
    await manager.load_models()
    controllers = {
        controller.lane.stream_id: controller
        for controller in _ScriptedController.instances
    }
    start_task = asyncio.create_task(manager.start_recognition())
    async with asyncio.timeout(0.2):
        while not any(
            command["type"] == "start" for command in controllers[0].commands
        ):
            await asyncio.sleep(0.001)

    await asyncio.wait_for(manager.stop_recognition(), timeout=0.2)
    await asyncio.wait_for(start_task, timeout=0.2)

    assert manager.state is PipelineState.LOADED
    assert not any(command["type"] == "start" for command in controllers[1].commands)
    assert not any(command["type"] == "stop" for command in controllers[1].commands)
    assert all(controller.is_alive() for controller in controllers.values())
    await manager.unload_models()


@pytest.mark.asyncio
async def test_start_rejects_an_unavailable_microphone_without_leaving_loaded_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    manager = ProcessPipelineManager(
        config=AppConfig(
            lanes=2,
            output_dir=tmp_path / "runs",
            lane_inputs=(
                LaneInputConfig(kind="microphone", microphone_device=20),
                LaneInputConfig(kind="local_wav", local_wav_path=tmp_path / "input.wav"),
            ),
        ),
        event_bus=FrontendEventBus(),
    )
    manager._state = PipelineState.LOADED
    monkeypatch.setattr(
        process_manager,
        "resolve_microphone_device",
        lambda _device: (_ for _ in ()).throw(RuntimeError("Invalid sample rate")),
    )

    with pytest.raises(ValueError, match=r"lane 1 microphone device 20 is unavailable"):
        await manager.start_recognition()

    assert manager.state is PipelineState.LOADED
    assert manager.run_dir is None


@pytest.mark.asyncio
async def test_start_rejects_missing_local_wav_before_entering_starting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    manager = ProcessPipelineManager(
        config=AppConfig(
            lanes=2,
            output_dir=tmp_path / "runs",
            lane_inputs=(
                LaneInputConfig(
                    kind="local_wav", local_wav_path=tmp_path / "missing.wav"
                ),
                LaneInputConfig(
                    kind="local_wav", local_wav_path=tmp_path / "missing-2.wav"
                ),
            ),
        ),
        event_bus=FrontendEventBus(),
    )
    manager._state = PipelineState.LOADED
    manager._controllers = {
        lane.stream_id: _ScriptedController(
            app_config=manager._config,
            lane=lane,
            fake=False,
        )
        for lane in manager._config.lane_configs
    }

    with pytest.raises(FileNotFoundError):
        await manager.start_recognition()

    assert manager.state is PipelineState.LOADED
    assert manager.run_dir is None
    assert not any(
        command["type"] == "start"
        for controller in _ScriptedController.instances
        for command in controller.commands
    )


@pytest.mark.asyncio
async def test_start_rejects_missing_default_inputs_before_entering_starting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager,
        "select_input_files",
        lambda _input_dir, _lanes: (_ for _ in ()).throw(
            FileNotFoundError("default inputs unavailable")
        ),
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path / "runs"),
        event_bus=FrontendEventBus(),
    )
    manager._state = PipelineState.LOADED
    manager._controllers = {
        lane.stream_id: _ScriptedController(
            app_config=manager._config,
            lane=lane,
            fake=False,
        )
        for lane in manager._config.lane_configs
    }

    with pytest.raises(FileNotFoundError, match="default inputs unavailable"):
        await manager.start_recognition()

    assert manager.state is PipelineState.LOADED
    assert manager.run_dir is None
    assert not any(
        command["type"] == "start"
        for controller in _ScriptedController.instances
        for command in controller.commands
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("input_kind", ["local_wav", "default_files"])
async def test_start_rejects_malformed_wav_before_any_run_side_effect(
    input_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    malformed_wav = tmp_path / "malformed.wav"
    malformed_wav.write_bytes(b"not a wav")
    lane_inputs = (
        (
            LaneInputConfig(kind="local_wav", local_wav_path=malformed_wav),
            LaneInputConfig(kind="local_wav", local_wav_path=malformed_wav),
        )
        if input_kind == "local_wav"
        else ()
    )
    if input_kind == "default_files":
        monkeypatch.setattr(
            process_manager,
            "select_input_files",
            lambda _input_dir, lanes: [malformed_wav] * lanes,
        )

    bus = FrontendEventBus()
    bus.publish(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=1,
            status=EventStatus.FINAL,
            text="previous result",
            timestamp_ms=1,
        )
    )
    previous_results = bus.snapshot_results()
    manager = ProcessPipelineManager(
        config=AppConfig(
            lanes=2,
            output_dir=tmp_path / "runs",
            lane_inputs=lane_inputs,
        ),
        event_bus=bus,
    )
    manager._state = PipelineState.LOADED
    manager._controllers = {
        lane.stream_id: _ScriptedController(
            app_config=manager._config,
            lane=lane,
            fake=False,
        )
        for lane in manager._config.lane_configs
    }

    start_task = asyncio.create_task(manager.start_recognition())
    await asyncio.sleep(0)
    if not start_task.done():
        start_task.cancel()
    caught: BaseException | None = None
    try:
        await start_task
    except BaseException as exc:
        caught = exc

    assert isinstance(caught, wave.Error)
    assert manager.state is PipelineState.LOADED
    assert manager.run_dir is None
    assert bus.snapshot_results() == previous_results
    assert not any(
        command["type"] == "start"
        for controller in _ScriptedController.instances
        for command in controller.commands
    )


@pytest.mark.asyncio
async def test_loaded_manager_uses_updated_lane_inputs_on_the_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    local_wav = tmp_path / "next.wav"
    _write_valid_wav(local_wav)
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", _NoopSharedQwenServerManager
    )
    monkeypatch.setattr(
        process_manager,
        "resolve_microphone_device",
        lambda device: "default-input" if device is None else f"device-{device}",
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path / "runs"),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.01,
    )

    await manager.load_models()
    manager.set_lane_inputs(
        (
            LaneInputConfig(kind="local_wav", local_wav_path=local_wav),
            LaneInputConfig(
                kind="microphone",
                microphone_device=3,
                microphone_device_id="stable-device, PulseAudio",
            ),
        )
    )
    await manager.start_recognition()

    commands = [
        next(command for command in controller.commands if command["type"] == "start")
        for controller in _ScriptedController.instances
    ]
    assert commands == [
        {
            "type": "start",
            "run_dir": str(manager.run_dir),
            "input_kind": "local_wav",
            "input_path": str(local_wav.resolve()),
            "loop_audio": False,
        },
        {
            "type": "start",
            "run_dir": str(manager.run_dir),
            "input_kind": "microphone",
            "microphone_device": 3,
            "microphone_device_name": "device-stable-device, PulseAudio",
        },
    ]

    with pytest.raises(ValueError, match="cannot update lane inputs from running"):
        manager.set_lane_inputs(())
    await manager.stop_recognition()
    with pytest.raises(ValueError, match="lane_inputs must match configured lanes"):
        manager.set_lane_inputs((LaneInputConfig(),))
    await manager.unload_models()


@pytest.mark.asyncio
async def test_start_validates_updated_microphone_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path / "runs"),
        event_bus=FrontendEventBus(),
    )
    manager._state = PipelineState.LOADED
    manager.set_lane_inputs(
        (
            LaneInputConfig(
                kind="microphone",
                microphone_device=3,
                microphone_device_id="stable-device, PulseAudio",
            ),
            LaneInputConfig(),
        )
    )
    validated_devices: list[int | str | None] = []

    def reject_microphone(device: int | str | None) -> str:
        validated_devices.append(device)
        raise RuntimeError("device is busy")

    monkeypatch.setattr(process_manager, "resolve_microphone_device", reject_microphone)

    with pytest.raises(
        ValueError,
        match=r"lane 1 microphone device stable-device, PulseAudio is unavailable",
    ):
        await manager.start_recognition()

    assert validated_devices == ["stable-device, PulseAudio"]
    assert manager.state is PipelineState.LOADED


@pytest.mark.asyncio
async def test_run_fatal_stops_sibling_cleans_readers_and_synthesizes_frontend_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    _ScriptedController.start_command_scripts[0] = [
        ("lifecycle", {"state": "lane_started", "message": ""}),
        (
            "fatal",
            {
                "phase": "run",
                "error_type": "RuntimeError",
                "message": "offline request failed",
                "traceback": "trace",
                "exit_code": 1,
            },
        ),
        ("lifecycle", {"state": "lane_stopped", "message": ""}),
    ]
    _ScriptedController.start_command_exit_codes[0] = 1
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    bus = FrontendEventBus()
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=bus,
        fake=True,
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.01,
    )

    await manager.load_models()
    await manager.start_recognition()
    try:
        with pytest.raises(RuntimeError, match="offline request failed"):
            await manager.wait_for_completion()

        assert manager.state is PipelineState.FATAL
        sibling = next(item for item in _ScriptedController.instances if item.lane.stream_id == 1)
        assert {"type": "stop", "reason": "sibling_fatal"} in sibling.commands
        event = await asyncio.wait_for(bus.next_event(), timeout=0.1)
        assert event.status is EventStatus.FATAL_ERROR
        assert event.stream_id == 0
        assert all(controller.closed for controller in _ScriptedController.instances)
    finally:
        await manager.unload_models()


@pytest.mark.asyncio
async def test_start_input_resolution_failure_preserves_loaded_controllers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", _NoopSharedQwenServerManager
    )
    manager = ProcessPipelineManager(
        config=AppConfig(
            lanes=2,
            input_dir=tmp_path / "missing-inputs",
            output_dir=tmp_path / "runs",
        ),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.2,
        queue_poll_seconds=0.01,
    )

    await manager.load_models()
    try:
        with pytest.raises(FileNotFoundError):
            await manager.start_recognition()
        assert manager.state is PipelineState.LOADED
        assert manager.run_dir is None
        assert not any(
            command["type"] == "start"
            for controller in _ScriptedController.instances
            for command in controller.commands
        )
        assert all(not controller.closed for controller in _ScriptedController.instances)
    finally:
        await manager.unload_models()


@pytest.mark.asyncio
async def test_controller_escalates_from_terminate_to_kill() -> None:
    class StubbornProcess:
        def __init__(self) -> None:
            self.alive = True
            self.exitcode: int | None = None
            self.terminate_calls = 0
            self.kill_calls = 0
            self.join_calls: list[float] = []

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

        def join(self, timeout: float) -> None:
            self.join_calls.append(timeout)
            if self.kill_calls:
                self.alive = False
                self.exitcode = -9

    process = StubbornProcess()
    controller = object.__new__(LaneProcessController)
    controller.process = process  # type: ignore[assignment]

    forced = await controller.terminate_then_kill(
        terminate_timeout=0.01,
        kill_timeout=0.02,
    )

    assert forced is True
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_calls == [0.01, 0.02]
    assert process.alive is False


def test_controller_queue_close_never_calls_blocking_join_thread() -> None:
    class RecordingQueue:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def cancel_join_thread(self) -> None:
            self.calls.append("cancel_join_thread")

        def close(self) -> None:
            self.calls.append("close")

        def join_thread(self) -> None:
            raise AssertionError("join_thread may block")

    controller = object.__new__(LaneProcessController)
    controller.command_queue = RecordingQueue()
    controller.event_queue = RecordingQueue()
    controller._queues_closed = False

    controller.close_queues()

    assert controller.command_queue.calls == ["cancel_join_thread", "close"]
    assert controller.event_queue.calls == ["cancel_join_thread", "close"]


def test_controller_closes_dead_process_handle_once() -> None:
    class DeadProcess:
        def __init__(self) -> None:
            self.close_calls = 0

        def is_alive(self) -> bool:
            return False

        def close(self) -> None:
            self.close_calls += 1

    process = DeadProcess()
    controller = object.__new__(LaneProcessController)
    controller.process = process  # type: ignore[assignment]
    controller._process_closed = False

    controller.close_process()
    controller.close_process()

    assert process.close_calls == 1


@pytest.mark.asyncio
async def test_stop_timeout_marks_lanes_fatal_and_cleans_orphans_after_all_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class EscalatingController(_ScriptedController):
        instances: list["EscalatingController"] = []

        def send(self, command: dict[str, Any]) -> None:
            if command["type"] in {"warmup", "start"}:
                super().send(command)
            else:
                self.commands.append(command)

        async def terminate_then_kill(
            self,
            *,
            terminate_timeout: float,
            kill_timeout: float,
        ) -> bool:
            self.commands.append(
                {
                    "type": "escalate",
                    "terminate_timeout": terminate_timeout,
                    "kill_timeout": kill_timeout,
                }
            )
            self._alive = False
            self._exitcode = -9
            return True

    cleanup_observations: list[list[bool]] = []

    class FakeSharedQwenServerManager(_NoopSharedQwenServerManager):
        def stop(self) -> None:
            cleanup_observations.append(
                [controller.is_alive() for controller in EscalatingController.instances]
            )
            super().stop()

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_valid_wav(input_dir / "input.wav")
    monkeypatch.setattr(process_manager, "LaneProcessController", EscalatingController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", FakeSharedQwenServerManager
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, input_dir=input_dir, output_dir=tmp_path / "runs"),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.01,
        terminate_timeout_seconds=0.02,
        kill_timeout_seconds=0.03,
        queue_poll_seconds=0.005,
    )

    await manager.load_models()
    await manager.start_recognition()
    try:
        await manager.stop_recognition()

        assert manager.state is PipelineState.FATAL
        assert [metrics.final_state for metrics in manager.lane_metrics] == [
            LaneFinalState.FATAL,
            LaneFinalState.FATAL,
        ]
        assert cleanup_observations == []
        for controller in EscalatingController.instances:
            assert {
                "type": "escalate",
                "terminate_timeout": 0.02,
                "kill_timeout": 0.03,
            } in controller.commands
    finally:
        await manager.unload_models()
    assert cleanup_observations == [[False, False]]


@pytest.mark.asyncio
async def test_forced_cleanup_and_queue_close_wait_for_last_process_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class DelayedExitController(_ScriptedController):
        instances: list["DelayedExitController"] = []

        def send(self, command: dict[str, Any]) -> None:
            if command["type"] in {"warmup", "start"}:
                super().send(command)
            else:
                self.commands.append(command)

        async def terminate_then_kill(
            self,
            *,
            terminate_timeout: float,
            kill_timeout: float,
        ) -> bool:
            self.commands.append({"type": "escalate"})
            if self.lane.stream_id == 1:
                self._alive = False
                self._exitcode = -9
            return True

    cleanup_observations: list[list[bool]] = []

    class FakeSharedQwenServerManager(_NoopSharedQwenServerManager):
        def stop(self) -> None:
            cleanup_observations.append(
                [controller.is_alive() for controller in DelayedExitController.instances]
            )
            super().stop()

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_valid_wav(input_dir / "input.wav")
    monkeypatch.setattr(process_manager, "LaneProcessController", DelayedExitController)
    monkeypatch.setattr(
        process_manager, "SharedQwenServerManager", FakeSharedQwenServerManager
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, input_dir=input_dir, output_dir=tmp_path / "runs"),
        event_bus=FrontendEventBus(),
        lifecycle_timeout_seconds=0.005,
        terminate_timeout_seconds=0.005,
        kill_timeout_seconds=0.005,
        queue_poll_seconds=0.005,
    )

    await manager.load_models()
    await manager.start_recognition()
    await manager.stop_recognition()

    assert manager.state is PipelineState.FATAL
    assert cleanup_observations == []
    assert not any(controller.closed for controller in DelayedExitController.instances)

    delayed = next(
        controller
        for controller in DelayedExitController.instances
        if controller.lane.stream_id == 0
    )
    delayed._alive = False
    delayed._exitcode = -9
    manager._exited.update({0, 1})
    await manager.unload_models()

    assert cleanup_observations == [[False, False]]
    assert all(controller.closed for controller in DelayedExitController.instances)


@pytest.mark.asyncio
async def test_graceful_stop_does_not_escalate_or_close_loaded_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class GracefulController(_ScriptedController):
        instances: list["GracefulController"] = []

        async def terminate_then_kill(
            self,
            *,
            terminate_timeout: float,
            kill_timeout: float,
        ) -> bool:
            raise AssertionError("graceful lanes must not be terminated")

    monkeypatch.setattr(process_manager, "LaneProcessController", GracefulController)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.1,
        queue_poll_seconds=0.005,
    )

    await manager.load_models()
    await manager.start_recognition()
    await manager.stop_recognition()

    assert manager.state is PipelineState.LOADED
    assert all(
        {"type": "stop", "reason": "user_requested"} in controller.commands
        for controller in GracefulController.instances
    )
    assert not any(controller.closed for controller in GracefulController.instances)
    assert not any(controller.process_closed for controller in GracefulController.instances)
    await manager.unload_models()
    assert all(controller.closed for controller in GracefulController.instances)
    assert all(controller.process_closed for controller in GracefulController.instances)


@pytest.mark.asyncio
async def test_reader_queue_get_is_bounded_and_unload_leaves_no_reader_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class TimeoutRecordingQueue:
        def __init__(self) -> None:
            self.inner: queue.Queue[dict[str, Any]] = queue.Queue()
            self.timeouts: list[float | None] = []
            self.waiting = threading.Event()
            self.get_active = threading.Event()
            self.closed_after_get = False

        def put(self, item: dict[str, Any]) -> None:
            self.inner.put(item)

        def get(self, timeout: float | None = None) -> dict[str, Any]:
            self.timeouts.append(timeout)
            self.waiting.set()
            self.get_active.set()
            try:
                return self.inner.get(timeout=timeout)
            finally:
                self.get_active.clear()

        def close(self) -> None:
            assert not self.get_active.is_set()
            self.closed_after_get = True

    class PollingController(_ScriptedController):
        instances: list["PollingController"] = []

        def __init__(self, *, app_config: AppConfig, lane: Any, fake: bool) -> None:
            super().__init__(app_config=app_config, lane=lane, fake=fake)
            self.event_queue = TimeoutRecordingQueue()  # type: ignore[assignment]

        def close_queues(self) -> None:
            self.event_queue.close()
            super().close_queues()

    monkeypatch.setattr(process_manager, "LaneProcessController", PollingController)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.1,
        queue_poll_seconds=0.01,
    )

    await manager.load_models()
    assert await asyncio.to_thread(
        lambda: all(
            controller.event_queue.waiting.wait(0.2)
            for controller in PollingController.instances
        )
    )
    await manager.unload_models()

    assert all(
        controller.event_queue.timeouts
        and all(timeout == 0.01 for timeout in controller.event_queue.timeouts)
        for controller in PollingController.instances
    )
    assert not [
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("asr-lane-reader-") and not task.done()
    ]
    assert all(
        controller.event_queue.closed_after_get
        for controller in PollingController.instances
    )


@pytest.mark.asyncio
async def test_unload_then_reload_allows_a_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.1,
        queue_poll_seconds=0.005,
    )

    await manager.load_models()
    await manager.start_recognition()
    await manager.stop_recognition()
    first_run = manager.run_dir
    await manager.unload_models()

    await manager.load_models()
    await manager.start_recognition()
    await manager.stop_recognition()

    assert manager.state is PipelineState.LOADED
    assert manager.run_dir is not None
    assert manager.run_dir != first_run
    await manager.unload_models()


@pytest.mark.asyncio
async def test_fake_run_returns_to_loaded_and_keeps_lane_children(tmp_path: Path) -> None:
    config = AppConfig(lanes=2, output_dir=tmp_path)
    bus = FrontendEventBus()
    manager = ProcessPipelineManager(
        config=config,
        event_bus=bus,
        fake=True,
        lifecycle_timeout_seconds=2.0,
    )

    await manager.load_models()
    assert manager.state is PipelineState.LOADED

    await manager.start_recognition()
    await asyncio.wait_for(manager.wait_for_completion(), timeout=1.0)

    assert manager.state is PipelineState.LOADED
    assert manager.run_dir is not None
    assert manager.run_dir.name.startswith("process-")
    assert [metrics.final_state for metrics in manager.lane_metrics] == [
        LaneFinalState.FINISHED,
        LaneFinalState.FINISHED,
    ]
    assert [summary["lane_metrics"]["stream_id"] for summary in manager.lane_summaries] == [0, 1]
    assert (manager.run_dir / "results" / "lane0.txt").exists()
    assert (manager.run_dir / "model_outputs" / "lane1.jsonl").exists()

    statuses: list[EventStatus] = []
    while True:
        try:
            event = await asyncio.wait_for(bus.next_event(), timeout=0.05)
        except TimeoutError:
            break
        statuses.append(event.status)
    assert EventStatus.PARTIAL in statuses
    assert EventStatus.PENDING in statuses
    assert EventStatus.FINAL in statuses
    assert statuses.count(EventStatus.LANE_FINISHED) == 2

    await manager.start_recognition()
    await asyncio.wait_for(manager.wait_for_completion(), timeout=1.0)
    assert manager.state is PipelineState.LOADED

    await manager.unload_models()

    assert manager.state is PipelineState.UNLOADED
    assert not [child for child in mp.active_children() if child.name.startswith("asr-lane-")]


@pytest.mark.asyncio
async def test_unload_retains_ownership_when_process_survives_kill_then_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class NeverDiesController(_ScriptedController):
        instances: list["NeverDiesController"] = []

        def send(self, command: dict[str, Any]) -> None:
            if command["type"] == "warmup":
                super().send(command)
            else:
                self.commands.append(command)

        async def terminate_then_kill(
            self,
            *,
            terminate_timeout: float,
            kill_timeout: float,
        ) -> bool:
            self.commands.append({"type": "terminate_then_kill"})
            return True

    monkeypatch.setattr(process_manager, "LaneProcessController", NeverDiesController)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.005,
        terminate_timeout_seconds=0.005,
        kill_timeout_seconds=0.005,
        queue_poll_seconds=0.005,
    )

    await manager.load_models()
    controllers = dict(manager._controllers)
    readers = dict(manager._reader_tasks)
    try:
        with pytest.raises(RuntimeError, match=r"teardown incomplete.*0.*1"):
            await manager.unload_models()

        assert manager.state is PipelineState.FATAL
        assert manager._controllers == controllers
        assert manager._reader_tasks == readers
        assert not any(controller.closed for controller in controllers.values())

        for controller in controllers.values():
            controller._alive = False
            controller._exitcode = -9
        await manager.unload_models()

        assert manager.state is PipelineState.UNLOADED
        assert manager._controllers == {}
        assert manager._reader_tasks == {}
        assert all(controller.closed for controller in controllers.values())
    finally:
        for controller in controllers.values():
            controller._alive = False
        for task in readers.values():
            task.cancel()
        await asyncio.gather(*readers.values(), return_exceptions=True)


@pytest.mark.asyncio
async def test_dead_child_gets_final_bounded_drain_before_silent_exit_fatal(
    tmp_path: Path,
) -> None:
    class EmptyThenMessagesQueue:
        def __init__(self, messages: list[dict[str, Any]]) -> None:
            self.messages = list(messages)
            self.calls = 0
            self.timeouts: list[float | None] = []

        def get(self, timeout: float | None = None) -> dict[str, Any]:
            self.calls += 1
            self.timeouts.append(timeout)
            if self.calls == 1 or not self.messages:
                raise queue.Empty
            return self.messages.pop(0)

    config = AppConfig(lanes=2, output_dir=tmp_path)
    lane = config.lane_configs[0]
    controller = _ScriptedController(app_config=config, lane=lane, fake=True)
    controller._alive = False
    controller._exitcode = 0
    final_payload = {
        "lane_metrics": {
            "stream_id": 0,
            "processed_frames": 7,
            "dropped_frames": 1,
            "offline_requests": 2,
            "final_state": "finished",
        },
        "online_frame_latency": dict(_ZERO_LATENCY_FOR_TEST),
        "offline_rtf": dict(_ZERO_RTF_FOR_TEST),
        "event_loop_lag": dict(_ZERO_LATENCY_FOR_TEST),
    }
    controller.event_queue = EmptyThenMessagesQueue(  # type: ignore[assignment]
        [
            make_message(
                message_type="metrics_final",
                stream_id=0,
                seq=1,
                payload=final_payload,
            ).to_dict(),
            make_message(
                message_type="paths",
                stream_id=0,
                seq=2,
                payload={
                    "results_path": "results/lane0.txt",
                    "model_outputs_path": "model_outputs/lane0.jsonl",
                },
            ).to_dict(),
            make_message(
                message_type="lifecycle",
                stream_id=0,
                seq=3,
                payload={"state": "lane_stopped", "message": ""},
            ).to_dict(),
        ]
    )
    manager = ProcessPipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        fake=True,
        queue_poll_seconds=0.005,
    )
    manager._state = PipelineState.LOADING
    manager._controllers = {0: controller}

    await manager._read_lane_events(controller)

    assert manager._failure is None
    assert manager._terminal == {0}
    assert manager._exited == {0}
    assert manager.lane_metrics[0].processed_frames == 7
    assert manager.lane_summaries[0]["results_path"] == "results/lane0.txt"
    assert controller.event_queue.calls == 5
    assert all(timeout == 0.005 for timeout in controller.event_queue.timeouts)


@pytest.mark.asyncio
async def test_main_process_fatal_normalizes_existing_and_late_final_summary(
    tmp_path: Path,
) -> None:
    config = AppConfig(lanes=2, output_dir=tmp_path)
    manager = ProcessPipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        fake=True,
    )
    manager._state = PipelineState.RUNNING
    initial_final = {
        "lane_metrics": {
            "stream_id": 0,
            "processed_frames": 7,
            "dropped_frames": 1,
            "offline_requests": 2,
            "final_state": "finished",
        },
        "online_frame_latency": {
            "count": 7,
            "total_ms": 70.0,
            "average_ms": 10.0,
            "max_ms": 15.0,
        },
        "offline_rtf": dict(_ZERO_RTF_FOR_TEST),
        "event_loop_lag": dict(_ZERO_LATENCY_FOR_TEST),
    }
    await manager._handle_message(
        make_message(
            message_type="metrics_final",
            stream_id=0,
            seq=1,
            payload=initial_final,
        )
    )
    await manager._handle_message(
        make_message(
            message_type="paths",
            stream_id=0,
            seq=2,
            payload={"results_path": "results/lane0.txt"},
        )
    )

    await manager._record_fatal(0, "forced timeout")

    late_final = {
        **initial_final,
        "lane_metrics": {
            **initial_final["lane_metrics"],
            "processed_frames": 8,
            "final_state": "finished",
        },
    }
    await manager._handle_message(
        make_message(
            message_type="metrics_final",
            stream_id=0,
            seq=3,
            payload=late_final,
        )
    )

    summary = manager.lane_summaries[0]
    assert manager.lane_metrics[0].final_state is LaneFinalState.FATAL
    assert summary["lane_metrics"]["processed_frames"] == 8
    assert summary["lane_metrics"]["final_state"] == "fatal"
    assert summary["online_frame_latency"] == initial_final["online_frame_latency"]
    assert summary["results_path"] == "results/lane0.txt"

    run_dir = RunRecorder().save_process_run(
        tmp_path / "recorded",
        config=config,
        lane_summaries=manager.lane_summaries,
    )
    recorded = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert recorded["lanes"][0]["final_state"] == "fatal"
    assert recorded["lanes"][0]["processed_frames"] == 8
    assert recorded["lanes"][0]["results_path"] == "results/lane0.txt"


@pytest.mark.asyncio
async def test_malformed_ipc_is_fatal_and_broken_origin_send_does_not_strand_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class MalformedController(_ScriptedController):
        instances: list["MalformedController"] = []

        def start(self) -> None:
            self._alive = True
            if self.lane.stream_id == 0:
                self.event_queue.put({"type": "lifecycle", "payload": {}})
            else:
                self.emit("lifecycle", {"state": "lane_loaded", "message": ""})

        def send(self, command: dict[str, Any]) -> None:
            self.commands.append(command)
            if self.lane.stream_id == 0:
                raise BrokenPipeError("origin command queue is broken")
            if command["type"] in {"stop", "shutdown"}:
                self.emit("lifecycle", {"state": "lane_stopped", "message": ""})
                self._alive = False
                self._exitcode = 0

        async def terminate_then_kill(
            self,
            *,
            terminate_timeout: float,
            kill_timeout: float,
        ) -> bool:
            self._alive = False
            self._exitcode = -9
            return True

    monkeypatch.setattr(process_manager, "LaneProcessController", MalformedController)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.02,
        queue_poll_seconds=0.005,
    )

    with pytest.raises(RuntimeError, match=r"IPC reader failed.*KeyError"):
        await manager.load_models()

    origin, sibling = MalformedController.instances
    assert any(command["type"] == "stop" for command in origin.commands)
    assert any(command["type"] == "stop" for command in sibling.commands)
    assert manager.state is PipelineState.FATAL
    assert all(controller.closed for controller in MalformedController.instances)
    assert all(controller.process_closed for controller in MalformedController.instances)
    await manager.unload_models()


@pytest.mark.asyncio
async def test_wrong_stream_ipc_is_fatal_for_controller_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class WrongStreamController(_ScriptedController):
        instances: list["WrongStreamController"] = []

        def start(self) -> None:
            self._alive = True
            stream_id = 1 if self.lane.stream_id == 0 else self.lane.stream_id
            self.event_queue.put(
                make_message(
                    message_type="lifecycle",
                    stream_id=stream_id,
                    seq=1,
                    payload={"state": "lane_loaded", "message": ""},
                ).to_dict()
            )

    monkeypatch.setattr(process_manager, "LaneProcessController", WrongStreamController)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.02,
        queue_poll_seconds=0.005,
    )

    with pytest.raises(RuntimeError, match=r"lane 0.*stream 1"):
        await manager.load_models()

    assert manager.state is PipelineState.FATAL
    assert manager.lane_metrics[0].final_state is LaneFinalState.FATAL
    assert all(controller.closed for controller in WrongStreamController.instances)
    await manager.unload_models()


@pytest.mark.asyncio
async def test_halfway_controller_start_failure_closes_started_and_unstarted_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class HalfStartController(_ScriptedController):
        instances: list["HalfStartController"] = []

        def start(self) -> None:
            if self.lane.stream_id == 1:
                raise RuntimeError("spawn failed halfway")
            super().start()

    monkeypatch.setattr(process_manager, "LaneProcessController", HalfStartController)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.02,
        queue_poll_seconds=0.005,
    )

    with pytest.raises(RuntimeError, match="spawn failed halfway"):
        await manager.load_models()

    started, never_started = HalfStartController.instances
    assert started.is_alive() is False
    assert never_started.is_alive() is False
    assert all(controller.closed for controller in HalfStartController.instances)
    assert all(controller.process_closed for controller in HalfStartController.instances)
    assert not [task for task in manager._reader_tasks.values() if not task.done()]
    await manager.unload_models()


@pytest.mark.parametrize(
    ("case_id", "message_type", "payload", "error_match"),
    [
        (
            "asr_payload_stream",
            "asr_event",
            asr_event_to_payload(
                ASREvent.asr_result(
                    stream_id=1,
                    sentence_id=0,
                    revision=0,
                    status=EventStatus.PARTIAL,
                    text="wrong lane",
                    timestamp_ms=1,
                )
            ),
            r"asr_event payload stream 1.*lane 0",
        ),
        (
            "metrics_final_stream",
            "metrics_final",
            {"lane_metrics": {"stream_id": 1, "final_state": "finished"}},
            r"metrics_final lane_metrics stream 1.*lane 0",
        ),
        (
            "metrics_snapshot_stream",
            "metrics_snapshot",
            {"lane_metrics": {"stream_id": 1, "final_state": "stopped"}},
            r"metrics_snapshot lane_metrics stream 1.*lane 0",
        ),
        (
            "unknown_type",
            "unknown",
            {},
            r"unsupported IPC message type.*unknown",
        ),
        (
            "metrics_final_missing",
            "metrics_final",
            {},
            r"metrics_final requires nonempty lane_metrics",
        ),
        (
            "metrics_final_empty",
            "metrics_final",
            {"lane_metrics": {}},
            r"metrics_final requires nonempty lane_metrics",
        ),
        (
            "metrics_snapshot_missing",
            "metrics_snapshot",
            {},
            r"metrics_snapshot requires nonempty lane_metrics",
        ),
        (
            "metrics_snapshot_empty",
            "metrics_snapshot",
            {"lane_metrics": {}},
            r"metrics_snapshot requires nonempty lane_metrics",
        ),
        (
            "invalid_lifecycle",
            "lifecycle",
            {"state": "lane_unknown", "message": ""},
            r"invalid lifecycle state.*lane_unknown",
        ),
        (
            "empty_paths",
            "paths",
            {"results_path": ""},
            r"paths requires exact lane 0 artifact paths",
        ),
        (
            "blank_fatal",
            "fatal",
            {"message": "   "},
            r"fatal requires a nonempty message",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.asyncio
async def test_semantically_invalid_ipc_is_fatal_before_dispatch(
    case_id: str,
    message_type: str,
    payload: dict[str, Any],
    error_match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class SemanticInvalidController(_ScriptedController):
        instances: list["SemanticInvalidController"] = []

        def start(self) -> None:
            self._alive = True
            if self.lane.stream_id == 0:
                self.event_queue.put(
                    make_message(
                        message_type=message_type,
                        stream_id=0,
                        seq=1,
                        payload=payload,
                    ).to_dict()
                )
            else:
                self.emit("lifecycle", {"state": "lane_loaded", "message": ""})

    monkeypatch.setattr(
        process_manager,
        "LaneProcessController",
        SemanticInvalidController,
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path / case_id),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.02,
        queue_poll_seconds=0.005,
    )

    with pytest.raises(RuntimeError, match=error_match):
        await manager.load_models()

    origin, sibling = SemanticInvalidController.instances
    assert any(command["type"] == "stop" for command in origin.commands)
    assert any(command["type"] == "stop" for command in sibling.commands)
    assert manager.state is PipelineState.FATAL
    assert manager.lane_metrics[0].final_state is LaneFinalState.FATAL
    assert all(controller.closed for controller in SemanticInvalidController.instances)
    await manager.unload_models()


_LANE0_METRICS = {
    "lane_metrics": {
        "stream_id": 0,
        "processed_frames": 1,
        "dropped_frames": 0,
        "offline_requests": 0,
        "final_state": "finished",
    }
}


@pytest.mark.parametrize(
    ("case_id", "message_type", "payload", "error_match"),
    [
        (
            "standalone_mixed_type",
            "paths",
            {
                "results_path": "results/lane0.txt",
                "model_outputs_path": 123,
            },
            r"paths requires exact lane 0 artifact paths",
        ),
        (
            "metrics_final_mixed_type",
            "metrics_final",
            {
                **_LANE0_METRICS,
                "results_path": "results/lane0.txt",
                "model_outputs_path": 123,
            },
            r"metrics_final requires exact lane 0 artifact paths",
        ),
        (
            "metrics_final_absolute",
            "metrics_final",
            {
                **_LANE0_METRICS,
                "results_path": "/tmp/lane0.txt",
                "model_outputs_path": "/tmp/lane0.jsonl",
            },
            r"metrics_final requires exact lane 0 artifact paths",
        ),
        (
            "metrics_final_traversal",
            "metrics_final",
            {
                **_LANE0_METRICS,
                "results_path": "../results/lane0.txt",
                "model_outputs_path": "model_outputs/../lane0.jsonl",
            },
            r"metrics_final requires exact lane 0 artifact paths",
        ),
        (
            "metrics_final_wrong_lane",
            "metrics_final",
            {
                **_LANE0_METRICS,
                "results_path": "results/lane1.txt",
                "model_outputs_path": "model_outputs/lane1.jsonl",
            },
            r"metrics_final requires exact lane 0 artifact paths",
        ),
        (
            "metrics_final_incomplete",
            "metrics_final",
            {
                **_LANE0_METRICS,
                "results_path": "results/lane0.txt",
            },
            r"metrics_final requires exact lane 0 artifact paths",
        ),
        (
            "standalone_wrong_lane",
            "paths",
            {
                "results_path": "results/lane1.txt",
                "model_outputs_path": "model_outputs/lane1.jsonl",
            },
            r"paths requires exact lane 0 artifact paths",
        ),
        (
            "standalone_traversal",
            "paths",
            {
                "results_path": "results/../lane0.txt",
                "model_outputs_path": "../model_outputs/lane0.jsonl",
            },
            r"paths requires exact lane 0 artifact paths",
        ),
        (
            "snapshot_artifacts",
            "metrics_snapshot",
            {
                **_LANE0_METRICS,
                "results_path": "results/lane0.txt",
                "model_outputs_path": "model_outputs/lane0.jsonl",
            },
            r"metrics_snapshot must not carry artifact paths",
        ),
    ],
    ids=[
        "standalone-mixed-type",
        "final-mixed-type",
        "final-absolute",
        "final-traversal",
        "final-wrong-lane",
        "final-incomplete",
        "standalone-wrong-lane",
        "standalone-traversal",
        "snapshot-artifacts",
    ],
)
@pytest.mark.asyncio
async def test_invalid_artifact_paths_are_fatal_before_dispatch(
    case_id: str,
    message_type: str,
    payload: dict[str, Any],
    error_match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class InvalidPathController(_ScriptedController):
        instances: list["InvalidPathController"] = []

        def start(self) -> None:
            self._alive = True
            if self.lane.stream_id == 0:
                self.event_queue.put(
                    make_message(
                        message_type=message_type,
                        stream_id=0,
                        seq=1,
                        payload=payload,
                    ).to_dict()
                )
            else:
                self.emit("lifecycle", {"state": "lane_loaded", "message": ""})

    monkeypatch.setattr(process_manager, "LaneProcessController", InvalidPathController)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path / case_id),
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=0.02,
        queue_poll_seconds=0.005,
    )

    with pytest.raises(RuntimeError, match=error_match):
        await manager.load_models()

    origin, sibling = InvalidPathController.instances
    assert any(command["type"] == "stop" for command in origin.commands)
    assert any(command["type"] == "stop" for command in sibling.commands)
    assert manager.state is PipelineState.FATAL
    assert manager.lane_metrics[0].final_state is LaneFinalState.FATAL
    assert all(controller.closed for controller in InvalidPathController.instances)
    await manager.unload_models()


def _qwen_error_payload(
    run_id: str,
    *,
    sentence_id: int = 4,
    pending_text: str = "zipformer pending",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sentence_id": sentence_id,
        "revision": 2,
        "pending_text": pending_text,
        "timestamp_ms": 1234,
        "request": {
            "category": "http_status",
            "error_type": "HTTPStatusError",
            "message": "HTTP 500",
            "method": "POST",
            "url": "http://qwen/v1/chat/completions",
            "timeout_ms": 120000,
            "audio_seconds": 1.0,
            "sample_count": 16000,
            "status_code": 500,
            "response_body": "decoder failed",
            "response_body_truncated": False,
        },
    }


def test_qwen_ipc_validation_accepts_exact_success_and_error_payloads(
    tmp_path: Path,
) -> None:
    config = AppConfig(lanes=2, output_dir=tmp_path)
    manager = ProcessPipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        fake=True,
    )
    manager._run_dir = tmp_path / "process-123"
    manager._qwen_run_active = True
    controller = _ScriptedController(
        app_config=config,
        lane=config.lane_configs[0],
        fake=True,
    )

    manager._validate_controller_message(
        controller,
        make_message(
            message_type="qwen_success",
            stream_id=0,
            seq=1,
            payload={"run_id": "process-123", "sentence_id": 4},
        ),
    )
    manager._validate_controller_message(
        controller,
        make_message(
            message_type="qwen_error",
            stream_id=0,
            seq=2,
            payload=_qwen_error_payload("process-123", pending_text=""),
        ),
    )


@pytest.mark.parametrize(
    ("message_type", "mutate", "error_match"),
    [
        (
            "qwen_success",
            lambda payload: payload.update(run_id="process-other"),
            r"qwen_success run_id.*active run",
        ),
        (
            "qwen_success",
            lambda payload: payload.update(sentence_id=True),
            r"qwen_success sentence_id must be an integer",
        ),
        (
            "qwen_error",
            lambda payload: payload.update(revision="2"),
            r"qwen_error revision must be an integer",
        ),
        (
            "qwen_error",
            lambda payload: payload.update(timestamp_ms=False),
            r"qwen_error timestamp_ms must be an integer",
        ),
        (
            "qwen_error",
            lambda payload: payload.update(pending_text=None),
            r"qwen_error pending_text must be a string",
        ),
        (
            "qwen_error",
            lambda payload: payload.update(request=[]),
            r"qwen_error request must be a mapping",
        ),
        (
            "qwen_error",
            lambda payload: payload["request"].pop("error_type"),
            r"qwen_error request missing required fields.*error_type",
        ),
        (
            "qwen_error",
            lambda payload: payload["request"].update(secret="unexpected"),
            r"qwen_error request has unsupported fields.*secret",
        ),
    ],
    ids=[
        "wrong-run",
        "boolean-sentence",
        "string-revision",
        "boolean-timestamp",
        "non-string-pending",
        "non-mapping-request",
        "missing-request-field",
        "extra-request-field",
    ],
)
def test_qwen_ipc_validation_rejects_malformed_payloads(
    message_type: str,
    mutate: Any,
    error_match: str,
    tmp_path: Path,
) -> None:
    config = AppConfig(lanes=2, output_dir=tmp_path)
    manager = ProcessPipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        fake=True,
    )
    manager._run_dir = tmp_path / "process-123"
    manager._qwen_run_active = True
    controller = _ScriptedController(
        app_config=config,
        lane=config.lane_configs[0],
        fake=True,
    )
    payload = (
        {"run_id": "process-123", "sentence_id": 4}
        if message_type == "qwen_success"
        else _qwen_error_payload("process-123")
    )
    mutate(payload)

    with pytest.raises(ValueError, match=error_match):
        manager._validate_controller_message(
            controller,
            make_message(
                message_type=message_type,
                stream_id=0,
                seq=1,
                payload=payload,
            ),
        )


@pytest.mark.parametrize("message_type", ["qwen_success", "qwen_error"])
def test_qwen_ipc_validation_rejects_matching_run_after_finish(
    message_type: str,
    tmp_path: Path,
) -> None:
    config = AppConfig(lanes=2, output_dir=tmp_path)
    manager = ProcessPipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        fake=True,
    )
    manager._run_dir = tmp_path / "process-123"
    controller = _ScriptedController(
        app_config=config,
        lane=config.lane_configs[0],
        fake=True,
    )
    payload = (
        {"run_id": "process-123", "sentence_id": 4}
        if message_type == "qwen_success"
        else _qwen_error_payload("process-123")
    )

    with pytest.raises(ValueError, match=rf"{message_type} requires an active run"):
        manager._validate_controller_message(
            controller,
            make_message(
                message_type=message_type,
                stream_id=0,
                seq=1,
                payload=payload,
            ),
        )


@pytest.mark.asyncio
async def test_fake_controller_qwen_emissions_cross_manager_whitelist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class QwenOutcomeController(_ScriptedController):
        instances: list["QwenOutcomeController"] = []

        def send(self, command: dict[str, Any]) -> None:
            super().send(command)
            if command["type"] != "start":
                return
            run_id = Path(command["run_dir"]).name
            self.emit(
                "qwen_success",
                {"run_id": run_id, "sentence_id": 1},
            )
            self.emit(
                "qwen_error",
                _qwen_error_payload(run_id, sentence_id=2),
            )

    monkeypatch.setattr(process_manager, "LaneProcessController", QwenOutcomeController)
    coordinator = _RecordingQwenErrors(
        [
            QwenFailureDecision(1, 1, False),
            QwenFailureDecision(2, 2, False),
        ]
    )
    bus = FrontendEventBus()
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=bus,
        fake=True,
        qwen_error_coordinator=coordinator,
        queue_poll_seconds=0.005,
    )

    await manager.load_models()
    await manager.start_recognition()
    await manager._wait_for(lambda: len(coordinator.failures) == 2, timeout=0.2)

    assert coordinator.successes == 2
    assert manager._failure is None
    assert {
        result["status"] for result in bus.snapshot_results().values()
    } == {EventStatus.QWEN_ERROR}

    await manager.stop_recognition()
    await manager.unload_models()


@pytest.mark.asyncio
async def test_qwen_failure_threshold_publishes_before_exactly_one_global_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = [
        QwenFailureDecision(1, 1, False),
        QwenFailureDecision(2, 2, False),
        QwenFailureDecision(3, 3, True),
    ]
    coordinator = _RecordingQwenErrors(decisions)
    bus = FrontendEventBus()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    manager = ProcessPipelineManager(
        config=config,
        event_bus=bus,
        fake=True,
        qwen_error_coordinator=coordinator,
    )
    manager._state = PipelineState.RUNNING
    manager._run_dir = tmp_path / "process-123"
    controllers = {
        lane.stream_id: _ScriptedController(
            app_config=config,
            lane=lane,
            fake=True,
        )
        for lane in config.lane_configs
    }
    for controller in controllers.values():
        controller._alive = True
    manager._controllers = controllers
    fatal_calls: list[tuple[int, str]] = []
    original_record_fatal = manager._record_fatal

    async def record_fatal(stream_id: int, message: str) -> None:
        fatal_calls.append((stream_id, message))
        await original_record_fatal(stream_id, message)

    monkeypatch.setattr(manager, "_record_fatal", record_fatal)

    for sequence in range(1, 4):
        await manager._handle_message(
            make_message(
                message_type="qwen_error",
                stream_id=0,
                seq=sequence,
                time_ms=9000 + sequence,
                payload=_qwen_error_payload(
                    "process-123",
                    sentence_id=sequence,
                ),
            )
        )
        qwen_event = await bus.next_event()
        assert qwen_event.status is EventStatus.QWEN_ERROR
        assert qwen_event.is_final is False
        if sequence < 3:
            assert manager._failure is None
            assert not any(controller.commands for controller in controllers.values())

    fatal_event = await bus.next_event()
    assert fatal_event.status is EventStatus.FATAL_ERROR
    assert fatal_calls == [(0, "Qwen request failure threshold reached (3/3)")]
    assert manager.state is PipelineState.FATAL
    assert coordinator.failures[0]["time_ms"] == 9001
    assert all(
        controller.commands == [{"type": "stop", "reason": "sibling_fatal"}]
        for controller in controllers.values()
    )


@pytest.mark.asyncio
async def test_qwen_error_artifact_write_failure_is_immediate_fatal(
    tmp_path: Path,
) -> None:
    class FailingQwenErrors(_RecordingQwenErrors):
        def record_failure(self, **kwargs: Any) -> QwenFailureDecision:
            raise OSError("disk full")

    coordinator = FailingQwenErrors([])
    bus = FrontendEventBus()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    manager = ProcessPipelineManager(
        config=config,
        event_bus=bus,
        fake=True,
        qwen_error_coordinator=coordinator,
    )
    manager._state = PipelineState.RUNNING
    manager._run_dir = tmp_path / "process-123"
    controllers = {
        lane.stream_id: _ScriptedController(
            app_config=config,
            lane=lane,
            fake=True,
        )
        for lane in config.lane_configs
    }
    for controller in controllers.values():
        controller._alive = True
    manager._controllers = controllers

    await manager._handle_message(
        make_message(
            message_type="qwen_error",
            stream_id=0,
            seq=1,
            payload=_qwen_error_payload("process-123"),
        )
    )

    event = await bus.next_event()
    assert event.status is EventStatus.FATAL_ERROR
    assert "failed to persist Qwen request error: OSError: disk full" in event.text
    assert manager.state is PipelineState.FATAL
    assert all(
        controller.commands == [{"type": "stop", "reason": "sibling_fatal"}]
        for controller in controllers.values()
    )


@pytest.mark.asyncio
async def test_qwen_failure_threshold_is_cumulative_across_lanes_and_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    coordinator = _RecordingQwenErrors(
        [
            QwenFailureDecision(1, 1, False),
            QwenFailureDecision(1, 2, False),
            QwenFailureDecision(2, 3, True),
        ]
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        qwen_error_coordinator=coordinator,
        queue_poll_seconds=0.005,
    )

    await manager.load_models()
    await manager.start_recognition()
    first_run_id = manager.run_dir.name
    _ScriptedController.instances[0].emit(
        "qwen_error",
        _qwen_error_payload(first_run_id, sentence_id=1),
    )
    await manager._wait_for(lambda: len(coordinator.failures) == 1, timeout=0.2)
    await manager.stop_recognition()

    await manager.start_recognition()
    second_run_id = manager.run_dir.name
    _ScriptedController.instances[1].emit(
        "qwen_error",
        _qwen_error_payload(second_run_id, sentence_id=2),
    )
    await manager._wait_for(lambda: len(coordinator.failures) == 2, timeout=0.2)
    assert manager._failure is None
    _ScriptedController.instances[0].emit(
        "qwen_error",
        _qwen_error_payload(second_run_id, sentence_id=3),
    )

    with pytest.raises(
        RuntimeError,
        match=r"Qwen request failure threshold reached \(3/3\)",
    ):
        await manager.wait_for_completion()

    assert [failure["stream_id"] for failure in coordinator.failures] == [0, 1, 0]
    assert [
        failure["payload"]["run_id"] for failure in coordinator.failures
    ] == [first_run_id, second_run_id, second_run_id]
    assert coordinator.finished == 2
    assert coordinator.reset_unload == 0
    assert manager.state is PipelineState.FATAL

    await manager.unload_models()
    assert coordinator.reset_unload == 1


@pytest.mark.asyncio
async def test_qwen_coordinator_count_lifecycle_spans_runs_and_resets_on_unload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    coordinator = _RecordingQwenErrors([])
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        fake=True,
        qwen_error_coordinator=coordinator,
        queue_poll_seconds=0.005,
    )

    await manager.load_models()
    for _ in range(2):
        await manager.start_recognition()
        await manager.stop_recognition()

    assert coordinator.reset_load == 1
    assert len(coordinator.started) == 2
    assert coordinator.started[0] != coordinator.started[1]
    assert coordinator.finished == 2
    assert coordinator.reset_unload == 0

    await manager.unload_models()

    assert coordinator.reset_load == 1
    assert coordinator.finished == 2
    assert coordinator.reset_unload == 1


@pytest.mark.asyncio
async def test_concurrent_fatal_unload_latches_qwen_finish_failure(
    tmp_path: Path,
) -> None:
    class FailingFinishQwenErrors(_RecordingQwenErrors):
        def finish_run(self) -> None:
            self.finished += 1
            raise OSError("finish failed")

    join_entered = asyncio.Event()
    release_join = asyncio.Event()

    class GatedJoinController(_ScriptedController):
        def send(self, command: dict[str, Any]) -> None:
            self.commands.append(command)

        async def join(self, timeout: float) -> int | None:
            join_entered.set()
            await release_join.wait()
            self._alive = False
            self._exitcode = 0
            return self._exitcode

    coordinator = FailingFinishQwenErrors([])
    config = AppConfig(lanes=2, output_dir=tmp_path)
    manager = ProcessPipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        fake=True,
        qwen_error_coordinator=coordinator,
    )
    manager._state = PipelineState.FATAL
    manager._failure = RuntimeError("lane failed")
    manager._run_dir = tmp_path / "process-123"
    manager._qwen_run_active = True
    manager._controllers = {
        lane.stream_id: GatedJoinController(
            app_config=config,
            lane=lane,
            fake=True,
        )
        for lane in config.lane_configs
    }
    for controller in manager._controllers.values():
        controller._alive = True

    first_unload = asyncio.create_task(manager.unload_models())
    await join_entered.wait()
    second_unload = asyncio.create_task(manager.unload_models())
    await asyncio.sleep(0)
    release_join.set()

    results = await asyncio.gather(
        first_unload,
        second_unload,
        return_exceptions=True,
    )

    assert results == [manager._failure, manager._failure]
    assert coordinator.finished == 1
    assert coordinator.reset_unload == 0
    assert manager.state is PipelineState.FATAL

    late_result = (
        await asyncio.gather(manager.unload_models(), return_exceptions=True)
    )[0]
    assert late_result is manager._failure
    assert coordinator.finished == 1
    assert coordinator.reset_unload == 0
    assert manager.state is PipelineState.FATAL
