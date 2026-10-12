from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest
import numpy as np

from multi_stream_asr.config import AppConfig, LaneConfig
from multi_stream_asr.models import ASREvent, AudioFrame, EventStatus, LaneFinalState
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.web.event_bus import FrontendEventBus

from conftest import (
    FakeAudioSource,
    FakeOfflineRecognizer,
    FakeOnlineRecognizer,
    make_frame,
    make_online,
)


class RecordingLlamaServerManager:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    def start_all(self):
        self.start_calls += 1
        return []

    def stop_all(self) -> None:
        self.stop_calls += 1


class SlowRecordingLlamaServerManager(RecordingLlamaServerManager):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = threading.Event()

    def start_all(self):
        self.start_calls += 1
        self.started.set()
        self.release.wait()
        return []


class BlockingAudioSource:
    def __init__(self, *, stream_id: int) -> None:
        self.stream_id = stream_id
        self.started = threading.Event()
        self.stop_requested = threading.Event()
        self.stop_calls = 0

    async def prepare(self) -> None:
        return None

    async def frames(self):
        yield make_frame(
            stream_id=self.stream_id,
            frame_index=0,
            samples=[0.1, 0.2],
            timestamp_ms=100,
        )
        self.started.set()
        await asyncio.to_thread(self.stop_requested.wait)

    async def stop(self) -> None:
        self.stop_calls += 1
        self.stop_requested.set()


class DrainingAudioSource:
    def __init__(self, *, stream_id: int) -> None:
        self.stream_id = stream_id
        self.started = threading.Event()
        self.stop_requested = threading.Event()
        self.unwinding_started = threading.Event()
        self.release_unwind = threading.Event()
        self.stop_calls = 0

    async def prepare(self) -> None:
        return None

    async def frames(self):
        yield make_frame(
            stream_id=self.stream_id,
            frame_index=0,
            samples=[0.1, 0.2],
            timestamp_ms=100,
        )
        self.started.set()
        await asyncio.to_thread(self.stop_requested.wait)
        self.unwinding_started.set()
        await asyncio.to_thread(self.release_unwind.wait)

    async def stop(self) -> None:
        self.stop_calls += 1
        self.stop_requested.set()


class RaisingOfflineRecognizer:
    def __init__(self, message: str = "offline boom") -> None:
        self.message = message
        self.requests: list[tuple[np.ndarray, str]] = []

    async def recognize(self, pcm, prompt):
        self.requests.append((pcm.copy(), prompt))
        raise RuntimeError(self.message)

    async def close(self) -> None:
        return None


def make_source(stream_id: int) -> FakeAudioSource:
    finish = make_frame(stream_id=stream_id, frame_index=2, samples=[0.0], timestamp_ms=300)
    return FakeAudioSource(
        [
            make_frame(stream_id=stream_id, frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            make_frame(stream_id=stream_id, frame_index=1, samples=[0.3, 0.4], timestamp_ms=200),
            finish.__class__.finish(stream_id=stream_id, frame_index=2, timestamp_ms=300),
        ],
        pause_between_frames=True,
    )


async def collect_until_finished(bus: FrontendEventBus, lanes: int):
    events = []
    finished = 0
    while finished < lanes:
        event = await asyncio.wait_for(bus.next_event(), timeout=1.0)
        events.append(event)
        if event.status is EventStatus.LANE_FINISHED:
            finished += 1
    return events


@pytest.mark.asyncio
async def test_pipeline_manager_rejects_recognition_before_models_are_loaded():
    from multi_stream_asr.pipeline.manager import PipelineManager

    config = AppConfig(lanes=2)
    manager = PipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        llama_server_manager=RecordingLlamaServerManager(),
        source_factory=lambda lane: make_source(lane.stream_id),
        online_recognizer_factory=lambda lane: FakeOnlineRecognizer([]),
        offline_recognizer_factory=lambda lane: FakeOfflineRecognizer(),
    )

    with pytest.raises(ValueError, match="illegal pipeline state transition: unloaded -> starting"):
        await manager.start_recognition()


@pytest.mark.asyncio
async def test_load_models_starts_servers_and_reaches_loaded_only_after_success():
    from multi_stream_asr.pipeline.manager import PipelineManager

    config = AppConfig(lanes=2)
    bus = FrontendEventBus()
    server_manager = RecordingLlamaServerManager()
    online_calls: list[int] = []
    offline_calls: list[int] = []

    manager = PipelineManager(
        config=config,
        event_bus=bus,
        llama_server_manager=server_manager,
        source_factory=lambda lane: make_source(lane.stream_id),
        online_recognizer_factory=lambda lane: online_calls.append(lane.stream_id) or FakeOnlineRecognizer([]),
        offline_recognizer_factory=lambda lane: offline_calls.append(lane.stream_id) or FakeOfflineRecognizer(),
    )

    assert manager.state is PipelineState.UNLOADED

    await manager.load_models()

    assert manager.state is PipelineState.LOADED
    assert server_manager.start_calls == 1
    assert server_manager.stop_calls == 0
    assert online_calls == [0, 1]
    assert offline_calls == [0, 1]


@pytest.mark.asyncio
async def test_overlapping_lifecycle_calls_wait_for_load_to_finish_before_running():
    from multi_stream_asr.pipeline.manager import PipelineManager

    config = AppConfig(lanes=2)
    server_manager = SlowRecordingLlamaServerManager()
    manager = PipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        llama_server_manager=server_manager,
        source_factory=lambda lane: make_source(lane.stream_id),
        online_recognizer_factory=lambda lane: FakeOnlineRecognizer([]),
        offline_recognizer_factory=lambda lane: FakeOfflineRecognizer(),
    )

    load_task = asyncio.create_task(manager.load_models())
    await asyncio.wait_for(server_manager.started.wait(), timeout=1.0)

    unload_task = asyncio.create_task(manager.unload_models())
    await asyncio.sleep(0.05)
    assert not unload_task.done()
    assert manager.state is PipelineState.LOADING

    server_manager.release.set()
    await load_task
    await unload_task

    assert manager.state is PipelineState.UNLOADED
    assert server_manager.start_calls == 1
    assert server_manager.stop_calls == 1


@pytest.mark.asyncio
async def test_load_failure_stops_already_started_servers():
    from multi_stream_asr.pipeline.manager import PipelineManager

    config = AppConfig(lanes=2)
    server_manager = RecordingLlamaServerManager()

    def online_factory(lane: LaneConfig):
        if lane.stream_id == 1:
            raise RuntimeError("boom")
        return FakeOnlineRecognizer([])

    manager = PipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        llama_server_manager=server_manager,
        source_factory=lambda lane: make_source(lane.stream_id),
        online_recognizer_factory=online_factory,
        offline_recognizer_factory=lambda lane: FakeOfflineRecognizer(),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await manager.load_models()

    assert manager.state is PipelineState.FATAL
    assert server_manager.start_calls == 1
    assert server_manager.stop_calls == 1


@pytest.mark.asyncio
async def test_active_run_can_be_stopped_before_sources_finish():
    from multi_stream_asr.pipeline.manager import PipelineManager

    config = AppConfig(lanes=2)
    sources: dict[int, BlockingAudioSource] = {}

    def source_factory(lane: LaneConfig) -> BlockingAudioSource:
        source = BlockingAudioSource(stream_id=lane.stream_id)
        sources[lane.stream_id] = source
        return source

    manager = PipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        llama_server_manager=RecordingLlamaServerManager(),
        source_factory=source_factory,
        online_recognizer_factory=lambda lane: FakeOnlineRecognizer([]),
        offline_recognizer_factory=lambda lane: FakeOfflineRecognizer(),
    )

    await manager.load_models()
    await asyncio.wait_for(manager.start_recognition(), timeout=0.2)
    await asyncio.wait_for(
        asyncio.gather(*(asyncio.to_thread(source.started.wait) for source in sources.values())),
        timeout=0.2,
    )

    assert manager.state is PipelineState.RUNNING

    await asyncio.wait_for(manager.stop_recognition(), timeout=0.2)

    assert manager.state is PipelineState.LOADED
    assert [manager.lane_metrics[stream_id].final_state for stream_id in range(config.lanes)] == [
        LaneFinalState.STOPPED,
        LaneFinalState.STOPPED,
    ]
    assert [sources[stream_id].stop_calls for stream_id in range(config.lanes)] == [2, 2]


@pytest.mark.asyncio
async def test_second_start_failure_preserves_previous_snapshot_and_loaded_state():
    from multi_stream_asr.pipeline.manager import PipelineManager

    config = AppConfig(lanes=2)
    bus = FrontendEventBus()
    sources: dict[int, BlockingAudioSource] = {}
    source_calls = 0

    def source_factory(lane: LaneConfig) -> BlockingAudioSource:
        nonlocal source_calls
        if source_calls >= config.lanes:
            raise RuntimeError("start boom")
        source_calls += 1
        source = BlockingAudioSource(stream_id=lane.stream_id)
        sources[lane.stream_id] = source
        return source

    manager = PipelineManager(
        config=config,
        event_bus=bus,
        llama_server_manager=RecordingLlamaServerManager(),
        source_factory=source_factory,
        online_recognizer_factory=lambda lane: FakeOnlineRecognizer([]),
        offline_recognizer_factory=lambda lane: FakeOfflineRecognizer(),
    )

    await manager.load_models()
    await asyncio.wait_for(manager.start_recognition(), timeout=0.2)
    await asyncio.wait_for(
        asyncio.gather(*(asyncio.to_thread(source.started.wait) for source in sources.values())),
        timeout=0.2,
    )
    await asyncio.wait_for(manager.stop_recognition(), timeout=0.2)

    bus.publish(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=7,
            status=EventStatus.FINAL,
            text="previous run",
            timestamp_ms=500,
        )
    )

    with pytest.raises(RuntimeError, match="start boom"):
        await manager.start_recognition()

    assert manager.state is PipelineState.LOADED
    assert bus.snapshot_results() == {
        (0, 0): {
            "revision": 7,
            "text": "previous run",
            "status": EventStatus.FINAL,
            "result_type": "offline",
            "is_final": True,
        }
    }
    assert [manager.lane_metrics[stream_id].final_state for stream_id in range(config.lanes)] == [
        LaneFinalState.STOPPED,
        LaneFinalState.STOPPED,
    ]


@pytest.mark.asyncio
async def test_lane_failure_publishes_fatal_event_preserves_snapshot_and_marks_lane_fatal():
    from multi_stream_asr.pipeline.manager import PipelineManager

    config = AppConfig(lanes=2)
    bus = FrontendEventBus()
    blocking_sources: dict[int, BlockingAudioSource] = {}

    def source_factory(lane: LaneConfig):
        if lane.stream_id == 0:
            finish = AudioFrame.finish(stream_id=0, frame_index=2, timestamp_ms=300)
            return FakeAudioSource(
                [
                    make_frame(stream_id=0, frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
                    make_frame(stream_id=0, frame_index=1, samples=[0.3, 0.4], timestamp_ms=200),
                    finish,
                ],
                pause_between_frames=True,
            )
        source = BlockingAudioSource(stream_id=lane.stream_id)
        blocking_sources[lane.stream_id] = source
        return source

    def online_factory(lane: LaneConfig):
        if lane.stream_id == 0:
            return FakeOnlineRecognizer(
                [
                    make_online(text="draft", timestamp_ms=100),
                    make_online(text="draft", timestamp_ms=200, is_endpoint=True),
                ]
            )
        return FakeOnlineRecognizer([])

    def offline_factory(lane: LaneConfig):
        if lane.stream_id == 0:
            return RaisingOfflineRecognizer()
        return FakeOfflineRecognizer()

    manager = PipelineManager(
        config=config,
        event_bus=bus,
        llama_server_manager=RecordingLlamaServerManager(),
        source_factory=source_factory,
        online_recognizer_factory=online_factory,
        offline_recognizer_factory=offline_factory,
    )

    await manager.load_models()
    await manager.start_recognition()

    with pytest.raises(RuntimeError, match="offline boom"):
        await manager.wait_for_completion()

    delivered = [await asyncio.wait_for(bus.next_event(), timeout=0.2) for _ in range(3)]
    assert [(event.status, event.text) for event in delivered] == [
        (EventStatus.PARTIAL, "draft"),
        (EventStatus.PENDING, "draft"),
        (EventStatus.FATAL_ERROR, "offline boom"),
    ]
    assert bus.snapshot_results()[(0, 0)] == {
        "revision": 2,
        "text": "draft",
        "status": EventStatus.PENDING,
        "result_type": delivered[1].result_type,
        "is_final": False,
    }
    assert [lane.final_state for lane in manager.lane_metrics] == [
        LaneFinalState.FATAL,
        LaneFinalState.STOPPED,
    ]
    assert manager.state is PipelineState.FATAL


@pytest.mark.asyncio
async def test_lane_failure_waits_for_sibling_lane_tasks_to_drain_before_completion_returns():
    from multi_stream_asr.pipeline.manager import PipelineManager

    config = AppConfig(lanes=2)
    draining_source: DrainingAudioSource | None = None

    def source_factory(lane: LaneConfig):
        nonlocal draining_source
        if lane.stream_id == 0:
            return FakeAudioSource(
                [
                    make_frame(stream_id=0, frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
                    make_frame(stream_id=0, frame_index=1, samples=[0.3, 0.4], timestamp_ms=200),
                    AudioFrame.finish(stream_id=0, frame_index=2, timestamp_ms=300),
                ],
                pause_between_frames=True,
            )
        draining_source = DrainingAudioSource(stream_id=lane.stream_id)
        return draining_source

    def online_factory(lane: LaneConfig):
        if lane.stream_id == 0:
            return FakeOnlineRecognizer(
                [
                    make_online(text="draft", timestamp_ms=100),
                    make_online(text="draft", timestamp_ms=200, is_endpoint=True),
                ]
            )
        return FakeOnlineRecognizer([])

    manager = PipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        llama_server_manager=RecordingLlamaServerManager(),
        source_factory=source_factory,
        online_recognizer_factory=online_factory,
        offline_recognizer_factory=lambda lane: (
            RaisingOfflineRecognizer("offline boom") if lane.stream_id == 0 else FakeOfflineRecognizer()
        ),
    )

    await manager.load_models()
    await manager.start_recognition()
    assert draining_source is not None
    completion_task = asyncio.create_task(manager.wait_for_completion())

    await asyncio.wait_for(asyncio.to_thread(draining_source.unwinding_started.wait), timeout=0.2)
    await asyncio.sleep(0.02)

    assert not completion_task.done()

    draining_source.release_unwind.set()
    with pytest.raises(RuntimeError, match="offline boom"):
        await completion_task


@pytest.mark.asyncio
async def test_fake_headless_mode_emits_events_saves_output_unloads_and_returns_zero(
    tmp_path: Path,
):
    from multi_stream_asr.headless.runner import HeadlessRunner
    from multi_stream_asr.pipeline.process_manager import ProcessPipelineManager

    config = AppConfig(lanes=2, output_dir=tmp_path)
    bus = FrontendEventBus()
    runner = HeadlessRunner(config=config, fake=True, event_bus=bus)
    collector = asyncio.create_task(collect_until_finished(bus, lanes=config.lanes))

    exit_code = await runner.run()
    events = await collector

    assert exit_code == 0
    assert isinstance(runner.manager, ProcessPipelineManager)
    assert runner.manager.state is PipelineState.UNLOADED
    assert runner.last_run_dir is not None
    assert (runner.last_run_dir / "config.json").is_file()
    assert (runner.last_run_dir / "metrics.json").is_file()
    assert not (runner.last_run_dir / "results.json").exists()
    assert not (runner.last_run_dir / "model_outputs.jsonl").exists()
    assert (runner.last_run_dir / "results" / "lane0.txt").is_file()
    assert (runner.last_run_dir / "results" / "lane1.txt").is_file()
    assert (runner.last_run_dir / "model_outputs" / "lane0.jsonl").is_file()
    assert (runner.last_run_dir / "model_outputs" / "lane1.jsonl").is_file()

    statuses = {event.status for event in events}
    assert EventStatus.PARTIAL in statuses
    assert EventStatus.PENDING in statuses
    assert EventStatus.FINAL in statuses
    assert EventStatus.LANE_FINISHED in statuses

    metrics_payload = json.loads((runner.last_run_dir / "metrics.json").read_text())
    assert metrics_payload["metrics_schema_version"] == 3
    assert [lane["final_state"] for lane in metrics_payload["lanes"]] == ["finished", "finished"]
    assert "event_loop_lag" in metrics_payload
    assert "max_ms" in metrics_payload["event_loop_lag"]
    assert "max_ms" in metrics_payload["online_frame_latency"]
    assert "values" not in metrics_payload["offline_rtf"]
    assert "max" in metrics_payload["offline_rtf"]

    assert (runner.last_run_dir / "results" / "lane0.txt").read_text(encoding="utf-8") == "fake final lane 0\n"
    assert (runner.last_run_dir / "results" / "lane1.txt").read_text(encoding="utf-8") == "fake final lane 1\n"

    model_outputs = [
        json.loads(line)
        for path in sorted((runner.last_run_dir / "model_outputs").glob("lane*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["model"] for record in model_outputs].count("zipformer") == 2
    assert [record["model"] for record in model_outputs].count("qwen3-asr") == 2
    assert set(model_outputs[0]) == {"model", "stream_id", "sentence_id", "audio_seconds", "rtf", "text"}
    assert sorted(
        record["text"]
        for record in model_outputs
        if record["model"] == "qwen3-asr"
    ) == ["fake final lane 0", "fake final lane 1"]


@pytest.mark.asyncio
async def test_headless_load_failure_does_not_create_run_artifacts(tmp_path: Path):
    from multi_stream_asr.headless.runner import HeadlessRunner

    class LoadFailingManager:
        state = PipelineState.UNLOADED
        run_dir = None

        async def load_models(self) -> None:
            self.state = PipelineState.FATAL
            raise RuntimeError("warmup failed")

        async def unload_models(self) -> None:
            self.state = PipelineState.UNLOADED

    class LoadFailingRunner(HeadlessRunner):
        def _build_manager(self):
            return LoadFailingManager()

    runner = LoadFailingRunner(config=AppConfig(lanes=2, output_dir=tmp_path))

    assert await runner.run() == 1
    assert runner.last_run_dir is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_headless_failure_persists_run_artifacts_before_returning_one(tmp_path: Path):
    from multi_stream_asr.headless.runner import HeadlessRunner
    from multi_stream_asr.pipeline.manager import PipelineManager

    class FailingHeadlessRunner(HeadlessRunner):
        def _build_manager(self) -> PipelineManager:
            blocking_sources: dict[int, BlockingAudioSource] = {}

            def source_factory(lane: LaneConfig):
                if lane.stream_id == 0:
                    return FakeAudioSource(
                        [
                            make_frame(
                                stream_id=0,
                                frame_index=0,
                                samples=[0.1, 0.2],
                                timestamp_ms=100,
                            ),
                            make_frame(
                                stream_id=0,
                                frame_index=1,
                                samples=[0.3, 0.4],
                                timestamp_ms=200,
                            ),
                            AudioFrame.finish(stream_id=0, frame_index=2, timestamp_ms=300),
                        ],
                        pause_between_frames=True,
                    )
                source = BlockingAudioSource(stream_id=lane.stream_id)
                blocking_sources[lane.stream_id] = source
                return source

            return PipelineManager(
                config=self._config,
                event_bus=self._event_bus,
                llama_server_manager=RecordingLlamaServerManager(),
                source_factory=source_factory,
                online_recognizer_factory=lambda lane: (
                    FakeOnlineRecognizer(
                        [
                            make_online(text="draft", timestamp_ms=100),
                            make_online(text="draft", timestamp_ms=200, is_endpoint=True),
                        ]
                    )
                    if lane.stream_id == 0
                    else FakeOnlineRecognizer([])
                ),
                offline_recognizer_factory=lambda lane: (
                    RaisingOfflineRecognizer("headless boom")
                    if lane.stream_id == 0
                    else FakeOfflineRecognizer()
                ),
            )

    runner = FailingHeadlessRunner(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
    )

    exit_code = await runner.run()

    assert exit_code == 1
    assert runner.last_run_dir is not None
    assert (runner.last_run_dir / "config.json").is_file()
    assert (runner.last_run_dir / "metrics.json").is_file()
    assert not (runner.last_run_dir / "results.json").exists()
    assert (runner.last_run_dir / "results" / "lane0.txt").is_file()
    assert (runner.last_run_dir / "results" / "lane1.txt").is_file()

    metrics_payload = json.loads((runner.last_run_dir / "metrics.json").read_text())
    assert metrics_payload["lanes"][0]["final_state"] == LaneFinalState.FATAL.value
    assert metrics_payload["lanes"][1]["final_state"] == LaneFinalState.STOPPED.value

    assert (runner.last_run_dir / "results" / "lane0.txt").read_text(encoding="utf-8") == ""
    assert (runner.last_run_dir / "results" / "lane1.txt").read_text(encoding="utf-8") == ""


def test_parse_args_accepts_headless_fake_and_network_overrides(tmp_path: Path):
    from multi_stream_asr.main import parse_args

    args = parse_args(
        [
            "--headless",
            "--lanes",
            "2",
            "--input-dir",
            str(tmp_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--fake",
            "--verbose",
        ]
    )

    assert args.headless is True
    assert args.lanes == 2
    assert args.input_dir == tmp_path
    assert args.host == "0.0.0.0"
    assert args.port == 9001
    assert args.fake is True
    assert args.verbose is True
