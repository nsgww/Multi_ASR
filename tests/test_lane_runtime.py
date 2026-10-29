from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import wave
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from multi_stream_asr.asr.offline.qwen3asr import QwenRequestError, QwenRequestFailure
from multi_stream_asr.config import AppConfig, ModelConfig, ZIPFORMER_WARMUP_AUDIO_PATH
from multi_stream_asr.models import AudioFrame, OnlineRecognition
from multi_stream_asr.pipeline.ipc import LaneIpcMessage
from multi_stream_asr.pipeline.lane_runtime import LaneProcessConfig, LaneRuntime


class MemoryQueue:
    def __init__(self) -> None:
        self.items: queue.Queue[dict[str, Any]] = queue.Queue()

    def put(self, item: dict[str, Any]) -> None:
        self.items.put(item)

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        return self.items.get(timeout=timeout)

    def get_nowait(self) -> dict[str, Any]:
        return self.items.get_nowait()


class CountingMemoryQueue(MemoryQueue):
    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        self.get_calls += 1
        return super().get(timeout=timeout)


class SecondSentenceEndpointRecognizer:
    def __init__(self, lane: object) -> None:
        stream_id = lane.stream_id
        self._outputs = [
            OnlineRecognition(stream_id, 0, 0, "", 100, True),
            OnlineRecognition(stream_id, 1, 0, "fake endpoint lane 0", 200, True),
        ]

    async def start_run(self) -> None:
        return None

    async def accept(self, frame: AudioFrame) -> OnlineRecognition | None:
        if frame.is_finish:
            return None
        return self._outputs.pop(0) if self._outputs else None

    async def reset(self) -> None:
        return None


def messages(queue_obj: MemoryQueue) -> list[LaneIpcMessage]:
    values: list[LaneIpcMessage] = []
    while True:
        try:
            values.append(LaneIpcMessage.from_dict(queue_obj.get(timeout=0.0)))
        except queue.Empty:
            return values


def put_warmup(commands: MemoryQueue) -> None:
    commands.put(
        {
            "type": "warmup",
            "input_path": str(ZIPFORMER_WARMUP_AUDIO_PATH),
        }
    )


def shutdown_after_lane_finished(events: MemoryQueue, commands: MemoryQueue) -> None:
    original_put = events.put

    def queue_shutdown(item: dict[str, Any]) -> None:
        original_put(item)
        message = LaneIpcMessage.from_dict(item)
        if message.type == "lifecycle" and message.payload["state"] == "lane_finished":
            commands.put({"type": "shutdown"})

    events.put = queue_shutdown  # type: ignore[method-assign]


def test_warmup_zipformer_feeds_fixed_wav_and_resets(tmp_path: Path) -> None:
    from multi_stream_asr.pipeline.lane_runtime import warmup_zipformer

    wav_path = tmp_path / "warmup.wav"
    samples = np.arange(22080, dtype="<i2")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    class RecordingRecognizer:
        def __init__(self) -> None:
            self.frames: list[AudioFrame] = []
            self.reset_calls = 0

        async def accept(self, frame: AudioFrame) -> None:
            self.frames.append(frame)

        async def reset(self) -> None:
            self.reset_calls += 1

    recognizer = RecordingRecognizer()
    asyncio.run(
        warmup_zipformer(
            recognizer,
            stream_id=2,
            input_path=wav_path,
            frame_duration_ms=100,
        )
    )

    assert len(recognizer.frames) == 14
    assert all(not frame.is_finish for frame in recognizer.frames)
    assert [frame.frame_index for frame in recognizer.frames] == list(range(14))
    assert recognizer.reset_calls == 1


def test_real_lane_warms_on_owner_thread_before_lane_loaded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    owner_threads: list[int] = []
    recognizer_actions: list[str] = []
    start_run_threads: list[int] = []
    accepted_frames: list[AudioFrame] = []
    reset_threads: list[int] = []
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(
        lanes=2,
        output_dir=tmp_path,
        models=ModelConfig(online_model_dir=tmp_path / "online"),
    )
    lane = config.lane_configs[0]

    class FakeOnlineRecognizer:
        def __init__(self, model_dir: Path, endpoint: object, *, provider_config: Path, **_kwargs) -> None:
            owner_threads.append(threading.get_ident())

        async def start_run(self) -> None:
            owner_threads.append(threading.get_ident())
            start_run_threads.append(threading.get_ident())
            recognizer_actions.append("start_run")

        async def accept(self, frame: AudioFrame) -> None:
            owner_threads.append(threading.get_ident())
            accepted_frames.append(frame)
            recognizer_actions.append("accept")

        async def reset(self) -> None:
            owner_threads.append(threading.get_ident())
            reset_threads.append(threading.get_ident())
            recognizer_actions.append("reset")

    class FakeOfflineRecognizer:
        def __init__(self, *, shared: object, **_kwargs) -> None:
            pass

        async def close(self) -> None:
            pass

    monkeypatch.setattr(lane_runtime, "ZipformerRecognizer", FakeOnlineRecognizer)
    monkeypatch.setattr(lane_runtime, "Qwen3AsrClient", FakeOfflineRecognizer)
    commands.put(
        {
            "type": "warmup",
            "input_path": str(ZIPFORMER_WARMUP_AUDIO_PATH),
        }
    )
    run_dir = tmp_path / "run"
    commands.put({"type": "start", "run_dir": str(run_dir)})
    shutdown_after_lane_finished(events, commands)
    monkeypatch.setattr(
        LaneRuntime,
        "_build_source",
        lambda self, command: lane_runtime._FakeAudioSource(
            self.lane.stream_id,
            self.app_config.frame_duration_ms,
        ),
    )

    runtime = LaneRuntime(
        process_config=LaneProcessConfig(app_config=config, lane=lane),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0
    emitted = messages(events)
    lifecycle = [
        message.payload["state"]
        for message in emitted
        if message.type == "lifecycle"
    ]
    assert lifecycle == [
        "lane_loading",
        "lane_model_loaded",
        "lane_loaded",
        "lane_started",
        "lane_finished",
        "lane_stopped",
    ]
    assert len(accepted_frames) == 16
    assert len(start_run_threads) == 1
    assert start_run_threads[0] == owner_threads[0]
    warmup_reset = recognizer_actions.index("reset")
    assert recognizer_actions[warmup_reset + 1 :] == [
        "start_run",
        "accept",
        "accept",
        "reset",
    ]
    assert len(set(owner_threads)) == 1
    assert len(reset_threads) == 2
    assert [
        message.payload["status"]
        for message in emitted
        if message.type == "asr_event"
    ] == ["lane_finished"]
    assert (run_dir / "results" / "lane0.txt").exists()
    assert (run_dir / "model_outputs" / "lane0.jsonl").exists()


def test_real_lane_loads_zipformer_and_shared_client_without_server(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    order: list[str] = []
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(
        lanes=2,
        output_dir=tmp_path,
        models=ModelConfig(online_model_dir=tmp_path / "online"),
    )
    lane = config.lane_configs[0]

    class FakeOnlineRecognizer:
        def __init__(self, model_dir, endpoint, *, provider_config, **_kwargs) -> None:
            assert model_dir == config.models.online_model_dir
            assert endpoint is config.endpoint
            assert provider_config == config.models.online_model_dir / "lane-0" / "zipformer.config"
            assert provider_config.read_text(encoding="utf-8") == (
                "SPACEMIT_EP_PERFER_CORE_ARCH = 0x5064\n"
                "SPACEMIT_EP_INTRA_THREAD_NUM = 1\n"
                "SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1\n"
            )
            order.append("online_init")

    class FakeOfflineRecognizer:
        def __init__(self, *, shared: object, **_kwargs) -> None:
            assert shared is config.shared_qwen
            order.append("offline_init")

        async def close(self) -> None:
            order.append("offline_close")

    original_put = events.put

    def record_event(item: dict[str, Any]) -> None:
        message = LaneIpcMessage.from_dict(item)
        if message.type == "lifecycle" and message.payload["state"] == "lane_loaded":
            order.append("lane_loaded")
        original_put(item)

    events.put = record_event  # type: ignore[method-assign]
    monkeypatch.setattr(lane_runtime, "ZipformerRecognizer", FakeOnlineRecognizer)
    monkeypatch.setattr(lane_runtime, "Qwen3AsrClient", FakeOfflineRecognizer)
    commands.put({"type": "stop"})

    runtime = LaneRuntime(
        process_config=LaneProcessConfig(app_config=config, lane=lane),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0
    assert order == [
        "online_init",
        "offline_init",
        "offline_close",
    ]
    assert not hasattr(runtime, "_server_manager")
    assert [
        message.payload["state"]
        for message in messages(events)
        if message.type == "lifecycle"
    ] == ["lane_loading", "lane_model_loaded", "lane_stopped"]


def test_real_lane_builds_microphone_source_from_input_descriptor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    created: list[tuple[int, int | str | None, int]] = []

    class FakeMicrophoneSource:
        def __init__(
            self, stream_id: int, device: int | str | None, frame_duration_ms: int
        ) -> None:
            created.append((stream_id, device, frame_duration_ms))

    monkeypatch.setattr(lane_runtime, "MicrophoneSource", FakeMicrophoneSource)
    config = AppConfig(lanes=2, output_dir=tmp_path)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[1],
        ),
        event_queue=MemoryQueue(),
        command_queue=MemoryQueue(),
    )

    source = runtime._build_source(
        {
            "input_kind": "microphone",
            "microphone_device": 20,
            "microphone_device_name": "alsa_input.usb-HM5_unique.analog-stereo",
        }
    )

    assert isinstance(source, FakeMicrophoneSource)
    assert created == [(1, "alsa_input.usb-HM5_unique.analog-stereo", 100)]


def test_real_lane_keeps_legacy_numeric_microphone_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    created: list[int | str | None] = []

    class FakeMicrophoneSource:
        def __init__(
            self, stream_id: int, device: int | str | None, frame_duration_ms: int
        ) -> None:
            created.append(device)

    monkeypatch.setattr(lane_runtime, "MicrophoneSource", FakeMicrophoneSource)
    config = AppConfig(lanes=2, output_dir=tmp_path)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[1],
        ),
        command_queue=MemoryQueue(),
        event_queue=MemoryQueue(),
    )

    runtime._build_source({"input_kind": "microphone", "microphone_device": 3})

    assert created == [3]


def test_lane_stop_before_source_ready_does_not_emit_lane_started(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(tmp_path / "run")})
    commands.put({"type": "stop"})
    shutdown_after_lane_finished(events, commands)

    class BlockingPipeline:
        def __init__(self, *, stream_id: int, **_kwargs: object) -> None:
            self.metrics = lane_runtime.LaneMetrics(stream_id=stream_id)
            self._stopped = asyncio.Event()

        async def run(self, _source: object, *, on_source_ready: object) -> None:
            await self._stopped.wait()

        def request_stop(self) -> None:
            self._stopped.set()

        async def wait_for_offline_tail(self, *, timeout_seconds: float) -> None:
            return None

    class BlockingSource:
        async def stop(self) -> None:
            return None

    monkeypatch.setattr(lane_runtime, "LanePipeline", BlockingPipeline)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )
    monkeypatch.setattr(runtime, "_build_source", lambda _command: BlockingSource())

    assert asyncio.run(asyncio.wait_for(runtime.run(), timeout=0.2)) == 0
    lifecycle_states = [
        message.payload["state"]
        for message in messages(events)
        if message.type == "lifecycle"
    ]
    assert "lane_started" not in lifecycle_states
    assert "lane_finished" in lifecycle_states


def test_fake_lane_run_forwards_events_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    run_dir = tmp_path / "run"
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(run_dir)})
    shutdown_after_lane_finished(events, commands)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[1],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0

    emitted = messages(events)
    statuses = [
        message.payload["status"]
        for message in emitted
        if message.type == "asr_event"
    ]
    assert statuses == ["partial", "partial", "pending", "final", "lane_finished"]
    final_metrics = next(message for message in emitted if message.type == "metrics_final")
    assert final_metrics.payload["lane_metrics"]["final_state"] == "finished"
    paths = next(message for message in emitted if message.type == "paths")
    assert paths.payload == {
        "results_path": "results/lane1.txt",
        "model_outputs_path": "model_outputs/lane1.jsonl",
    }
    assert (run_dir / "results" / "lane1.txt").read_text(encoding="utf-8") == (
        "fake final lane 1\n"
    )
    model_outputs = [
        json.loads(line)
        for line in (run_dir / "model_outputs" / "lane1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["model"] for record in model_outputs] == ["zipformer", "qwen3-asr"]
    assert runtime._pipeline is not None
    assert runtime._pipeline.online_recognizer is runtime._online_recognizer
    assert runtime._pipeline.offline_recognizer is runtime._offline_recognizer


def test_qwen_success_ipc_reports_run_and_sentence_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    monkeypatch.setattr(
        lane_runtime,
        "_FakeOnlineRecognizer",
        SecondSentenceEndpointRecognizer,
    )
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    run_dir = tmp_path / "run"
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(run_dir)})
    shutdown_after_lane_finished(events, commands)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0

    emitted = messages(events)
    qwen_successes = [message for message in emitted if message.type == "qwen_success"]
    assert len(qwen_successes) == 1
    assert qwen_successes[0].payload == {
        "run_id": run_dir.name,
        "sentence_id": 1,
    }


def test_qwen_error_ipc_reports_structured_request_without_fatal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    class FailingOfflineRecognizer:
        def __init__(self, lane: object) -> None:
            return None

        async def recognize(self, pcm: np.ndarray, prompt: str) -> None:
            raise QwenRequestError(
                QwenRequestFailure(
                    category="http_status",
                    error_type="HTTPStatusError",
                    message="HTTP 500",
                    method="POST",
                    url="http://qwen/v1/chat/completions",
                    timeout_ms=120000,
                    audio_seconds=1.0,
                    sample_count=16000,
                    status_code=500,
                    response_body="decoder failed",
                )
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        lane_runtime,
        "_FakeOnlineRecognizer",
        SecondSentenceEndpointRecognizer,
    )
    monkeypatch.setattr(
        lane_runtime,
        "_FakeOfflineRecognizer",
        FailingOfflineRecognizer,
    )
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    run_dir = tmp_path / "run"
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(run_dir)})
    shutdown_after_lane_finished(events, commands)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0

    emitted = messages(events)
    qwen_error = next(message for message in emitted if message.type == "qwen_error")
    assert qwen_error.payload == {
        "run_id": run_dir.name,
        "sentence_id": 1,
        "revision": 3,
        "pending_text": "fake endpoint lane 0",
        "timestamp_ms": 200,
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
    assert json.loads(json.dumps(qwen_error.to_dict(), allow_nan=False)) == qwen_error.to_dict()
    assert not any(message.type == "fatal" for message in emitted)


def test_verbose_lane_run_writes_metrics_detail_files(tmp_path: Path) -> None:
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path, verbose=True)
    run_dir = tmp_path / "run"
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(run_dir), "loop_audio": False})
    shutdown_after_lane_finished(events, commands)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[1],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0

    detail_dir = run_dir / "metrics_detail"
    frame_latency = [
        json.loads(line)
        for line in (detail_dir / "frame_latency" / "lane1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rtf_records = [
        json.loads(line)
        for line in (detail_dir / "rtf" / "lane1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(frame_latency) >= 1
    assert all(set(entry) == {"t_ms", "latency_ms"} for entry in frame_latency)
    assert rtf_records == [
        {
            "t_ms": rtf_records[0]["t_ms"],
            "sentence_id": 0,
            "audio_seconds": 1.0,
            "elapsed_ms": 400.0,
            "rtf": 0.4,
        }
    ]
    model_outputs_sentence_ids = {
        json.loads(line)["sentence_id"]
        for line in (run_dir / "model_outputs" / "lane1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert {record["sentence_id"] for record in rtf_records} <= model_outputs_sentence_ids
    assert (detail_dir / "event_loop_lag" / "lane1.jsonl").exists()
    assert (detail_dir / "shared_timing" / "lane1.jsonl").exists()


def test_non_verbose_lane_run_writes_no_metrics_detail_directory(tmp_path: Path) -> None:
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path, verbose=False)
    run_dir = tmp_path / "run"
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(run_dir), "loop_audio": False})
    shutdown_after_lane_finished(events, commands)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[1],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0

    assert not (run_dir / "metrics_detail").exists()


def test_verbose_loop_audio_lane_run_writes_no_metrics_detail_directory(tmp_path: Path) -> None:
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path, verbose=True)
    run_dir = tmp_path / "run"
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(run_dir), "loop_audio": True})
    shutdown_after_lane_finished(events, commands)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[1],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0

    assert not (run_dir / "metrics_detail").exists()
    assert (run_dir / "results" / "lane1.txt").read_text(encoding="utf-8") == ""


def test_fake_lane_reuses_loaded_models_for_a_second_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    recognizer_instances: list[Any] = []

    class RecordingOnlineRecognizer:
        def __init__(self, lane: object) -> None:
            self.actions: list[str] = []
            recognizer_instances.append(self)

        async def start_run(self) -> None:
            self.actions.append("start_run")

        async def accept(self, frame: AudioFrame) -> None:
            self.actions.append("accept")

        async def reset(self) -> None:
            self.actions.append("reset")

    monkeypatch.setattr(
        lane_runtime,
        "_FakeOnlineRecognizer",
        RecordingOnlineRecognizer,
    )
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    first_run_dir = tmp_path / "first"
    second_run_dir = tmp_path / "second"
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(first_run_dir)})
    original_put = events.put
    completed_runs = 0

    def queue_next_run(item: dict[str, Any]) -> None:
        nonlocal completed_runs
        original_put(item)
        message = LaneIpcMessage.from_dict(item)
        if message.type != "lifecycle" or message.payload["state"] != "lane_finished":
            return
        completed_runs += 1
        if completed_runs == 1:
            commands.put({"type": "start", "run_dir": str(second_run_dir)})
        else:
            commands.put({"type": "shutdown"})

    events.put = queue_next_run  # type: ignore[method-assign]
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0

    lifecycle_states = [
        message.payload["state"]
        for message in messages(events)
        if message.type == "lifecycle"
    ]
    assert lifecycle_states == [
        "lane_loading",
        "lane_model_loaded",
        "lane_loaded",
        "lane_started",
        "lane_finished",
        "lane_started",
        "lane_finished",
        "lane_stopped",
    ]
    assert len(recognizer_instances) == 1
    assert runtime._online_recognizer is recognizer_instances[0]
    assert recognizer_instances[0].actions == [
        "start_run",
        "accept",
        "accept",
        "reset",
        "start_run",
        "accept",
        "accept",
        "reset",
    ]
    assert (first_run_dir / "results" / "lane0.txt").exists()
    assert (second_run_dir / "results" / "lane0.txt").exists()


def test_running_lane_emits_only_compact_json_metrics_snapshots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(tmp_path / "run")})
    shutdown_after_lane_finished(events, commands)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0

    emitted = messages(events)
    snapshots = [message for message in emitted if message.type == "metrics_snapshot"]
    assert snapshots
    assert set(snapshots[0].payload) == {
        "lane_metrics",
        "online_frame_latency",
        "offline_rtf",
        "offline_shared_timings",
        "event_loop_lag",
    }
    forbidden_keys = {"pcm", "raw_response", "file_content", "audio"}
    for message in emitted:
        encoded = json.dumps(message.to_dict(), allow_nan=False)
        assert encoded
        assert forbidden_keys.isdisjoint(message.payload)


def test_stop_command_interrupts_source_then_drains_offline_tail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.asr.offline.base import OfflineRecognitionResult
    from multi_stream_asr.models import AudioFrame, OnlineRecognition
    from multi_stream_asr.pipeline import lane_runtime

    class LongRunningSource:
        instance: "LongRunningSource | None" = None

        def __init__(self, stream_id: int, frame_duration_ms: int) -> None:
            self.stream_id = stream_id
            self.frame_duration_ms = frame_duration_ms
            self.stop_calls = 0
            self.stopped = False
            type(self).instance = self

        async def prepare(self) -> None:
            return None

        async def frames(self):
            yield AudioFrame(
                self.stream_id,
                0,
                np.asarray([0.1, 0.2], dtype=np.float32),
                self.frame_duration_ms,
            )
            while not self.stopped:
                await asyncio.sleep(0.005)
            yield AudioFrame.finish(self.stream_id, 1, self.frame_duration_ms * 2)

        async def stop(self) -> None:
            self.stop_calls += 1
            self.stopped = True

    class EndpointOnlineRecognizer:
        def __init__(self, lane) -> None:
            self.lane = lane

        async def accept(self, frame):
            return OnlineRecognition(
                self.lane.stream_id,
                0,
                0,
                "stop endpoint",
                frame.timestamp_ms,
                True,
            )

        async def reset(self) -> None:
            return None

    class BlockingOfflineRecognizer:
        instance: "BlockingOfflineRecognizer | None" = None

        def __init__(self, lane) -> None:
            self.lane = lane
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = False
            type(self).instance = self

        async def recognize(self, pcm, prompt):
            self.started.set()
            await self.release.wait()
            return OfflineRecognitionResult(
                text="drained after stop",
                elapsed_ms=100.0,
                audio_seconds=1.0,
                rtf=0.1,
            )

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(lane_runtime, "_FakeAudioSource", LongRunningSource)
    monkeypatch.setattr(lane_runtime, "_FakeOnlineRecognizer", EndpointOnlineRecognizer)
    monkeypatch.setattr(lane_runtime, "_FakeOfflineRecognizer", BlockingOfflineRecognizer)
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(tmp_path / "run")})
    shutdown_after_lane_finished(events, commands)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    async def run_and_stop() -> None:
        run_task = asyncio.create_task(runtime.run())
        while BlockingOfflineRecognizer.instance is None:
            await asyncio.sleep(0)
        offline = BlockingOfflineRecognizer.instance
        assert offline is not None
        await asyncio.wait_for(offline.started.wait(), timeout=0.5)
        commands.put({"type": "stop", "reason": "user_requested"})
        try:
            for _ in range(50):
                source = LongRunningSource.instance
                if source is not None and source.stop_calls:
                    break
                await asyncio.sleep(0.005)
            source = LongRunningSource.instance
            assert source is not None
            assert source.stop_calls >= 1
            assert not run_task.done()
            offline.release.set()
            assert await asyncio.wait_for(run_task, timeout=0.5) == 0
            assert offline.closed is True
        finally:
            offline.release.set()
            if not run_task.done():
                run_task.cancel()
                with suppress(asyncio.CancelledError):
                    await run_task

    asyncio.run(run_and_stop())

    emitted = messages(events)
    statuses = [
        message.payload["status"]
        for message in emitted
        if message.type == "asr_event"
    ]
    assert statuses == ["partial", "pending", "final"]
    final_metrics = next(message for message in emitted if message.type == "metrics_final")
    assert final_metrics.payload["lane_metrics"]["final_state"] == "stopped"


def test_fatal_cancel_stops_in_flight_offline_request_without_tail_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.asr.offline.base import OfflineRecognitionResult
    from multi_stream_asr.pipeline import lane_runtime

    class LongRunningSource:
        def __init__(self, stream_id: int, frame_duration_ms: int) -> None:
            self.stream_id = stream_id
            self.frame_duration_ms = frame_duration_ms
            self._stopped = False

        async def prepare(self) -> None:
            return None

        async def frames(self):
            yield AudioFrame(
                self.stream_id,
                0,
                np.asarray([0.1, 0.2], dtype=np.float32),
                self.frame_duration_ms,
            )
            while not self._stopped:
                await asyncio.sleep(0.005)
            yield AudioFrame.finish(self.stream_id, 1, self.frame_duration_ms * 2)

        async def stop(self) -> None:
            self._stopped = True

    class EndpointOnlineRecognizer:
        def __init__(self, lane) -> None:
            self.lane = lane

        async def accept(self, frame):
            return OnlineRecognition(
                self.lane.stream_id,
                0,
                0,
                "fatal endpoint",
                frame.timestamp_ms,
                True,
            )

        async def reset(self) -> None:
            return None

    class BlockingOfflineRecognizer:
        instance: "BlockingOfflineRecognizer | None" = None

        def __init__(self, lane) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            type(self).instance = self

        async def recognize(self, pcm, prompt):
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return OfflineRecognitionResult(
                text="unexpected offline completion",
                elapsed_ms=100.0,
                audio_seconds=1.0,
                rtf=0.1,
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(lane_runtime, "_FakeAudioSource", LongRunningSource)
    monkeypatch.setattr(lane_runtime, "_FakeOnlineRecognizer", EndpointOnlineRecognizer)
    monkeypatch.setattr(lane_runtime, "_FakeOfflineRecognizer", BlockingOfflineRecognizer)
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(tmp_path / "run")})
    shutdown_after_lane_finished(events, commands)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    async def run_and_stop_for_sibling_fatal() -> None:
        run_task = asyncio.create_task(runtime.run())
        while BlockingOfflineRecognizer.instance is None:
            await asyncio.sleep(0)
        offline = BlockingOfflineRecognizer.instance
        assert offline is not None
        await asyncio.wait_for(offline.started.wait(), timeout=0.5)
        commands.put({"type": "stop", "reason": "sibling_fatal"})
        try:
            assert await asyncio.wait_for(asyncio.shield(run_task), timeout=1.0) == 0
            assert offline.cancelled.is_set()
        finally:
            offline.release.set()
            if not run_task.done():
                await asyncio.wait_for(run_task, timeout=0.5)

    asyncio.run(run_and_stop_for_sibling_fatal())

    lifecycle = [
        message.payload["state"]
        for message in messages(events)
        if message.type == "lifecycle"
    ]
    assert lifecycle[-1] == "lane_stopped"
    assert all(message.type != "qwen_error" for message in messages(events))


def test_runtime_command_watcher_cancels_without_blocking_asyncio_shutdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    events = MemoryQueue()
    commands = CountingMemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(tmp_path / "run")})
    shutdown_after_lane_finished(events, commands)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    started = time.monotonic()
    assert asyncio.run(runtime.run()) == 0
    elapsed = time.monotonic() - started

    assert commands.get_calls >= 2
    assert elapsed < 0.5


def test_snapshot_failure_stops_indefinite_pipeline_and_drains_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.asr.offline.base import OfflineRecognitionResult
    from multi_stream_asr.models import AudioFrame, OnlineRecognition
    from multi_stream_asr.pipeline import lane_runtime

    offline_started = threading.Event()
    source_stopped = threading.Event()
    offline_release = asyncio.Event()
    drain_called = asyncio.Event()

    class IndefiniteSource:
        def __init__(self, stream_id: int, frame_duration_ms: int) -> None:
            self.stream_id = stream_id
            self.frame_duration_ms = frame_duration_ms

        async def prepare(self) -> None:
            return None

        async def frames(self):
            yield AudioFrame(
                self.stream_id,
                0,
                np.asarray([0.1, 0.2], dtype=np.float32),
                self.frame_duration_ms,
            )
            while not source_stopped.is_set():
                await asyncio.sleep(0.005)

        async def stop(self) -> None:
            source_stopped.set()

    class EndpointOnlineRecognizer:
        def __init__(self, lane) -> None:
            self.lane = lane

        async def accept(self, frame):
            return OnlineRecognition(
                self.lane.stream_id,
                0,
                0,
                "supervised endpoint",
                frame.timestamp_ms,
                True,
            )

        async def reset(self) -> None:
            return None

    class DelayedOfflineRecognizer:
        def __init__(self, lane) -> None:
            return None

        async def recognize(self, pcm, prompt):
            offline_started.set()
            await offline_release.wait()
            return OfflineRecognitionResult(
                text="supervised final",
                elapsed_ms=100.0,
                audio_seconds=1.0,
                rtf=0.1,
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(lane_runtime, "_FakeAudioSource", IndefiniteSource)
    monkeypatch.setattr(lane_runtime, "_FakeOnlineRecognizer", EndpointOnlineRecognizer)
    monkeypatch.setattr(lane_runtime, "_FakeOfflineRecognizer", DelayedOfflineRecognizer)
    original_wait_for_offline_tail = lane_runtime.LanePipeline.wait_for_offline_tail

    async def record_offline_tail_wait(pipeline, *, timeout_seconds):
        drain_called.set()
        return await original_wait_for_offline_tail(
            pipeline,
            timeout_seconds=timeout_seconds,
        )

    monkeypatch.setattr(
        lane_runtime.LanePipeline,
        "wait_for_offline_tail",
        record_offline_tail_wait,
    )
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    run_dir = tmp_path / "run"
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(run_dir)})
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    async def fail_snapshot() -> None:
        await asyncio.to_thread(offline_started.wait)
        raise RuntimeError("metrics snapshot failed")

    monkeypatch.setattr(runtime, "_emit_metrics_snapshots", fail_snapshot)

    async def run_failure() -> int:
        run_task = asyncio.create_task(runtime.run())
        try:
            await asyncio.wait_for(asyncio.to_thread(source_stopped.wait), timeout=0.5)
            await asyncio.wait_for(drain_called.wait(), timeout=0.5)
            offline_release.set()
            return await asyncio.wait_for(run_task, timeout=0.5)
        finally:
            commands.put({"type": "stop"})
            source_stopped.set()
            offline_release.set()

    assert asyncio.run(run_failure()) == 1

    emitted = messages(events)
    fatal = next(message for message in emitted if message.type == "fatal")
    assert fatal.payload["message"] == "metrics snapshot failed"
    statuses = [
        message.payload["status"]
        for message in emitted
        if message.type == "asr_event"
    ]
    assert statuses.count("fatal_error") == 1
    assert "final" in statuses


def test_load_failure_emits_fatal_outputs_without_constructing_shared_client(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    order: list[str] = []
    events = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)

    def fail_online_load(*args, **kwargs):
        raise RuntimeError("zipformer load failed")

    def fail_offline_construction(*, shared):
        raise AssertionError("Qwen client must not be built after Zipformer fails")

    original_put = events.put

    def record_event(item: dict[str, Any]) -> None:
        message = LaneIpcMessage.from_dict(item)
        if message.type == "lifecycle":
            order.append(str(message.payload["state"]))
        else:
            order.append(message.type)
        original_put(item)

    events.put = record_event  # type: ignore[method-assign]
    monkeypatch.setattr(lane_runtime, "ZipformerRecognizer", fail_online_load)
    monkeypatch.setattr(lane_runtime, "Qwen3AsrClient", fail_offline_construction)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
        ),
        event_queue=events,
        command_queue=MemoryQueue(),
    )

    assert asyncio.run(runtime.run()) == 1

    emitted = messages(events)
    fatal = next(message for message in emitted if message.type == "fatal")
    assert fatal.payload["phase"] == "load"
    assert fatal.payload["error_type"] == "RuntimeError"
    assert all(message.type != "metrics_final" for message in emitted)
    assert all(message.type != "asr_event" for message in emitted)
    assert all(message.type != "paths" for message in emitted)
    assert order == [
        "lane_loading",
        "fatal",
        "lane_failed",
        "lane_stopped",
    ]


def test_shutdown_before_start_emits_stopped_metrics_without_artifact_paths(
    tmp_path: Path,
) -> None:
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    commands.put({"type": "shutdown"})
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0

    emitted = messages(events)
    final_metrics = next(message for message in emitted if message.type == "metrics_final")
    assert final_metrics.payload["lane_metrics"] == {
        "stream_id": 0,
        "processed_frames": 0,
        "dropped_frames": 0,
        "backlogged_frames": 0,
        "offline_requests": 0,
        "final_state": "stopped",
    }
    assert "results_path" not in final_metrics.payload
    assert "model_outputs_path" not in final_metrics.payload
    assert all(message.type != "paths" for message in emitted)


def test_run_failure_preserves_metrics_emits_fatal_event_and_closes_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.models import AudioFrame
    from multi_stream_asr.pipeline import lane_runtime

    class FailingSource:
        def __init__(self, stream_id: int, frame_duration_ms: int) -> None:
            self.stream_id = stream_id
            self.frame_duration_ms = frame_duration_ms
            self.stopped = False

        async def prepare(self) -> None:
            return None

        async def frames(self):
            yield AudioFrame(
                self.stream_id,
                0,
                np.asarray([0.1, 0.2], dtype=np.float32),
                self.frame_duration_ms,
            )
            await asyncio.sleep(0.02)
            raise RuntimeError("audio source failed")

        async def stop(self) -> None:
            self.stopped = True

    class ClosingOfflineRecognizer:
        instance: "ClosingOfflineRecognizer | None" = None

        def __init__(self, lane) -> None:
            self.closed = False
            type(self).instance = self

        async def recognize(self, pcm, prompt):
            raise AssertionError("offline recognition is not expected")

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(lane_runtime, "_FakeAudioSource", FailingSource)
    monkeypatch.setattr(lane_runtime, "_FakeOfflineRecognizer", ClosingOfflineRecognizer)
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    run_dir = tmp_path / "run"
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(run_dir)})
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 1

    emitted = messages(events)
    fatal = next(message for message in emitted if message.type == "fatal")
    assert fatal.payload["phase"] == "run"
    statuses = [
        message.payload["status"]
        for message in emitted
        if message.type == "asr_event"
    ]
    assert statuses[-1] == "fatal_error"
    final_metrics = next(message for message in emitted if message.type == "metrics_final")
    assert final_metrics.payload["lane_metrics"]["processed_frames"] == 1
    assert final_metrics.payload["lane_metrics"]["final_state"] == "fatal"
    message_types = [message.type for message in emitted]
    assert message_types.index("metrics_final") < message_types.index("paths")
    assert message_types.index("paths") < message_types.index("fatal")
    assert message_types.index("fatal") < next(
        index
        for index, message in enumerate(emitted)
        if message.type == "lifecycle" and message.payload["state"] == "lane_failed"
    )
    assert (run_dir / "results" / "lane0.txt").is_file()
    assert (run_dir / "model_outputs" / "lane0.jsonl").is_file()
    offline = ClosingOfflineRecognizer.instance
    assert offline is not None
    assert offline.closed is True


def test_run_failure_drains_active_offline_before_closing_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.asr.offline.base import OfflineRecognitionResult
    from multi_stream_asr.models import AudioFrame, OnlineRecognition
    from multi_stream_asr.pipeline import lane_runtime

    offline_started = threading.Event()
    source_failed = threading.Event()
    offline_release = asyncio.Event()
    drain_called = asyncio.Event()

    class FailingAfterEndpointSource:
        def __init__(self, stream_id: int, frame_duration_ms: int) -> None:
            self.stream_id = stream_id
            self.frame_duration_ms = frame_duration_ms

        async def prepare(self) -> None:
            return None

        async def frames(self):
            yield AudioFrame(
                self.stream_id,
                0,
                np.asarray([0.1, 0.2], dtype=np.float32),
                self.frame_duration_ms,
            )
            offline_started.wait()
            source_failed.set()
            raise RuntimeError("source failed with offline active")

        async def stop(self) -> None:
            offline_started.set()

    class EndpointOnlineRecognizer:
        def __init__(self, lane) -> None:
            self.lane = lane

        async def accept(self, frame):
            return OnlineRecognition(
                self.lane.stream_id,
                0,
                0,
                "active endpoint",
                frame.timestamp_ms,
                True,
            )

        async def reset(self) -> None:
            return None

    class DelayedOfflineRecognizer:
        def __init__(self, lane) -> None:
            self.closed = False

        async def recognize(self, pcm, prompt):
            offline_started.set()
            await offline_release.wait()
            return OfflineRecognitionResult(
                text="drained final after failure",
                elapsed_ms=100.0,
                audio_seconds=1.0,
                rtf=0.1,
            )

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(lane_runtime, "_FakeAudioSource", FailingAfterEndpointSource)
    monkeypatch.setattr(lane_runtime, "_FakeOnlineRecognizer", EndpointOnlineRecognizer)
    monkeypatch.setattr(lane_runtime, "_FakeOfflineRecognizer", DelayedOfflineRecognizer)
    original_result_writer_close = lane_runtime.LaneResultWriter.close

    def fail_result_writer_close(writer) -> None:
        original_result_writer_close(writer)
        raise RuntimeError("result writer close failed")

    monkeypatch.setattr(
        lane_runtime.LaneResultWriter,
        "close",
        fail_result_writer_close,
    )
    original_wait_for_offline_tail = lane_runtime.LanePipeline.wait_for_offline_tail

    async def record_offline_tail_wait(pipeline, *, timeout_seconds):
        drain_called.set()
        return await original_wait_for_offline_tail(
            pipeline,
            timeout_seconds=timeout_seconds,
        )

    monkeypatch.setattr(
        lane_runtime.LanePipeline,
        "wait_for_offline_tail",
        record_offline_tail_wait,
    )
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    run_dir = tmp_path / "run"
    put_warmup(commands)
    commands.put({"type": "start", "run_dir": str(run_dir)})
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    async def run_failure() -> int:
        run_task = asyncio.create_task(runtime.run())
        try:
            await asyncio.wait_for(asyncio.to_thread(source_failed.wait), timeout=0.5)
            await asyncio.wait_for(drain_called.wait(), timeout=0.5)
            offline_release.set()
            return await asyncio.wait_for(run_task, timeout=0.5)
        finally:
            offline_started.set()
            offline_release.set()

    assert asyncio.run(run_failure()) == 1

    emitted = messages(events)
    fatal = next(message for message in emitted if message.type == "fatal")
    assert fatal.payload["message"] == "source failed with offline active"
    statuses = [
        message.payload["status"]
        for message in emitted
        if message.type == "asr_event"
    ]
    assert statuses.count("fatal_error") == 1
    assert "final" in statuses
    final_metrics = next(message for message in emitted if message.type == "metrics_final")
    assert final_metrics.payload["lane_metrics"]["offline_requests"] == 1
    assert final_metrics.payload["lane_metrics"]["final_state"] == "fatal"
    assert (run_dir / "results" / "lane0.txt").read_text(encoding="utf-8") == (
        "drained final after failure\n"
    )
    model_outputs = (run_dir / "model_outputs" / "lane0.jsonl").read_text(
        encoding="utf-8"
    )
    assert "drained final after failure" in model_outputs


def test_stop_failure_reports_phase_before_stopped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    events = MemoryQueue()
    config = AppConfig(
        lanes=2,
        output_dir=tmp_path,
        models=ModelConfig(online_model_dir=tmp_path / "online"),
    )

    class FakeOnlineRecognizer:
        def __init__(self, model_dir, endpoint, *, provider_config, **_kwargs) -> None:
            return None

    class FailingCloseOfflineRecognizer:
        def __init__(self, *, shared, **_kwargs) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("offline close failed")

    monkeypatch.setattr(lane_runtime, "ZipformerRecognizer", FakeOnlineRecognizer)
    monkeypatch.setattr(
        lane_runtime,
        "Qwen3AsrClient",
        FailingCloseOfflineRecognizer,
    )
    commands = MemoryQueue()
    commands.put({"type": "shutdown"})
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
        ),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 1

    emitted = messages(events)
    fatal = next(message for message in emitted if message.type == "fatal")
    assert fatal.payload["phase"] == "stop"
    failed_index = next(
        index
        for index, message in enumerate(emitted)
        if message.type == "lifecycle" and message.payload["state"] == "lane_failed"
    )
    stopped_index = next(
        index
        for index, message in enumerate(emitted)
        if message.type == "lifecycle" and message.payload["state"] == "lane_stopped"
    )
    assert failed_index < stopped_index


def test_track_event_loop_lag_records_into_writer(tmp_path: Path) -> None:
    from multi_stream_asr.log.lane_artifacts import DetailJsonlWriter

    config = AppConfig(lanes=2, output_dir=tmp_path, verbose=True)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[0],
            fake=True,
        ),
        event_queue=MemoryQueue(),
        command_queue=MemoryQueue(),
    )
    writer = DetailJsonlWriter(tmp_path / "event_loop_lag" / "lane0.jsonl")
    runtime._event_loop_lag_writer = writer

    async def drive() -> None:
        task = asyncio.create_task(runtime._track_event_loop_lag(interval_seconds=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    writer.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "event_loop_lag" / "lane0.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) >= 1
    assert all(set(entry) == {"t_ms", "lag_ms"} for entry in records)
