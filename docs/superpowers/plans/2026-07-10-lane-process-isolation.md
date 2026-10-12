# Lane Process Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move each ASR lane into its own Python child process so the lane owns audio input, Zipformer, Qwen3, per-lane files, and per-lane performance metrics while the main process only orchestrates and writes global summaries.

**Architecture:** Keep the existing in-process `LanePipeline` as the per-lane execution unit, but run it inside a new lane child process. The main process uses process controllers and JSON-able IPC messages to drive load/start/stop/shutdown, publish ASR events to the existing `FrontendEventBus`, and aggregate final metrics into schema v3.

**Tech Stack:** Python 3.13, `asyncio`, `multiprocessing` with `spawn`, existing FastAPI/WebSocket event bus, existing sherpa-onnx Zipformer wrapper, existing Qwen3 `llama-server` manager/client, pytest.

## Global Constraints

- Use Python 3.13 only.
- Use a project-local `.venv313` virtual environment on remote devices.
- Export the Spacemit and Aliyun package index variables before installing Python packages on remote devices.
- Install dependencies only after activating the Python 3.13 virtual environment.
- Do not bind Linux threads directly in this project; A100/X100 binding remains in sherpa-onnx, SpaceMiT EP, and `llama-server` configuration.
- Do not pass PCM, `numpy.ndarray`, segment cache content, Qwen raw responses, or model output file content over IPC.
- `results/laneX.txt` and `model_outputs/laneX.jsonl` must be written by the lane child process that owns lane X.
- The main process writes global `config.json` and `metrics.json`.
- Keep online audio input at 100 ms per frame through the existing producer thread plus `OnlineFrameQueue(maxsize=1)` model.
- Target board validation remains online frame latency about 50 ms and whole-segment RTF about 0.4.

---

## File Structure

Create:

- `src/multi_stream_asr/pipeline/ipc.py`
  - Defines JSON-able IPC message helpers, ASR event conversion, lane metrics conversion, and timestamp helpers.
- `src/multi_stream_asr/log/lane_artifacts.py`
  - Writes `results/laneX.txt` and `model_outputs/laneX.jsonl` from inside the lane child process.
- `src/multi_stream_asr/pipeline/lane_runtime.py`
  - Runs one lane inside a child process: starts Qwen3, loads Zipformer, waits for `start`, runs `LanePipeline`, reports events/metrics/paths/fatal.
- `src/multi_stream_asr/pipeline/process_manager.py`
  - Main-process lifecycle manager that spawns lane processes, reads IPC events, aggregates state, and exposes the same manager surface used by headless/web.
- `tests/test_ipc.py`
  - Covers IPC serialization and dataclass conversion.
- `tests/test_lane_artifacts.py`
  - Covers per-lane results/model outputs writers.
- `tests/test_process_manager.py`
  - Covers main-process event aggregation, fatal propagation, and fake lane end-to-end behavior.

Modify:

- `src/multi_stream_asr/asr/offline/llama_server.py`
  - Add single-lane start support without global cleanup in each lane child process.
- `src/multi_stream_asr/headless/runner.py`
  - Use `ProcessPipelineManager` by default; keep the current in-process fake classes only where tests still need them or move them into lane runtime fake mode.
- `src/multi_stream_asr/web/server.py`
  - Record process-run summaries when the manager exposes process run artifacts.
- `src/multi_stream_asr/log/metrics.py`
  - Add summary merge helpers for latency-like and RTF-like metrics.
- `src/multi_stream_asr/log/recorder.py`
  - Add schema v3 process-run recorder method that writes only config/global metrics and never rewrites lane result/model output files.
- `src/multi_stream_asr/pipeline/__init__.py`
  - Export `ProcessPipelineManager`.
- `docs/thread-model.md`
  - Rewrite the thread/process model for lane child processes.
- Existing tests:
  - Update recorder/headless/web tests to cover process mode while preserving current in-process `PipelineManager` unit tests.

---

### Task 1: IPC Message Contract

**Files:**
- Create: `src/multi_stream_asr/pipeline/ipc.py`
- Test: `tests/test_ipc.py`

**Interfaces:**
- Produces: `LaneIpcMessage`, `make_message(...)`, `asr_event_to_payload(...)`, `asr_event_from_payload(...)`, `lane_metrics_to_payload(...)`, `lane_metrics_from_payload(...)`, `now_ms()`.
- Consumes: Existing `ASREvent`, `EventStatus`, `ResultType`, `LaneMetrics`, `LaneFinalState`.

- [ ] **Step 1: Write failing IPC tests**

Create `tests/test_ipc.py`:

```python
from __future__ import annotations

from multi_stream_asr.models import ASREvent, EventStatus, LaneFinalState, LaneMetrics
from multi_stream_asr.pipeline.ipc import (
    LaneIpcMessage,
    asr_event_from_payload,
    asr_event_to_payload,
    lane_metrics_from_payload,
    lane_metrics_to_payload,
    make_message,
)


def test_lane_ipc_message_round_trips_jsonable_payload():
    message = make_message(
        message_type="lifecycle",
        stream_id=2,
        seq=7,
        payload={"state": "lane_loaded", "message": ""},
        time_ms=123456,
    )

    encoded = message.to_dict()
    decoded = LaneIpcMessage.from_dict(encoded)

    assert encoded == {
        "type": "lifecycle",
        "stream_id": 2,
        "seq": 7,
        "time_ms": 123456,
        "payload": {"state": "lane_loaded", "message": ""},
    }
    assert decoded == message


def test_asr_event_payload_round_trips_enums_as_strings():
    event = ASREvent.asr_result(
        stream_id=1,
        sentence_id=3,
        revision=4,
        status=EventStatus.FINAL,
        text="final text",
        timestamp_ms=900,
    )

    payload = asr_event_to_payload(event)
    decoded = asr_event_from_payload(payload)

    assert payload["status"] == "final"
    assert payload["result_type"] == "offline"
    assert decoded == event


def test_lane_metrics_payload_round_trips_final_state():
    metrics = LaneMetrics(
        stream_id=3,
        processed_frames=11,
        dropped_frames=2,
        offline_requests=4,
        final_state=LaneFinalState.FINISHED,
    )

    payload = lane_metrics_to_payload(metrics)
    decoded = lane_metrics_from_payload(payload)

    assert payload["final_state"] == "finished"
    assert decoded == metrics
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_ipc.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'multi_stream_asr.pipeline.ipc'`.

- [ ] **Step 3: Implement IPC helpers**

Create `src/multi_stream_asr/pipeline/ipc.py`:

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from multi_stream_asr.models import ASREvent, EventStatus, LaneFinalState, LaneMetrics, ResultType


LaneMessageType = str


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class LaneIpcMessage:
    type: LaneMessageType
    stream_id: int
    seq: int
    time_ms: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "stream_id": self.stream_id,
            "seq": self.seq,
            "time_ms": self.time_ms,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LaneIpcMessage":
        return cls(
            type=str(value["type"]),
            stream_id=int(value["stream_id"]),
            seq=int(value["seq"]),
            time_ms=int(value["time_ms"]),
            payload=dict(value.get("payload", {})),
        )


def make_message(
    *,
    message_type: LaneMessageType,
    stream_id: int,
    seq: int,
    payload: dict[str, Any],
    time_ms: int | None = None,
) -> LaneIpcMessage:
    return LaneIpcMessage(
        type=message_type,
        stream_id=stream_id,
        seq=seq,
        time_ms=now_ms() if time_ms is None else int(time_ms),
        payload=payload,
    )


def asr_event_to_payload(event: ASREvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "stream_id": event.stream_id,
        "sentence_id": event.sentence_id,
        "revision": event.revision,
        "status": event.status.value,
        "result_type": event.result_type.value,
        "is_final": event.is_final,
        "text": event.text,
        "timestamp_ms": event.timestamp_ms,
    }


def asr_event_from_payload(payload: dict[str, Any]) -> ASREvent:
    return ASREvent(
        event_type=str(payload["event_type"]),
        stream_id=int(payload["stream_id"]),
        sentence_id=int(payload["sentence_id"]),
        revision=int(payload["revision"]),
        status=EventStatus(str(payload["status"])),
        result_type=ResultType(str(payload["result_type"])),
        is_final=bool(payload["is_final"]),
        text=str(payload.get("text", "")),
        timestamp_ms=int(payload["timestamp_ms"]),
    )


def lane_metrics_to_payload(metrics: LaneMetrics) -> dict[str, Any]:
    return {
        "stream_id": metrics.stream_id,
        "processed_frames": metrics.processed_frames,
        "dropped_frames": metrics.dropped_frames,
        "offline_requests": metrics.offline_requests,
        "final_state": metrics.final_state.value,
    }


def lane_metrics_from_payload(payload: dict[str, Any]) -> LaneMetrics:
    return LaneMetrics(
        stream_id=int(payload["stream_id"]),
        processed_frames=int(payload.get("processed_frames", 0)),
        dropped_frames=int(payload.get("dropped_frames", 0)),
        offline_requests=int(payload.get("offline_requests", 0)),
        final_state=LaneFinalState(str(payload.get("final_state", LaneFinalState.STOPPED.value))),
    )
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
pytest tests/test_ipc.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/pipeline/ipc.py tests/test_ipc.py
git commit -m "feat: add lane IPC message contract"
```

---

### Task 2: Per-Lane Artifact Writers And Metrics v3 Summaries

**Files:**
- Create: `src/multi_stream_asr/log/lane_artifacts.py`
- Modify: `src/multi_stream_asr/log/metrics.py`
- Modify: `src/multi_stream_asr/log/recorder.py`
- Test: `tests/test_lane_artifacts.py`
- Test: `tests/test_buffers_metrics.py`

**Interfaces:**
- Consumes: `ASREvent`, `EventStatus`, `ModelOutputRecord`, `LaneMetrics`, tracker `.as_dict()` output.
- Produces: `LaneResultWriter`, `LaneModelOutputJsonlRecorder`, `merge_latency_summaries(...)`, `merge_rtf_summaries(...)`, `RunRecorder.save_process_run(...)`.

- [ ] **Step 1: Write failing lane artifact tests**

Create `tests/test_lane_artifacts.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from multi_stream_asr.log.lane_artifacts import LaneModelOutputJsonlRecorder, LaneResultWriter
from multi_stream_asr.log.model_output import ModelOutputRecord
from multi_stream_asr.models import ASREvent, EventStatus


def test_lane_result_writer_appends_only_final_events(tmp_path: Path):
    writer = LaneResultWriter(tmp_path / "results" / "lane0.txt")

    writer.record_event(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=1,
            status=EventStatus.PENDING,
            text="draft",
            timestamp_ms=100,
        )
    )
    writer.record_event(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=2,
            status=EventStatus.FINAL,
            text="final one",
            timestamp_ms=200,
        )
    )
    writer.record_event(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=1,
            revision=2,
            status=EventStatus.FINAL,
            text="",
            timestamp_ms=300,
        )
    )
    writer.close()

    assert (tmp_path / "results" / "lane0.txt").read_text(encoding="utf-8") == "final one\n\n"


def test_lane_model_output_jsonl_recorder_writes_compact_records(tmp_path: Path):
    recorder = LaneModelOutputJsonlRecorder(tmp_path / "model_outputs" / "lane1.jsonl")

    recorder.record(
        ModelOutputRecord(
            model="qwen3-asr",
            stream_id=1,
            sentence_id=2,
            status="final",
            text="recognized",
            audio_seconds=1.5,
            elapsed_ms=600.0,
            rtf=0.4,
            raw_response={"large": "not written"},
        )
    )
    recorder.close()

    lines = (tmp_path / "model_outputs" / "lane1.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {
        "audio_seconds": 1.5,
        "model": "qwen3-asr",
        "rtf": 0.4,
        "sentence_id": 2,
        "stream_id": 1,
        "text": "recognized",
    }
```

- [ ] **Step 2: Add failing metrics v3 recorder test**

Append to `tests/test_buffers_metrics.py`:

```python
def test_run_recorder_save_process_run_writes_schema_v3_without_rewriting_lane_files(tmp_path: Path):
    recorder = RunRecorder()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    run_dir = tmp_path / "run"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "model_outputs").mkdir(parents=True)
    (run_dir / "results" / "lane0.txt").write_text("lane-owned\n", encoding="utf-8")
    (run_dir / "model_outputs" / "lane0.jsonl").write_text('{"lane":0}\n', encoding="utf-8")

    saved = recorder.save_process_run(
        run_dir,
        config=config,
        lane_summaries=[
            {
                "lane_metrics": {
                    "stream_id": 0,
                    "processed_frames": 10,
                    "dropped_frames": 1,
                    "offline_requests": 2,
                    "final_state": "finished",
                },
                "online_frame_latency": {"count": 2, "total_ms": 100.0, "average_ms": 50.0, "max_ms": 60.0},
                "offline_rtf": {"count": 1, "total": 0.4, "average": 0.4, "max": 0.4},
                "event_loop_lag": {"count": 3, "total_ms": 6.0, "average_ms": 2.0, "max_ms": 4.0},
                "results_path": "results/lane0.txt",
                "model_outputs_path": "model_outputs/lane0.jsonl",
            },
            {
                "lane_metrics": {
                    "stream_id": 1,
                    "processed_frames": 20,
                    "dropped_frames": 0,
                    "offline_requests": 3,
                    "final_state": "finished",
                },
                "online_frame_latency": {"count": 2, "total_ms": 120.0, "average_ms": 60.0, "max_ms": 70.0},
                "offline_rtf": {"count": 1, "total": 0.5, "average": 0.5, "max": 0.5},
                "event_loop_lag": {"count": 2, "total_ms": 2.0, "average_ms": 1.0, "max_ms": 1.5},
                "results_path": "results/lane1.txt",
                "model_outputs_path": "model_outputs/lane1.jsonl",
            },
        ],
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved == run_dir
    assert metrics["metrics_schema_version"] == 3
    assert metrics["online_frame_latency"] == {
        "count": 4,
        "total_ms": 220.0,
        "average_ms": 55.0,
        "max_ms": 70.0,
    }
    assert metrics["offline_rtf"] == {"count": 2, "total": 0.9, "average": 0.45, "max": 0.5}
    assert metrics["event_loop_lag"]["max_ms"] == 4.0
    assert metrics["lanes"][0]["results_path"] == "results/lane0.txt"
    assert (run_dir / "results" / "lane0.txt").read_text(encoding="utf-8") == "lane-owned\n"
    assert (run_dir / "model_outputs" / "lane0.jsonl").read_text(encoding="utf-8") == '{"lane":0}\n'
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
pytest tests/test_lane_artifacts.py tests/test_buffers_metrics.py::test_run_recorder_save_process_run_writes_schema_v3_without_rewriting_lane_files -v
```

Expected: fail because `lane_artifacts.py` and `save_process_run` do not exist.

- [ ] **Step 4: Implement lane artifact writers**

Create `src/multi_stream_asr/log/lane_artifacts.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

from multi_stream_asr.log.model_output import ModelOutputRecord
from multi_stream_asr.models import ASREvent, EventStatus
from multi_stream_asr.utils.paths import ensure_dir


def compact_model_output_record(record: ModelOutputRecord) -> dict[str, object]:
    return {
        "model": record.model,
        "stream_id": record.stream_id,
        "sentence_id": record.sentence_id,
        "audio_seconds": record.audio_seconds,
        "rtf": record.rtf,
        "text": record.text,
    }


class LaneResultWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(path.parent)
        self._file: TextIO = path.open("w", encoding="utf-8")

    def record_event(self, event: ASREvent) -> None:
        if event.status is not EventStatus.FINAL:
            return
        self._file.write(f"{event.text}\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class LaneModelOutputJsonlRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(path.parent)
        self._file: TextIO = path.open("w", encoding="utf-8")

    def record(self, record: ModelOutputRecord) -> None:
        self._file.write(
            json.dumps(compact_model_output_record(record), ensure_ascii=False, sort_keys=True)
        )
        self._file.write("\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
```

- [ ] **Step 5: Implement summary merge helpers**

Append to `src/multi_stream_asr/log/metrics.py`:

```python
def merge_latency_summaries(summaries: list[dict[str, float | int]]) -> dict[str, float | int]:
    count = sum(int(summary.get("count", 0)) for summary in summaries)
    total_ms = float(sum(float(summary.get("total_ms", 0.0)) for summary in summaries))
    max_ms = max((float(summary.get("max_ms", 0.0)) for summary in summaries), default=0.0)
    return {
        "count": count,
        "total_ms": total_ms,
        "average_ms": total_ms / count if count else 0.0,
        "max_ms": max_ms,
    }


def merge_rtf_summaries(summaries: list[dict[str, float | int]]) -> dict[str, float | int]:
    count = sum(int(summary.get("count", 0)) for summary in summaries)
    total = float(sum(float(summary.get("total", 0.0)) for summary in summaries))
    max_value = max((float(summary.get("max", 0.0)) for summary in summaries), default=0.0)
    return {
        "count": count,
        "total": total,
        "average": total / count if count else 0.0,
        "max": max_value,
    }
```

- [ ] **Step 6: Implement process-run recorder**

Modify `src/multi_stream_asr/log/recorder.py` imports:

```python
from multi_stream_asr.log.metrics import (
    EventLoopLagTracker,
    LatencyTracker,
    RtfTracker,
    merge_latency_summaries,
    merge_rtf_summaries,
)
```

Add this method to `RunRecorder` below `save(...)`:

```python
    def save_process_run(
        self,
        run_dir: str | Path,
        *,
        config: AppConfig,
        lane_summaries: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> Path:
        output_dir = ensure_dir(run_dir)
        ensure_dir(output_dir / "results")
        ensure_dir(output_dir / "model_outputs")

        config_payload = _jsonable(config)
        config_payload["segment_cache_seconds"] = config.segment_cache_seconds
        (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True))

        lanes = [_jsonable(summary) for summary in lane_summaries]
        metrics_payload = {
            "metrics_schema_version": 3,
            "online_frame_latency": merge_latency_summaries(
                [dict(lane.get("online_frame_latency", {})) for lane in lanes]
            ),
            "offline_rtf": merge_rtf_summaries(
                [dict(lane.get("offline_rtf", {})) for lane in lanes]
            ),
            "event_loop_lag": merge_latency_summaries(
                [dict(lane.get("event_loop_lag", {})) for lane in lanes]
            ),
            "lanes": lanes,
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
        return output_dir
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
pytest tests/test_lane_artifacts.py tests/test_buffers_metrics.py::test_run_recorder_save_process_run_writes_schema_v3_without_rewriting_lane_files -v
```

Expected: all selected tests pass.

- [ ] **Step 8: Run existing recorder tests**

Run:

```bash
pytest tests/test_buffers_metrics.py -v
```

Expected: all tests pass; schema v2 behavior for `RunRecorder.save(...)` remains unchanged.

- [ ] **Step 9: Commit**

```bash
git add src/multi_stream_asr/log/lane_artifacts.py src/multi_stream_asr/log/metrics.py src/multi_stream_asr/log/recorder.py tests/test_lane_artifacts.py tests/test_buffers_metrics.py
git commit -m "feat: add lane-owned artifacts and metrics v3 summaries"
```

---

### Task 3: Single-Lane Qwen3 Server Lifecycle

**Files:**
- Modify: `src/multi_stream_asr/asr/offline/llama_server.py`
- Test: `tests/test_llama_server.py`

**Interfaces:**
- Consumes: Existing `LlamaServerManager`, `LaneConfig`, `LlamaServerSpec`.
- Produces: `LlamaServerManager.start_lane(lane: LaneConfig, cleanup: bool = False) -> LlamaServerSpec`.

- [ ] **Step 1: Add failing tests for single-lane start**

Append to `tests/test_llama_server.py`:

```python
def test_start_lane_starts_only_requested_lane_without_global_cleanup(monkeypatch, tmp_path: Path):
    from multi_stream_asr.asr.offline import llama_server as module

    config = AppConfig(lanes=4)
    manager = module.LlamaServerManager(config)
    cleanup_calls = 0
    writes: list[tuple[Path, int]] = []
    popen_commands: list[list[str]] = []

    class StartedProcess(FakeProcess):
        pass

    def fake_popen(command, env, stdout, stderr, text):
        popen_commands.append(command)
        return StartedProcess(pid=9400)

    def fake_write_config(config_dir: Path, lane: LaneConfig) -> None:
        writes.append((config_dir, lane.stream_id))
        config_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(manager, "_wait_for_health", lambda port: None)
    monkeypatch.setattr(manager, "_write_config", fake_write_config)

    def fake_cleanup():
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr(manager, "cleanup_before_start", fake_cleanup)

    spec = manager.start_lane(config.lane_configs[2], cleanup=False)

    assert cleanup_calls == 0
    assert writes == [(spec.config_dir, 2)]
    assert spec.stream_id == 2
    assert spec.port == config.lane_configs[2].offline_port
    assert len(popen_commands) == 1
    assert popen_commands[0][popen_commands[0].index("--port") + 1] == str(spec.port)


def test_start_all_still_runs_global_cleanup_once(monkeypatch):
    from multi_stream_asr.asr.offline import llama_server as module

    config = AppConfig(lanes=2)
    manager = module.LlamaServerManager(config)
    cleanup_calls = 0

    def fake_cleanup():
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr(manager, "cleanup_before_start", fake_cleanup)
    monkeypatch.setattr(manager, "_write_config", lambda config_dir, lane: config_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(manager, "_wait_for_health", lambda port: None)
    monkeypatch.setattr(module.subprocess, "Popen", lambda command, env, stdout, stderr, text: FakeProcess(pid=9500))

    specs = manager.start_all()

    assert cleanup_calls == 1
    assert [spec.stream_id for spec in specs] == [0, 1]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_llama_server.py::test_start_lane_starts_only_requested_lane_without_global_cleanup tests/test_llama_server.py::test_start_all_still_runs_global_cleanup_once -v
```

Expected: fail because `start_lane` does not exist.

- [ ] **Step 3: Implement `start_lane` and refactor `start_all`**

Modify `LlamaServerManager.start_all` and add `start_lane`:

```python
    def start_all(self) -> list[LlamaServerSpec]:
        self.cleanup_before_start()
        started_specs: list[LlamaServerSpec] = []
        try:
            for lane in self._config.lane_configs:
                started_specs.append(self.start_lane(lane, cleanup=False))
            return started_specs
        except Exception:
            self.stop_all()
            raise

    def start_lane(self, lane: LaneConfig, *, cleanup: bool = False) -> LlamaServerSpec:
        if cleanup:
            self.cleanup_before_start()
        spec = self._build_spec(lane)
        self._write_config(spec.config_dir, lane)
        log_handle = spec.log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                spec.command,
                env=spec.env,
                stdout=log_handle,
                stderr=log_handle,
                text=True,
            )
            spec.pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
            self._started.append((spec, process, log_handle))
            self._wait_for_health(spec.port)
            return spec
        except Exception:
            log_handle.close()
            self.stop_all()
            raise
```

- [ ] **Step 4: Run llama server tests**

Run:

```bash
pytest tests/test_llama_server.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/asr/offline/llama_server.py tests/test_llama_server.py
git commit -m "feat: support single-lane qwen server lifecycle"
```

---

### Task 4: Lane Child Runtime

**Files:**
- Create: `src/multi_stream_asr/pipeline/lane_runtime.py`
- Test: `tests/test_lane_runtime.py`

**Interfaces:**
- Consumes: `LanePipeline`, `LocalWavSource`, `ZipformerRecognizer`, `Qwen3AsrClient`, `LlamaServerManager.start_lane(...)`, `LaneResultWriter`, `LaneModelOutputJsonlRecorder`, IPC helpers.
- Produces: `LaneProcessConfig`, `run_lane_process(...)`, `LaneRuntime`, and `LaneEventSink`.

- [ ] **Step 1: Write fake lane runtime tests**

Create `tests/test_lane_runtime.py`:

```python
from __future__ import annotations

import asyncio
import queue
from pathlib import Path

from multi_stream_asr.config import AppConfig
from multi_stream_asr.models import EventStatus
from multi_stream_asr.pipeline.ipc import LaneIpcMessage
from multi_stream_asr.pipeline.lane_runtime import LaneProcessConfig, LaneRuntime


class MemoryQueue:
    def __init__(self) -> None:
        self.items: queue.Queue[dict] = queue.Queue()

    def put(self, item: dict) -> None:
        self.items.put(item)

    def get(self, timeout: float | None = None) -> dict:
        return self.items.get(timeout=timeout)


async def collect_until(queue_obj: MemoryQueue, message_type: str):
    while True:
        message = LaneIpcMessage.from_dict(await asyncio.to_thread(queue_obj.get, 1.0))
        if message.type == message_type:
            return message


async def collect_lifecycle(queue_obj: MemoryQueue, state: str):
    while True:
        message = LaneIpcMessage.from_dict(await asyncio.to_thread(queue_obj.get, 1.0))
        if message.type == "lifecycle" and message.payload["state"] == state:
            return message


def test_lane_runtime_fake_load_start_writes_lane_artifacts(tmp_path: Path):
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    lane = config.lane_configs[0]
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=lane,
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    async def run_runtime():
        run_task = asyncio.create_task(runtime.run())
        loaded = await collect_lifecycle(events, "lane_loaded")
        assert loaded.payload["state"] == "lane_loaded"
        commands.put({"type": "start", "run_dir": str(tmp_path / "run")})
        final_metrics = await collect_until(events, "metrics_final")
        paths = await collect_until(events, "paths")
        await asyncio.wait_for(run_task, timeout=1.0)
        return final_metrics, paths

    final_metrics, paths = asyncio.run(run_runtime())

    assert final_metrics.payload["lane_metrics"]["final_state"] == "finished"
    assert paths.payload == {
        "results_path": "results/lane0.txt",
        "model_outputs_path": "model_outputs/lane0.jsonl",
    }
    assert (tmp_path / "run" / "results" / "lane0.txt").read_text(encoding="utf-8") == "fake final lane 0\n"
    model_output_text = (tmp_path / "run" / "model_outputs" / "lane0.jsonl").read_text(encoding="utf-8")
    assert "qwen3-asr" in model_output_text


def test_lane_runtime_event_sink_sends_asr_events_and_writes_finals(tmp_path: Path):
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=config.lane_configs[1],
            fake=True,
        ),
        event_queue=events,
        command_queue=commands,
    )

    async def run_runtime():
        run_task = asyncio.create_task(runtime.run())
        await collect_lifecycle(events, "lane_loaded")
        commands.put({"type": "start", "run_dir": str(tmp_path / "run")})
        statuses = []
        while True:
            message = LaneIpcMessage.from_dict(await asyncio.to_thread(events.get, 1.0))
            if message.type == "asr_event":
                statuses.append(message.payload["status"])
            if message.type == "metrics_final":
                break
        await asyncio.wait_for(run_task, timeout=1.0)
        return statuses

    statuses = asyncio.run(run_runtime())

    assert EventStatus.PARTIAL.value in statuses
    assert EventStatus.PENDING.value in statuses
    assert EventStatus.FINAL.value in statuses
    assert EventStatus.LANE_FINISHED.value in statuses
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_lane_runtime.py -v
```

Expected: fail because `lane_runtime.py` does not exist.

- [ ] **Step 3: Implement lane runtime config, fake models, event sink, and metrics payload**

Create `src/multi_stream_asr/pipeline/lane_runtime.py` with these public pieces:

```python
from __future__ import annotations

import asyncio
import traceback
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from multi_stream_asr.asr.offline.base import OfflineRecognitionResult
from multi_stream_asr.asr.offline.llama_server import LlamaServerManager
from multi_stream_asr.asr.offline.qwen3asr import Qwen3AsrClient
from multi_stream_asr.asr.online.zipformer import ZipformerRecognizer
from multi_stream_asr.audio.local_file import LocalWavSource
from multi_stream_asr.buffer.segment_cache import SegmentCache
from multi_stream_asr.config import AppConfig, LaneConfig
from multi_stream_asr.log.lane_artifacts import LaneModelOutputJsonlRecorder, LaneResultWriter
from multi_stream_asr.log.metrics import EventLoopLagTracker, LatencyTracker, RtfTracker
from multi_stream_asr.models import ASREvent, AudioFrame, LaneFinalState, LaneMetrics, OnlineRecognition
from multi_stream_asr.pipeline.ipc import asr_event_to_payload, lane_metrics_to_payload, make_message
from multi_stream_asr.pipeline.pipeline import LanePipeline


@dataclass(frozen=True)
class LaneProcessConfig:
    app_config: AppConfig
    lane: LaneConfig
    fake: bool = False


class _FakeAudioSource:
    def __init__(self, stream_id: int, frame_duration_ms: int) -> None:
        self.stream_id = stream_id
        self.frame_duration_ms = frame_duration_ms
        self._stopped = False

    async def prepare(self) -> None:
        self._stopped = False

    async def frames(self) -> AsyncIterator[AudioFrame]:
        frames = [
            AudioFrame(self.stream_id, 0, np.asarray([0.1, 0.2], dtype=np.float32), self.frame_duration_ms),
            AudioFrame(self.stream_id, 1, np.asarray([0.3, 0.4], dtype=np.float32), self.frame_duration_ms * 2),
        ]
        for frame in frames:
            if self._stopped:
                break
            yield frame
            await asyncio.sleep(0.001)
        yield AudioFrame.finish(self.stream_id, len(frames), self.frame_duration_ms * (len(frames) + 1))

    async def stop(self) -> None:
        self._stopped = True


class _FakeOnlineRecognizer:
    def __init__(self, lane: LaneConfig) -> None:
        self._outputs = [
            OnlineRecognition(lane.stream_id, 0, 0, f"fake partial lane {lane.stream_id}", 100),
            OnlineRecognition(lane.stream_id, 0, 0, f"fake partial lane {lane.stream_id}", 200, True),
        ]

    async def accept(self, frame: AudioFrame) -> OnlineRecognition | None:
        if frame.is_finish:
            return None
        return self._outputs.pop(0) if self._outputs else None

    async def reset(self) -> None:
        return None


class _FakeOfflineRecognizer:
    def __init__(self, lane: LaneConfig) -> None:
        self._lane = lane

    async def recognize(self, pcm: np.ndarray, prompt: str) -> OfflineRecognitionResult:
        audio_seconds = max(float(np.asarray(pcm).size) / 16000.0, 1.0)
        return OfflineRecognitionResult(
            text=f"fake final lane {self._lane.stream_id}",
            elapsed_ms=400.0,
            audio_seconds=audio_seconds,
            rtf=0.4,
        )

    async def close(self) -> None:
        return None


class LaneEventSink:
    def __init__(self, *, runtime: "LaneRuntime", result_writer: LaneResultWriter) -> None:
        self._runtime = runtime
        self._result_writer = result_writer

    def publish(self, event: ASREvent) -> None:
        self._result_writer.record_event(event)
        self._runtime.send("asr_event", asr_event_to_payload(event))


class LaneRuntime:
    def __init__(self, *, process_config: LaneProcessConfig, event_queue: Any, command_queue: Any) -> None:
        self.process_config = process_config
        self.event_queue = event_queue
        self.command_queue = command_queue
        self._seq = 0
        self._pipeline: LanePipeline | None = None
        self._source: Any | None = None
        self._server_manager: LlamaServerManager | None = None
        self._latency_tracker = LatencyTracker()
        self._rtf_tracker = RtfTracker()
        self._event_loop_lag_tracker = EventLoopLagTracker()

    @property
    def lane(self) -> LaneConfig:
        return self.process_config.lane

    @property
    def app_config(self) -> AppConfig:
        return self.process_config.app_config

    def send(self, message_type: str, payload: dict[str, Any]) -> None:
        self._seq += 1
        self.event_queue.put(
            make_message(
                message_type=message_type,
                stream_id=self.lane.stream_id,
                seq=self._seq,
                payload=payload,
            ).to_dict()
        )
```

Continue the file with runtime methods:

```python
    async def run(self) -> int:
        try:
            await self._load()
            command = await self._wait_for_command({"start", "stop", "shutdown"})
            if command["type"] in {"stop", "shutdown"}:
                await self._shutdown()
                return 0
            await self._run_started(command)
            await self._shutdown()
            return 0
        except BaseException as exc:
            self._send_fatal(exc, phase="run")
            await self._shutdown()
            return 1

    async def _load(self) -> None:
        self.send("lifecycle", {"state": "lane_loading", "message": ""})
        if self.process_config.fake:
            self._server_manager = None
        else:
            self._server_manager = LlamaServerManager(self.app_config)
            await asyncio.to_thread(self._server_manager.start_lane, self.lane, cleanup=False)
        self.send("lifecycle", {"state": "lane_loaded", "message": ""})

    async def _run_started(self, command: dict[str, Any]) -> None:
        run_dir = Path(str(command["run_dir"]))
        result_writer = LaneResultWriter(run_dir / "results" / f"lane{self.lane.stream_id}.txt")
        model_output_recorder = LaneModelOutputJsonlRecorder(
            run_dir / "model_outputs" / f"lane{self.lane.stream_id}.jsonl"
        )
        try:
            event_sink = LaneEventSink(runtime=self, result_writer=result_writer)
            source = self._build_source(command)
            self._source = source
            pipeline = LanePipeline(
                stream_id=self.lane.stream_id,
                online_recognizer=self._build_online_recognizer(),
                offline_recognizer=self._build_offline_recognizer(),
                event_bus=event_sink,
                segment_cache=SegmentCache(
                    sample_rate=16000,
                    capacity_seconds=self.app_config.segment_cache_seconds,
                ),
                latency_tracker=self._latency_tracker,
                rtf_tracker=self._rtf_tracker,
                model_output_recorder=model_output_recorder,
            )
            self._pipeline = pipeline
            self.send("lifecycle", {"state": "lane_started", "message": ""})
            lag_task = asyncio.create_task(self._track_event_loop_lag())
            command_task = asyncio.create_task(self._watch_runtime_commands())
            try:
                await pipeline.run(source)
            finally:
                command_task.cancel()
                lag_task.cancel()
                try:
                    await command_task
                except asyncio.CancelledError:
                    pass
                try:
                    await lag_task
                except asyncio.CancelledError:
                    pass
            self._send_final_metrics(pipeline.metrics, run_dir)
            self.send(
                "paths",
                {
                    "results_path": f"results/lane{self.lane.stream_id}.txt",
                    "model_outputs_path": f"model_outputs/lane{self.lane.stream_id}.jsonl",
                },
            )
            self.send("lifecycle", {"state": "lane_finished", "message": ""})
        finally:
            result_writer.close()
            model_output_recorder.close()

    def _build_source(self, command: dict[str, Any]):
        if self.process_config.fake:
            return _FakeAudioSource(self.lane.stream_id, self.app_config.frame_duration_ms)
        input_path = Path(str(command["input_path"]))
        return LocalWavSource(
            stream_id=self.lane.stream_id,
            path=input_path,
            frame_duration_ms=self.app_config.frame_duration_ms,
            realtime=True,
        )

    def _build_online_recognizer(self):
        if self.process_config.fake:
            return _FakeOnlineRecognizer(self.lane)
        return ZipformerRecognizer(self.app_config.models.online_model_dir, self.app_config.endpoint)

    def _build_offline_recognizer(self):
        if self.process_config.fake:
            return _FakeOfflineRecognizer(self.lane)
        return Qwen3AsrClient(lane=self.lane, offline=self.app_config.offline)
```

Finish the file with command/metrics/fatal helpers and process entrypoint:

```python
    async def _wait_for_command(self, allowed: set[str]) -> dict[str, Any]:
        while True:
            command = await asyncio.to_thread(self.command_queue.get)
            command_type = str(command.get("type", ""))
            if command_type in allowed:
                return dict(command)

    async def _watch_runtime_commands(self) -> None:
        while True:
            command = await asyncio.to_thread(self.command_queue.get)
            command_type = str(command.get("type", ""))
            if command_type in {"stop", "shutdown"}:
                if self._pipeline is not None:
                    self._pipeline.request_stop()
                if self._source is not None:
                    await self._source.stop()
                return

    async def _track_event_loop_lag(self, interval_seconds: float = 0.1) -> None:
        import time

        expected = time.perf_counter() + interval_seconds
        while True:
            await asyncio.sleep(interval_seconds)
            now = time.perf_counter()
            self._event_loop_lag_tracker.add(max(0.0, (now - expected) * 1000.0))
            expected = now + interval_seconds

    def _send_final_metrics(self, lane_metrics: LaneMetrics, run_dir: Path) -> None:
        payload = {
            "lane_metrics": lane_metrics_to_payload(lane_metrics),
            "online_frame_latency": self._latency_tracker.as_dict(),
            "offline_rtf": self._rtf_tracker.as_dict(),
            "event_loop_lag": self._event_loop_lag_tracker.as_dict(),
            "results_path": f"results/lane{self.lane.stream_id}.txt",
            "model_outputs_path": f"model_outputs/lane{self.lane.stream_id}.jsonl",
        }
        self.send("metrics_final", payload)

    def _send_fatal(self, exc: BaseException, *, phase: str) -> None:
        self.send(
            "fatal",
            {
                "phase": phase,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
                "exit_code": 1,
            },
        )
        metrics = LaneMetrics(stream_id=self.lane.stream_id, final_state=LaneFinalState.FATAL)
        self.send(
            "metrics_final",
            {
                "lane_metrics": lane_metrics_to_payload(metrics),
                "online_frame_latency": self._latency_tracker.as_dict(),
                "offline_rtf": self._rtf_tracker.as_dict(),
                "event_loop_lag": self._event_loop_lag_tracker.as_dict(),
                "results_path": f"results/lane{self.lane.stream_id}.txt",
                "model_outputs_path": f"model_outputs/lane{self.lane.stream_id}.jsonl",
            },
        )

    async def _shutdown(self) -> None:
        if self._pipeline is not None:
            self._pipeline.request_stop()
        if self._source is not None:
            await self._source.stop()
        if self._server_manager is not None:
            await asyncio.to_thread(self._server_manager.stop_all)
        self.send("lifecycle", {"state": "lane_stopped", "message": ""})


def run_lane_process(process_config: LaneProcessConfig, event_queue: Any, command_queue: Any) -> None:
    runtime = LaneRuntime(
        process_config=process_config,
        event_queue=event_queue,
        command_queue=command_queue,
    )
    raise SystemExit(asyncio.run(runtime.run()))
```

Ensure imports include `lane_metrics_to_payload`:

```python
from multi_stream_asr.pipeline.ipc import asr_event_to_payload, lane_metrics_to_payload, make_message
```

- [ ] **Step 4: Run lane runtime tests**

Run:

```bash
pytest tests/test_lane_runtime.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run pipeline tests to catch event sink compatibility**

Run:

```bash
pytest tests/test_pipeline.py -v
```

Expected: all existing in-process pipeline tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/multi_stream_asr/pipeline/lane_runtime.py tests/test_lane_runtime.py
git commit -m "feat: add lane child runtime"
```

---

### Task 5: Main-Process Lane Process Manager

**Files:**
- Create: `src/multi_stream_asr/pipeline/process_manager.py`
- Modify: `src/multi_stream_asr/pipeline/__init__.py`
- Test: `tests/test_process_manager.py`

**Interfaces:**
- Consumes: `LaneProcessConfig`, `run_lane_process`, IPC helpers, `FrontendEventBus`, `RunRecorder.save_process_run`.
- Produces: `LaneProcessController`, `ProcessPipelineManager` with `load_models()`, `start_recognition()`, `wait_for_completion()`, `stop_recognition()`, `unload_models()`, `lane_metrics`, `lane_summaries`, `run_dir`, `state`.

- [ ] **Step 1: Write fake process manager tests**

Create `tests/test_process_manager.py`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from multi_stream_asr.config import AppConfig
from multi_stream_asr.models import EventStatus, LaneFinalState
from multi_stream_asr.pipeline.process_manager import ProcessPipelineManager
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.web.event_bus import FrontendEventBus


async def collect_events(bus: FrontendEventBus, count: int):
    return [await asyncio.wait_for(bus.next_event(), timeout=2.0) for _ in range(count)]


@pytest.mark.asyncio
async def test_process_pipeline_manager_fake_run_reaches_stopped_and_collects_summaries(tmp_path: Path):
    config = AppConfig(lanes=2, output_dir=tmp_path)
    bus = FrontendEventBus()
    manager = ProcessPipelineManager(config=config, event_bus=bus, fake=True)

    await manager.load_models()
    assert manager.state is PipelineState.LOADED

    await manager.start_recognition()
    await manager.wait_for_completion()

    assert manager.state is PipelineState.STOPPED
    assert manager.run_dir is not None
    assert [metrics.final_state for metrics in manager.lane_metrics] == [
        LaneFinalState.FINISHED,
        LaneFinalState.FINISHED,
    ]
    assert len(manager.lane_summaries) == 2
    assert (manager.run_dir / "results" / "lane0.txt").exists()
    assert (manager.run_dir / "model_outputs" / "lane1.jsonl").exists()

    events = await collect_events(bus, 8)
    statuses = [event.status for event in events]
    assert EventStatus.PARTIAL in statuses
    assert EventStatus.PENDING in statuses
    assert EventStatus.FINAL in statuses
    assert statuses.count(EventStatus.LANE_FINISHED) == 2

    await manager.unload_models()
    assert manager.state is PipelineState.UNLOADED


@pytest.mark.asyncio
async def test_process_pipeline_manager_stop_before_completion_marks_stopped(tmp_path: Path):
    config = AppConfig(lanes=2, output_dir=tmp_path)
    manager = ProcessPipelineManager(config=config, event_bus=FrontendEventBus(), fake=True)

    await manager.load_models()
    await manager.start_recognition()
    await manager.stop_recognition()

    assert manager.state in {PipelineState.STOPPED, PipelineState.UNLOADED}
    await manager.unload_models()
    assert manager.state is PipelineState.UNLOADED
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_process_manager.py -v
```

Expected: fail because `process_manager.py` does not exist.

- [ ] **Step 3: Implement process controller and manager**

Create `src/multi_stream_asr/pipeline/process_manager.py`:

```python
from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from multi_stream_asr.config import AppConfig, LaneConfig
from multi_stream_asr.models import ASREvent, LaneFinalState, LaneMetrics
from multi_stream_asr.pipeline.ipc import (
    LaneIpcMessage,
    asr_event_from_payload,
    lane_metrics_from_payload,
)
from multi_stream_asr.pipeline.lane_runtime import LaneProcessConfig, run_lane_process
from multi_stream_asr.pipeline.state import PipelineState, assert_transition
from multi_stream_asr.web.event_bus import FrontendEventBus


class LaneProcessController:
    def __init__(self, *, app_config: AppConfig, lane: LaneConfig, fake: bool) -> None:
        self.app_config = app_config
        self.lane = lane
        self.fake = fake
        self._ctx = mp.get_context("spawn")
        self.event_queue = self._ctx.Queue()
        self.command_queue = self._ctx.Queue()
        self.process: mp.Process | None = None

    def start(self) -> None:
        process_config = LaneProcessConfig(
            app_config=self.app_config,
            lane=self.lane,
            fake=self.fake,
        )
        self.process = self._ctx.Process(
            target=run_lane_process,
            args=(process_config, self.event_queue, self.command_queue),
            name=f"asr-lane-{self.lane.stream_id}",
        )
        self.process.start()

    def send(self, command: dict[str, Any]) -> None:
        self.command_queue.put(command)

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    async def join(self, timeout: float = 5.0) -> int | None:
        if self.process is None:
            return None
        await asyncio.to_thread(self.process.join, timeout)
        return self.process.exitcode

    async def terminate_then_kill(self, *, terminate_timeout: float = 5.0, kill_timeout: float = 2.0) -> None:
        if self.process is None or not self.process.is_alive():
            return
        self.process.terminate()
        await asyncio.to_thread(self.process.join, terminate_timeout)
        if self.process.is_alive():
            self.process.kill()
            await asyncio.to_thread(self.process.join, kill_timeout)
```

Continue with manager initialization and properties:

```python
class ProcessPipelineManager:
    def __init__(
        self,
        *,
        config: AppConfig,
        event_bus: FrontendEventBus,
        fake: bool = False,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._fake = fake
        self._state = PipelineState.UNLOADED
        self._lifecycle_lock = asyncio.Lock()
        self._controllers: dict[int, LaneProcessController] = {}
        self._reader_tasks: dict[int, asyncio.Task[None]] = {}
        self._loaded: set[int] = set()
        self._finished: set[int] = set()
        self._last_seq: dict[int, int] = {}
        self._lane_metrics: dict[int, LaneMetrics] = {
            lane.stream_id: LaneMetrics(stream_id=lane.stream_id)
            for lane in config.lane_configs
        }
        self._lane_summaries: dict[int, dict[str, Any]] = {}
        self._paths: dict[int, dict[str, str]] = {}
        self._failure: BaseException | None = None
        self._run_dir: Path | None = None
        self._completion_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    @property
    def lane_metrics(self) -> list[LaneMetrics]:
        return [self._lane_metrics[lane.stream_id] for lane in self._config.lane_configs]

    @property
    def lane_summaries(self) -> list[dict[str, Any]]:
        return [
            self._lane_summaries.get(
                lane.stream_id,
                {
                    "lane_metrics": {
                        "stream_id": lane.stream_id,
                        "processed_frames": 0,
                        "dropped_frames": 0,
                        "offline_requests": 0,
                        "final_state": "stopped",
                    },
                    "online_frame_latency": {"count": 0, "total_ms": 0.0, "average_ms": 0.0, "max_ms": 0.0},
                    "offline_rtf": {"count": 0, "total": 0.0, "average": 0.0, "max": 0.0},
                    "event_loop_lag": {"count": 0, "total_ms": 0.0, "average_ms": 0.0, "max_ms": 0.0},
                    "results_path": f"results/lane{lane.stream_id}.txt",
                    "model_outputs_path": f"model_outputs/lane{lane.stream_id}.jsonl",
                },
            )
            for lane in self._config.lane_configs
        ]
```

Continue with lifecycle methods:

```python
    async def load_models(self) -> None:
        async with self._lifecycle_lock:
            assert_transition(self._state, PipelineState.LOADING)
            self._state = PipelineState.LOADING
            self._controllers = {
                lane.stream_id: LaneProcessController(
                    app_config=self._config,
                    lane=lane,
                    fake=self._fake,
                )
                for lane in self._config.lane_configs
            }
            for controller in self._controllers.values():
                controller.start()
                self._reader_tasks[controller.lane.stream_id] = asyncio.create_task(
                    self._read_lane_events(controller)
                )
            await self._wait_until_loaded()
            assert_transition(self._state, PipelineState.LOADED)
            self._state = PipelineState.LOADED

    async def start_recognition(self) -> None:
        async with self._lifecycle_lock:
            assert_transition(self._state, PipelineState.STARTING)
            self._state = PipelineState.STARTING
            self._event_bus.clear_results()
            self._finished.clear()
            self._lane_summaries.clear()
            self._run_dir = self._config.output_dir / f"process-{int(time.time() * 1000)}"
            self._run_dir.mkdir(parents=True, exist_ok=True)
            input_paths = self._resolve_input_paths()
            for lane in self._config.lane_configs:
                command: dict[str, Any] = {"type": "start", "run_dir": str(self._run_dir)}
                if lane.stream_id in input_paths:
                    command["input_path"] = str(input_paths[lane.stream_id])
                self._controllers[lane.stream_id].send(command)
            self._completion_task = asyncio.create_task(self._wait_for_finished())
            assert_transition(self._state, PipelineState.RUNNING)
            self._state = PipelineState.RUNNING

    async def wait_for_completion(self) -> None:
        if self._completion_task is not None:
            await self._completion_task
        if self._failure is not None:
            raise self._failure

    async def stop_recognition(self) -> None:
        async with self._lifecycle_lock:
            if self._state is PipelineState.STOPPED:
                return
            if self._state in {PipelineState.STARTING, PipelineState.RUNNING}:
                assert_transition(self._state, PipelineState.STOPPING)
                self._state = PipelineState.STOPPING
            for controller in self._controllers.values():
                controller.send({"type": "stop", "reason": "user_requested"})
            await self._join_all()
            if self._state is not PipelineState.FATAL:
                assert_transition(self._state, PipelineState.STOPPED)
                self._state = PipelineState.STOPPED

    async def unload_models(self) -> None:
        async with self._lifecycle_lock:
            if self._state is PipelineState.UNLOADED:
                return
            if self._state is not PipelineState.FATAL:
                assert_transition(self._state, PipelineState.UNLOADING)
                self._state = PipelineState.UNLOADING
            for controller in self._controllers.values():
                controller.send({"type": "shutdown"})
            await self._join_all()
            for task in self._reader_tasks.values():
                task.cancel()
            await asyncio.gather(*self._reader_tasks.values(), return_exceptions=True)
            self._controllers.clear()
            self._reader_tasks.clear()
            self._loaded.clear()
            self._finished.clear()
            self._state = PipelineState.UNLOADED
```

Finish manager with event handling:

```python
    def _resolve_input_paths(self) -> dict[int, Path]:
        if self._fake:
            return {}
        from multi_stream_asr.audio.local_file import select_input_files

        paths = select_input_files(self._config.input_dir, self._config.lanes)
        return {
            lane.stream_id: paths[lane.stream_id]
            for lane in self._config.lane_configs
        }

    async def _wait_until_loaded(self) -> None:
        while len(self._loaded) < self._config.lanes:
            if self._failure is not None:
                raise self._failure
            await asyncio.sleep(0.01)

    async def _wait_for_finished(self) -> None:
        while len(self._finished) < self._config.lanes:
            if self._failure is not None:
                raise self._failure
            await asyncio.sleep(0.01)
        if self._state is PipelineState.RUNNING:
            assert_transition(self._state, PipelineState.STOPPING)
            self._state = PipelineState.STOPPING
            assert_transition(self._state, PipelineState.STOPPED)
            self._state = PipelineState.STOPPED

    async def _read_lane_events(self, controller: LaneProcessController) -> None:
        while True:
            raw = await asyncio.to_thread(controller.event_queue.get)
            message = LaneIpcMessage.from_dict(raw)
            await self._handle_message(message)

    async def _handle_message(self, message: LaneIpcMessage) -> None:
        last_seq = self._last_seq.get(message.stream_id, -1)
        if message.seq <= last_seq:
            return
        self._last_seq[message.stream_id] = message.seq
        if message.type == "asr_event":
            self._event_bus.publish(asr_event_from_payload(message.payload))
            return
        if message.type == "lifecycle":
            state = str(message.payload.get("state", ""))
            if state == "lane_loaded":
                self._loaded.add(message.stream_id)
            elif state in {"lane_finished", "lane_stopped"}:
                self._finished.add(message.stream_id)
            return
        if message.type in {"metrics_snapshot", "metrics_final"}:
            payload = dict(message.payload)
            metrics_payload = dict(payload.get("lane_metrics", {}))
            if metrics_payload:
                self._lane_metrics[message.stream_id] = lane_metrics_from_payload(metrics_payload)
            if message.type == "metrics_final":
                self._lane_summaries[message.stream_id] = payload
            return
        if message.type == "paths":
            self._paths[message.stream_id] = {
                "results_path": str(message.payload.get("results_path", "")),
                "model_outputs_path": str(message.payload.get("model_outputs_path", "")),
            }
            if message.stream_id in self._lane_summaries:
                self._lane_summaries[message.stream_id].update(self._paths[message.stream_id])
            return
        if message.type == "fatal":
            self._failure = RuntimeError(str(message.payload.get("message", "lane failed")))
            self._lane_metrics[message.stream_id] = replace(
                self._lane_metrics.get(message.stream_id, LaneMetrics(stream_id=message.stream_id)),
                final_state=LaneFinalState.FATAL,
            )
            self._state = PipelineState.FATAL
            for stream_id, controller in self._controllers.items():
                if stream_id != message.stream_id:
                    controller.send({"type": "stop", "reason": "sibling_fatal"})

    async def _join_all(self) -> None:
        for controller in self._controllers.values():
            await controller.join(timeout=5.0)
        for controller in self._controllers.values():
            await controller.terminate_then_kill()
```

- [ ] **Step 4: Export the process manager**

Modify `src/multi_stream_asr/pipeline/__init__.py`:

```python
from .process_manager import ProcessPipelineManager
from .state import PipelineState, assert_transition

__all__ = ["PipelineState", "ProcessPipelineManager", "assert_transition"]
```

- [ ] **Step 5: Run process manager tests**

Run:

```bash
pytest tests/test_process_manager.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run state and web event bus tests**

Run:

```bash
pytest tests/test_models_state.py tests/test_event_bus.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/multi_stream_asr/pipeline/process_manager.py src/multi_stream_asr/pipeline/__init__.py tests/test_process_manager.py
git commit -m "feat: add process-based pipeline manager"
```

---

### Task 6: Wire Headless And Web To Process Mode

**Files:**
- Modify: `src/multi_stream_asr/headless/runner.py`
- Modify: `src/multi_stream_asr/web/server.py`
- Test: `tests/test_manager_headless.py`
- Test: `tests/test_web_server.py`

**Interfaces:**
- Consumes: `ProcessPipelineManager.run_dir`, `ProcessPipelineManager.lane_summaries`, `RunRecorder.save_process_run(...)`.
- Produces: Headless and web default managers use process mode; legacy in-process `PipelineManager` tests remain available by direct import.

- [ ] **Step 1: Add failing headless process-mode test**

Append to `tests/test_manager_headless.py`:

```python
@pytest.mark.asyncio
async def test_headless_runner_fake_uses_process_manager_and_saves_schema_v3(tmp_path: Path):
    from multi_stream_asr.headless.runner import HeadlessRunner
    from multi_stream_asr.pipeline.process_manager import ProcessPipelineManager

    runner = HeadlessRunner(config=AppConfig(lanes=2, output_dir=tmp_path), fake=True)

    result = await runner.run()

    assert result == 0
    assert isinstance(runner.manager, ProcessPipelineManager)
    assert runner.last_run_dir is not None
    metrics = json.loads((runner.last_run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metrics_schema_version"] == 3
    assert (runner.last_run_dir / "results" / "lane0.txt").exists()
    assert (runner.last_run_dir / "model_outputs" / "lane1.jsonl").exists()
```

- [ ] **Step 2: Add failing web recorder compatibility test**

Modify `RecordingRecorder.save(...)` in `tests/test_web_server.py` to accept `event_loop_lag_tracker=None, model_outputs=None` if it does not already, then add a `save_process_run` method:

```python
    def save_process_run(self, run_dir: str | Path, *, config: AppConfig, lane_summaries):
        self.saved_runs.append(
            {
                "run_dir": Path(run_dir),
                "config": config,
                "lane_summaries": list(lane_summaries),
                "process": True,
            }
        )
        return Path(run_dir)
```

Append this test:

```python
def test_record_results_uses_process_run_summary_when_manager_exposes_run_dir(tmp_path):
    class ProcessStyleManager(RecordingManager):
        def __init__(self):
            super().__init__(state=PipelineState.STOPPED)
            self.run_dir = tmp_path / "process-run"
            self.lane_summaries = [
                {
                    "lane_metrics": {
                        "stream_id": 0,
                        "processed_frames": 1,
                        "dropped_frames": 0,
                        "offline_requests": 1,
                        "final_state": "finished",
                    },
                    "online_frame_latency": {"count": 1, "total_ms": 50.0, "average_ms": 50.0, "max_ms": 50.0},
                    "offline_rtf": {"count": 1, "total": 0.4, "average": 0.4, "max": 0.4},
                    "event_loop_lag": {"count": 1, "total_ms": 1.0, "average_ms": 1.0, "max_ms": 1.0},
                    "results_path": "results/lane0.txt",
                    "model_outputs_path": "model_outputs/lane0.jsonl",
                }
            ]

    recorder = RecordingRecorder()
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=ProcessStyleManager(),
        event_bus=FrontendEventBus(),
        recorder=recorder,
    )

    with TestClient(app):
        pass

    assert recorder.saved_runs[-1]["process"] is True
    assert recorder.saved_runs[-1]["run_dir"] == tmp_path / "process-run"
```

- [ ] **Step 3: Run selected tests and verify they fail**

Run:

```bash
pytest tests/test_manager_headless.py::test_headless_runner_fake_uses_process_manager_and_saves_schema_v3 tests/test_web_server.py::test_record_results_uses_process_run_summary_when_manager_exposes_run_dir -v
```

Expected: fail because `HeadlessRunner` still builds the in-process manager and web `_record_results` does not call `save_process_run`.

- [ ] **Step 4: Wire `HeadlessRunner` to `ProcessPipelineManager`**

Modify imports in `src/multi_stream_asr/headless/runner.py`:

```python
from multi_stream_asr.pipeline.process_manager import ProcessPipelineManager
```

Replace `_build_manager(...)` with:

```python
    def _build_manager(self) -> ProcessPipelineManager:
        return ProcessPipelineManager(
            config=self._config,
            event_bus=self._event_bus,
            fake=self._fake,
        )
```

Replace `_save_results(...)` with:

```python
    def _save_results(self) -> Path:
        assert self.manager is not None
        run_dir = getattr(self.manager, "run_dir", None)
        if run_dir is None:
            run_dir = self._config.output_dir / f"headless-{int(time.time() * 1000)}"
        lane_summaries = getattr(self.manager, "lane_summaries", None)
        if lane_summaries is not None:
            return self._recorder.save_process_run(
                run_dir,
                config=self._config,
                lane_summaries=lane_summaries,
            )
        return self._recorder.save(
            run_dir,
            config=self._config,
            lane_metrics=self.manager.lane_metrics,
            latency_tracker=self.manager.latency_tracker,
            rtf_tracker=self.manager.rtf_tracker,
            event_loop_lag_tracker=self.manager.event_loop_lag_tracker,
            results={
                key: value
                for key, value in self._event_bus.snapshot_results().items()
                if value["status"] is not EventStatus.LANE_FINISHED
            },
            model_outputs=self._model_output_recorder.as_dicts(),
        )
```

Remove unused imports and fake classes from `headless/runner.py` if no tests import them. Keep direct `PipelineManager` tests unchanged because they import `multi_stream_asr.pipeline.manager.PipelineManager`.

- [ ] **Step 5: Wire web result recording to process summaries**

Modify `_record_results(app)` in `src/multi_stream_asr/web/server.py`:

```python
def _record_results(app: FastAPI) -> None:
    manager = app.state.manager
    run_dir = getattr(manager, "run_dir", None)
    lane_summaries = getattr(manager, "lane_summaries", None)
    if run_dir is not None and lane_summaries is not None:
        app.state.recorder.save_process_run(
            run_dir,
            config=app.state.config,
            lane_summaries=lane_summaries,
        )
        return

    results = {
        key: value
        for key, value in app.state.event_bus.snapshot_results().items()
        if value["status"] is not EventStatus.LANE_FINISHED
    }
    if manager.state is PipelineState.UNLOADED and not results:
        return

    run_dir = app.state.config.output_dir / f"web-{int(time.time() * 1000)}"
    app.state.recorder.save(
        run_dir,
        config=app.state.config,
        lane_metrics=getattr(manager, "lane_metrics", ()),
        latency_tracker=getattr(manager, "latency_tracker", None),
        rtf_tracker=getattr(manager, "rtf_tracker", None),
        event_loop_lag_tracker=getattr(manager, "event_loop_lag_tracker", None),
        results=results,
    )
```

- [ ] **Step 6: Run selected tests**

Run:

```bash
pytest tests/test_manager_headless.py::test_headless_runner_fake_uses_process_manager_and_saves_schema_v3 tests/test_web_server.py::test_record_results_uses_process_run_summary_when_manager_exposes_run_dir -v
```

Expected: selected tests pass.

- [ ] **Step 7: Run manager/headless/web suites**

Run:

```bash
pytest tests/test_manager_headless.py tests/test_web_server.py -v
```

Expected: all tests pass. Keep existing direct `PipelineManager` unit tests focused on the in-process manager, and keep the new `HeadlessRunner` assertions focused on process-mode behavior.

- [ ] **Step 8: Commit**

```bash
git add src/multi_stream_asr/headless/runner.py src/multi_stream_asr/web/server.py tests/test_manager_headless.py tests/test_web_server.py
git commit -m "feat: wire headless and web to lane process manager"
```

---

### Task 7: Documentation, Full Test Run, And Board Validation

**Files:**
- Modify: `docs/thread-model.md`
- Test: all tests
- Validate: remote K3 board headless run

**Interfaces:**
- Consumes: implemented process manager behavior and metrics schema v3.
- Produces: updated Chinese thread/process model documentation and real validation notes.

- [ ] **Step 1: Update thread model documentation**

Replace the opening sections of `docs/thread-model.md` with this content, then keep or adapt the existing pipeline-order sections so they describe the new process model:

```markdown
# 线程与进程模型

本项目现在使用一个主 Python 控制进程和每路一个 lane Python 子进程。主进程不加载
Zipformer，不创建 Qwen3 client，也不直接启动所有 `llama-server`。主进程负责
状态机、IPC 事件汇总、WebSocket 广播、run 目录创建和全局 `metrics.json` 落盘。

每个 lane 子进程独占该路音频输入、Zipformer recognizer、Qwen3 `llama-server`、
Qwen3 client、结果落盘和性能统计。lane 子进程内部仍使用两个 Python OS 线程：

- `asr-lane-<N>-audio-producer`：按 100 ms 一帧读取音频并写入 `OnlineFrameQueue(maxsize=1)`。
- `asr-lane-<N>-zipformer-consumer`：阻塞读取最新音频帧并同步调用 sherpa-onnx Zipformer。

每个 lane 子进程还有一个 asyncio event loop，用于接收主进程命令、调度本路 Qwen3 请求、
采样 event loop lag、向主进程发送 IPC 事件。Qwen3 推理由该 lane 子进程启动的独立
`llama-server` 进程执行。

A100/X100 绑核仍不由本项目显式执行。sherpa-onnx、SpaceMiT EP 和 `llama-server`
根据配置完成实际绑核。本项目只负责把正确的模型、端口和 EP 配置传给对应 lane。
```

Ensure the rest of the document states:

- 4 路时有 1 个主 Python 进程、4 个 lane Python 子进程、4 个 `llama-server` 子进程。
- 每个 lane Python 子进程有 2 个项目创建的 OS 线程。
- `results/laneX.txt` 和 `model_outputs/laneX.jsonl` 由 lane 子进程写。
- `config.json` 和 `metrics.json` 由主进程写。
- `metrics_schema_version = 3`。

- [ ] **Step 2: Run formatting-free static checks**

Run:

```bash
python -m compileall src
```

Expected: compile completes without syntax errors.

- [ ] **Step 3: Run full local tests**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Run local fake headless smoke**

Run:

```bash
python -m multi_stream_asr.main --headless --fake --lanes 4 --verbose
```

Expected:

- exit code 0
- one run directory under `runs/process-*`
- `metrics.json` has `"metrics_schema_version": 3`
- `results/lane0.txt` through `results/lane3.txt` exist
- `model_outputs/lane0.jsonl` through `model_outputs/lane3.jsonl` exist

- [ ] **Step 5: Sync to K3 board**

Run from repo root:

```bash
rsync -av \
  src tests docs pyproject.toml \
  bianbu@10.0.91.51:~/Workspace/projects/multi-stream-asr/
```

Expected: rsync completes without errors.

- [ ] **Step 6: Run remote fake smoke with Python 3.13 venv**

Run:

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr && . .venv313/bin/activate && python --version && python -m multi_stream_asr.main --headless --fake --lanes 4 --verbose'
```

Expected:

- Python prints `Python 3.13.x`
- command exits 0
- generated run has schema v3 metrics and per-lane files

- [ ] **Step 7: Run remote real 4-lane headless validation**

Run:

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr && . .venv313/bin/activate && python -m multi_stream_asr.main --headless --lanes 4 --verbose'
```

Expected:

- 4 lane Python child processes appear during run:

```bash
ssh bianbu@10.0.91.51 "pgrep -af 'asr-lane|multi_stream_asr'"
```

- 4 `llama-server` processes appear during run:

```bash
ssh bianbu@10.0.91.51 "pgrep -af llama-server"
```

- lane model load logs interleave rather than completing lane-by-lane serially.
- run exits 0.
- `metrics.json` has `"metrics_schema_version": 3`.
- `results/lane0.txt` through `results/lane3.txt` contain lane-owned final text.
- `model_outputs/lane0.jsonl` through `model_outputs/lane3.jsonl` contain compact records.
- online latency target is about 50 ms; record actual average and max.
- whole-segment RTF target is about 0.4; record actual average and max.
- dropped frames are checked per lane.

- [ ] **Step 8: Commit docs and any test fixes**

```bash
git add docs/thread-model.md
git commit -m "docs: update thread model for lane processes"
```

If Task 7 required small test or code fixes, include only those directly related files in this commit with a message that names the fix.

---

## Self-Review Notes

- Spec coverage: The plan covers lane child process ownership, parallel model loading, lane-owned `results` and `model_outputs`, main-process metrics summary, IPC event handling, fatal propagation, and remote board validation.
- Scope: This is one coherent subsystem change. It keeps existing `LanePipeline` behavior and adds process orchestration around it.
- Type consistency: `LaneIpcMessage`, `LaneProcessConfig`, `ProcessPipelineManager`, and `RunRecorder.save_process_run(...)` are introduced before downstream tasks consume them.
- Backward compatibility: Existing in-process `PipelineManager` tests remain valid by direct import; default headless/web behavior moves to `ProcessPipelineManager`.
