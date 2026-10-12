# verbose 模式细化性能指标记录 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `AppConfig.verbose`（已存在但从未被消费的字段）驱动一套按 lane、按指标类型分目录的详细 jsonl 流，仅在 `verbose=True` 且非循环识别（`loop_audio=False`）时落盘；普通模式和循环识别场景下行为与今天完全一致（只有 `metrics.json` 聚合值）。

**Architecture:** 新增一个通用的 `DetailJsonlWriter`（`lane_artifacts.py`），语义与现有 `LaneModelOutputJsonlRecorder` 一致——按 `run_dir/metrics_detail/<metric>/lane<N>.jsonl` 流式追加写入、每条 `flush()`、不在内存中累积。`LaneRuntime._run_started` 按 `config.verbose and not loop_audio` 决定是否构造 4 个该类实例（`frame_latency`/`event_loop_lag`/`rtf`/`shared_timing`），其中 3 个传给 `LanePipeline`，`event_loop_lag` 那个留在 `LaneRuntime` 自己的 `_track_event_loop_lag` 循环里用。`LanePipeline` 在现有 4 个 tracker `.add(...)` 调用点旁各自多调一次对应 writer 的 `.record(...)`（`None` 时跳过）。`metrics.py`、`metrics.json` schema、`recorder.py` 聚合/合并逻辑不改动。

**Tech Stack:** Python 3.13、pytest / pytest-asyncio，现有项目内的 `multiprocessing` lane 子进程模型（`AppConfig` 已整体经 pickle 传给子进程，无需新增 IPC 字段）。

## Global Constraints

- 详细记录仅在 `config.verbose and not loop_audio` 时启用；`loop_audio=True` 时不写详细文件，无论 verbose 与否（与现有 `LaneResultWriter`/`LaneModelOutputJsonlRecorder` 的 `enabled=not loop_audio` 行为保持一致）。
- 普通模式（`verbose=False`，默认）不产生 `metrics_detail/` 目录，任何字节。
- `metrics.json` 的聚合 schema（`count/total(_ms)/average(_ms)/max(_ms)`）不改动；`metrics.py` 文件不改动。
- 4 个指标各自独立目录：`metrics_detail/{frame_latency,event_loop_lag,rtf,shared_timing}/lane<N>.jsonl`，每个文件内 schema 单一（不需要 `kind` 字段）。
- `t_ms` 为该 lane run 开始到本次事件的毫秒偏移（`time.perf_counter()` 相对值），不用挂钟时间。
- 不新增命令行参数，复用已有 `--verbose`。
- 参考 spec：`docs/superpowers/specs/2026-07-14-verbose-metrics-detail-logging-design.md`。

---

## File Structure

- `src/multi_stream_asr/log/lane_artifacts.py`（修改）：新增 `DetailJsonlWriter` 类。
- `src/multi_stream_asr/pipeline/pipeline.py`（修改）：`LanePipeline` 新增 3 个可选 writer 构造参数，在 4 个现有 tracker 埋点中的 3 个旁调用对应 writer。
- `src/multi_stream_asr/pipeline/lane_runtime.py`（修改）：`_run_started` 按条件构造/传递/关闭 4 个 writer；`_track_event_loop_lag` 调用第 4 个 writer；`_reset_for_run` 重置引用。
- `docs/thread-model.md`（修改）：补充 `metrics_detail/` 文件归属说明。
- `tests/test_lane_artifacts.py`（修改）：新增 `DetailJsonlWriter` 单测。
- `tests/test_pipeline.py`（修改）：新增 `LanePipeline` writer 埋点单测。
- `tests/test_lane_runtime.py`（修改）：新增 verbose × loop_audio 组合的端到端行为单测。

---

### Task 1: `DetailJsonlWriter`

**Files:**
- Modify: `src/multi_stream_asr/log/lane_artifacts.py`
- Test: `tests/test_lane_artifacts.py`

**Interfaces:**
- Produces: `DetailJsonlWriter(path: Path, *, monotonic: Callable[[], float] = time.perf_counter)`，方法 `record(fields: dict[str, object]) -> None`（写入 `{"t_ms": <float>, **fields}` 一行 JSON，`sort_keys=True`）、`close() -> None`（幂等）。`t_ms` 是构造时刻到本次 `record()` 调用时刻的毫秒差（用传入的 `monotonic` 可调用对象，默认为 `time.perf_counter`）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_lane_artifacts.py` 顶部的 import 里加入 `DetailJsonlWriter`：

```python
from multi_stream_asr.log.lane_artifacts import (
    DetailJsonlWriter,
    LaneModelOutputJsonlRecorder,
    LaneResultWriter,
)
```

在文件末尾追加：

```python
def test_detail_jsonl_writer_appends_t_ms_and_records_in_order(tmp_path: Path) -> None:
    clock_values = iter([10.0, 10.05, 10.2])
    writer = DetailJsonlWriter(
        tmp_path / "metrics_detail" / "frame_latency" / "lane0.jsonl",
        monotonic=lambda: next(clock_values),
    )

    writer.record({"latency_ms": 12.5})
    writer.record({"latency_ms": 8.0})
    writer.close()

    lines = [
        json.loads(line)
        for line in (tmp_path / "metrics_detail" / "frame_latency" / "lane0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert lines == [
        {"t_ms": 50.0, "latency_ms": 12.5},
        {"t_ms": 200.0, "latency_ms": 8.0},
    ]


def test_detail_jsonl_writer_close_is_idempotent_and_stops_writes(tmp_path: Path) -> None:
    writer = DetailJsonlWriter(tmp_path / "detail.jsonl")
    writer.record({"rtf": 0.4, "sentence_id": 0})
    writer.close()
    writer.close()
    writer.record({"rtf": 0.5, "sentence_id": 1})

    lines = (tmp_path / "detail.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["sentence_id"] == 0


def test_detail_jsonl_writer_handles_concurrent_records(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.jsonl"
    writer = DetailJsonlWriter(path)
    start = Barrier(3)

    def record_range(offset: int) -> None:
        start.wait()
        for i in range(offset, offset + 50):
            writer.record({"sentence_id": i})

    threads = [Thread(target=record_range, args=(offset,)) for offset in (0, 50)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()
    writer.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {record["sentence_id"] for record in records} == set(range(100))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/liangff/Workspace/projects/multi-stream-asr && .venv313/bin/python -m pytest tests/test_lane_artifacts.py -v 2>&1 | tail -30`
Expected: `ImportError: cannot import name 'DetailJsonlWriter'`（3 个新测试全部因导入失败而报错）。

- [ ] **Step 3: 实现 `DetailJsonlWriter`**

在 `src/multi_stream_asr/log/lane_artifacts.py` 顶部的 import 里加入 `time` 和 `Callable`：

```python
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import TextIO

from multi_stream_asr.log.model_output import ModelOutputRecord
from multi_stream_asr.models import ASREvent, EventStatus
from multi_stream_asr.utils.paths import ensure_dir
```

在 `compact_model_output_record` 函数之后、`LaneResultWriter` 类之前插入：

```python
class DetailJsonlWriter:
    def __init__(
        self,
        path: Path,
        *,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.path = path
        self._monotonic = monotonic
        self._started_at = monotonic()
        ensure_dir(path.parent)
        self._file: TextIO | None = path.open("w", encoding="utf-8")
        self._lock = Lock()

    def record(self, fields: dict[str, object]) -> None:
        t_ms = (self._monotonic() - self._started_at) * 1000.0
        payload = json.dumps({"t_ms": t_ms, **fields}, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if self._file is None:
                return
            self._file.write(payload)
            self._file.write("\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is None:
                return
            self._file.close()
            self._file = None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/liangff/Workspace/projects/multi-stream-asr && .venv313/bin/python -m pytest tests/test_lane_artifacts.py -v 2>&1 | tail -30`
Expected: 全部 PASS（含之前已有的 5 个测试 + 新增 3 个）。

- [ ] **Step 5: Commit**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr
git add src/multi_stream_asr/log/lane_artifacts.py tests/test_lane_artifacts.py
git commit -m "feat: add DetailJsonlWriter for verbose metrics detail logging"
```

---

### Task 2: `LanePipeline` 埋点

**Files:**
- Modify: `src/multi_stream_asr/pipeline/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `DetailJsonlWriter`（Task 1）的 `record(fields: dict[str, object]) -> None`。
- Produces: `LanePipeline.__init__` 新增关键字参数 `frame_latency_writer: DetailJsonlWriter | None = None`、`rtf_writer: DetailJsonlWriter | None = None`、`shared_timing_writer: DetailJsonlWriter | None = None`，均存为同名实例属性；`event_loop_lag` 不在这里处理（属于 Task 3 的 `LaneRuntime`）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_pipeline.py` 顶部 import 区加入：

```python
from multi_stream_asr.log.metrics import SHARED_TIMING_FIELDS
```

（`LatencyTracker, RtfTracker, SharedTimingsTracker` 已有的 import 行不变，只追加 `SHARED_TIMING_FIELDS`。）

在文件末尾追加一个记录用的 fake writer 和三个测试：

```python
class RecordingDetailWriter:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, fields: dict) -> None:
        self.records.append(fields)


@pytest.mark.asyncio
async def test_frame_latency_writer_records_one_entry_per_accept_call():
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            make_frame(frame_index=1, samples=[0.3, 0.4], timestamp_ms=200),
            AudioFrame.finish(stream_id=0, frame_index=2, timestamp_ms=300),
        ],
        pause_between_frames=True,
    )
    online = FakeOnlineRecognizer(
        [
            make_online(text="hello", timestamp_ms=100),
            make_online(text="hello", timestamp_ms=200, is_endpoint=True),
        ]
    )
    frame_latency_writer = RecordingDetailWriter()
    clock_values = iter([1.0, 1.050, 2.0, 2.025])
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=FakeOfflineRecognizer(text="offline hello"),
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        frame_latency_writer=frame_latency_writer,
        monotonic=lambda: next(clock_values),
    )

    await pipeline.run(source)

    assert [entry["latency_ms"] for entry in frame_latency_writer.records] == pytest.approx(
        [50.0, 25.0]
    )
    assert all("t_ms" not in entry for entry in frame_latency_writer.records)


@pytest.mark.asyncio
async def test_rtf_writer_records_sentence_id_and_offline_timing():
    bus = FrontendEventBus()
    rtf_writer = RecordingDetailWriter()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [make_online(text="online endpoint", timestamp_ms=100, is_endpoint=True)]
        ),
        offline_recognizer=FakeOfflineRecognizer(text="offline final"),
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        rtf_writer=rtf_writer,
    )
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            AudioFrame.finish(stream_id=0, frame_index=1, timestamp_ms=200),
        ],
        pause_between_frames=True,
    )

    await pipeline.run(source)

    assert rtf_writer.records == [
        {"sentence_id": 0, "audio_seconds": 1.0, "elapsed_ms": 400.0, "rtf": 0.4}
    ]


@pytest.mark.asyncio
async def test_shared_timing_writer_records_sentence_id_and_filtered_fields():
    bus = FrontendEventBus()
    shared_timing_writer = RecordingDetailWriter()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [make_online(text="online endpoint", timestamp_ms=100, is_endpoint=True)]
        ),
        offline_recognizer=RawResponseOfflineRecognizer(),
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        shared_timing_writer=shared_timing_writer,
    )
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            AudioFrame.finish(stream_id=0, frame_index=1, timestamp_ms=200),
        ],
        pause_between_frames=True,
    )

    await pipeline.run(source)

    assert shared_timing_writer.records == [
        {
            "sentence_id": 0,
            "queue_ms": 10.0,
            "encode_ms": 100.0,
            "prefill_ms": 20.0,
            "decode_ms": 200.0,
            "total_ms": 330.0,
        }
    ]
    assert set(SHARED_TIMING_FIELDS) == {"queue_ms", "encode_ms", "prefill_ms", "decode_ms", "total_ms"}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/liangff/Workspace/projects/multi-stream-asr && .venv313/bin/python -m pytest tests/test_pipeline.py -v -k "writer" 2>&1 | tail -40`
Expected: FAIL，`TypeError: LanePipeline.__init__() got an unexpected keyword argument 'frame_latency_writer'`（三个新测试均失败）。

- [ ] **Step 3: 实现埋点**

在 `src/multi_stream_asr/pipeline/pipeline.py` 顶部 import 区，把：

```python
from multi_stream_asr.log.metrics import LatencyTracker, RtfTracker, SharedTimingsTracker
```

改为：

```python
from multi_stream_asr.log.lane_artifacts import DetailJsonlWriter
from multi_stream_asr.log.metrics import (
    SHARED_TIMING_FIELDS,
    LatencyTracker,
    RtfTracker,
    SharedTimingsTracker,
)
```

`__init__` 签名（第 27-44 行）里，在 `model_output_recorder: ModelOutputRecorder | None = None,` 之后加入 3 个新参数：

```python
        model_output_recorder: ModelOutputRecorder | None = None,
        frame_latency_writer: DetailJsonlWriter | None = None,
        rtf_writer: DetailJsonlWriter | None = None,
        shared_timing_writer: DetailJsonlWriter | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
```

`__init__` 函数体里，在 `self.model_output_recorder = model_output_recorder` 之后加入：

```python
        self.model_output_recorder = model_output_recorder
        self.frame_latency_writer = frame_latency_writer
        self.rtf_writer = rtf_writer
        self.shared_timing_writer = shared_timing_writer
```

第 347-351 行（`_consume_frames_thread` 里在线延迟埋点）从：

```python
            started_at = self._monotonic()
            recognition = self._run_async_in_thread(self.online_recognizer.accept(frame))
            self.latency_tracker.add((self._monotonic() - started_at) * 1000.0)
            if recognition is not None:
                self._handle_online_result(recognition)
```

改为：

```python
            started_at = self._monotonic()
            recognition = self._run_async_in_thread(self.online_recognizer.accept(frame))
            latency_ms = (self._monotonic() - started_at) * 1000.0
            self.latency_tracker.add(latency_ms)
            if self.frame_latency_writer is not None:
                self.frame_latency_writer.record({"latency_ms": latency_ms})
            if recognition is not None:
                self._handle_online_result(recognition)
```

`_schedule_offline` 里的 `run_offline()`（第 471-484 行）从：

```python
            result = await self.offline_recognizer.recognize(pcm=pcm, prompt=prompt)
            self.rtf_tracker.add(result.audio_seconds, result.elapsed_ms)
            if result.timings is not None:
                self.shared_timings_tracker.add(result.timings)
            self._record_model_output(
```

改为：

```python
            result = await self.offline_recognizer.recognize(pcm=pcm, prompt=prompt)
            self.rtf_tracker.add(result.audio_seconds, result.elapsed_ms)
            if self.rtf_writer is not None:
                self.rtf_writer.record(
                    {
                        "sentence_id": sentence_id,
                        "audio_seconds": result.audio_seconds,
                        "elapsed_ms": result.elapsed_ms,
                        "rtf": result.rtf,
                    }
                )
            if result.timings is not None:
                self.shared_timings_tracker.add(result.timings)
                if self.shared_timing_writer is not None:
                    filtered = _filter_shared_timings(result.timings)
                    if filtered:
                        self.shared_timing_writer.record(
                            {"sentence_id": sentence_id, **filtered}
                        )
            self._record_model_output(
```

在文件末尾（`class LanePipeline` 定义之后，模块级别）加入辅助函数：

```python
def _filter_shared_timings(timings: dict[str, float]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in timings.items()
        if key in SHARED_TIMING_FIELDS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
```

（`math` 已在文件顶部 `import math` 引入，无需新增。）

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/liangff/Workspace/projects/multi-stream-asr && .venv313/bin/python -m pytest tests/test_pipeline.py -v 2>&1 | tail -50`
Expected: 全部 PASS（含既有测试 + 新增 3 个）。

- [ ] **Step 5: Commit**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr
git add src/multi_stream_asr/pipeline/pipeline.py tests/test_pipeline.py
git commit -m "feat: wire LanePipeline to optional verbose detail writers"
```

---

### Task 3: `LaneRuntime` 接线（verbose × loop_audio 开关）

**Files:**
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py`
- Test: `tests/test_lane_runtime.py`

**Interfaces:**
- Consumes: `DetailJsonlWriter`（Task 1）、`LanePipeline(..., frame_latency_writer=..., rtf_writer=..., shared_timing_writer=...)`（Task 2）。
- Produces: 无新公开接口；行为改动——`config.verbose and not loop_audio` 时，run 目录下出现 `metrics_detail/{frame_latency,event_loop_lag,rtf,shared_timing}/lane<stream_id>.jsonl` 四个文件。

- [ ] **Step 1: 写失败测试**

在 `tests/test_lane_runtime.py` 里找到 `test_fake_lane_run_forwards_events_and_writes_artifacts` 测试（第 411 行附近），在它之后追加以下三个新测试：

```python
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
```

还要单独测一下 `_track_event_loop_lag` 确实往 writer 写数据（避免上面 fake-lane 端到端测试因运行太快、0.1s 采样间隔来不及触发而只验证了"文件存在"这种弱断言）。在文件末尾追加：

```python
def test_track_event_loop_lag_records_into_writer(tmp_path: Path) -> None:
    from multi_stream_asr.log.lane_artifacts import DetailJsonlWriter

    config = AppConfig(lanes=1, output_dir=tmp_path, verbose=True)
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
```

`tests/test_lane_runtime.py` 顶部已经 `from contextlib import suppress` 和 `import json`（第 4、9 行），新测试直接复用，无需新增 import。

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/liangff/Workspace/projects/multi-stream-asr && .venv313/bin/python -m pytest tests/test_lane_runtime.py -v -k "verbose or event_loop_lag_records" 2>&1 | tail -60`
Expected: FAIL——`test_verbose_lane_run_writes_metrics_detail_files` 和 `test_track_event_loop_lag_records_into_writer` 报 `FileNotFoundError`（目录/文件未创建，或 `AttributeError: 'LaneRuntime' object has no attribute '_event_loop_lag_writer'`）；`test_non_verbose_...` 和 `test_verbose_loop_audio_...` 此时因为目前代码根本不产生 `metrics_detail` 目录会**意外通过**——这是预期的（占位断言，等 Step 3 加入功能后它们仍应保持通过，构成回归测试）。

- [ ] **Step 3: 实现接线**

在 `src/multi_stream_asr/pipeline/lane_runtime.py` 顶部 import 区，把：

```python
from multi_stream_asr.log.lane_artifacts import LaneModelOutputJsonlRecorder, LaneResultWriter
```

改为：

```python
from multi_stream_asr.log.lane_artifacts import (
    DetailJsonlWriter,
    LaneModelOutputJsonlRecorder,
    LaneResultWriter,
)
```

在 `LaneRuntime.__init__`（第 174-200 行）里，`self._event_loop_lag_tracker = EventLoopLagTracker()` 之后加入 4 个 writer 属性初始化：

```python
        self._event_loop_lag_tracker = EventLoopLagTracker()
        self._frame_latency_writer: DetailJsonlWriter | None = None
        self._event_loop_lag_writer: DetailJsonlWriter | None = None
        self._rtf_writer: DetailJsonlWriter | None = None
        self._shared_timing_writer: DetailJsonlWriter | None = None
```

在 `_reset_for_run`（第 427-437 行）里，`self._event_loop_lag_tracker = EventLoopLagTracker()` 之后同样加入重置：

```python
        self._event_loop_lag_tracker = EventLoopLagTracker()
        self._frame_latency_writer = None
        self._event_loop_lag_writer = None
        self._rtf_writer = None
        self._shared_timing_writer = None
```

`_run_started`（第 306-425 行）里，在声明 `model_output_recorder: LaneModelOutputJsonlRecorder | None = None` 那一行之后加入本地变量声明：

```python
        result_writer: LaneResultWriter | None = None
        model_output_recorder: LaneModelOutputJsonlRecorder | None = None
        frame_latency_writer: DetailJsonlWriter | None = None
        event_loop_lag_writer: DetailJsonlWriter | None = None
        rtf_writer: DetailJsonlWriter | None = None
        shared_timing_writer: DetailJsonlWriter | None = None
```

在 `model_output_recorder = LaneModelOutputJsonlRecorder(...)` 语句之后（`self._artifacts_created = True` 之前）插入按条件构造 4 个 writer 的代码：

```python
            model_output_recorder = LaneModelOutputJsonlRecorder(
                run_dir / "model_outputs" / f"lane{self.lane.stream_id}.jsonl",
                enabled=not loop_audio,
            )
            if self.app_config.verbose and not loop_audio:
                detail_dir = run_dir / "metrics_detail"
                lane_filename = f"lane{self.lane.stream_id}.jsonl"
                frame_latency_writer = DetailJsonlWriter(detail_dir / "frame_latency" / lane_filename)
                event_loop_lag_writer = DetailJsonlWriter(detail_dir / "event_loop_lag" / lane_filename)
                rtf_writer = DetailJsonlWriter(detail_dir / "rtf" / lane_filename)
                shared_timing_writer = DetailJsonlWriter(detail_dir / "shared_timing" / lane_filename)
                self._event_loop_lag_writer = event_loop_lag_writer
            self._artifacts_created = True
```

在 `self._pipeline = LanePipeline(` 调用（第 328-347 行）里，`model_output_recorder=model_output_recorder,` 之后加入 3 个新关键字参数：

```python
                model_output_recorder=model_output_recorder,
                frame_latency_writer=frame_latency_writer,
                rtf_writer=rtf_writer,
                shared_timing_writer=shared_timing_writer,
```

在 `finally` 块里（第 409-420 行），`model_output_recorder` 关闭逻辑之后加入 4 个 writer 的关闭逻辑（沿用相同的 `close_error` 收集模式）：

```python
            if model_output_recorder is not None:
                try:
                    model_output_recorder.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
            for writer in (
                frame_latency_writer,
                event_loop_lag_writer,
                rtf_writer,
                shared_timing_writer,
            ):
                if writer is not None:
                    try:
                        writer.close()
                    except BaseException as exc:
                        if close_error is None:
                            close_error = exc
```

`_track_event_loop_lag`（第 603-609 行）从：

```python
    async def _track_event_loop_lag(self, interval_seconds: float = 0.1) -> None:
        expected = time.perf_counter() + interval_seconds
        while True:
            await asyncio.sleep(interval_seconds)
            now = time.perf_counter()
            self._event_loop_lag_tracker.add(max(0.0, (now - expected) * 1000.0))
            expected = now + interval_seconds
```

改为：

```python
    async def _track_event_loop_lag(self, interval_seconds: float = 0.1) -> None:
        expected = time.perf_counter() + interval_seconds
        while True:
            await asyncio.sleep(interval_seconds)
            now = time.perf_counter()
            lag_ms = max(0.0, (now - expected) * 1000.0)
            self._event_loop_lag_tracker.add(lag_ms)
            if self._event_loop_lag_writer is not None:
                self._event_loop_lag_writer.record({"lag_ms": lag_ms})
            expected = now + interval_seconds
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/liangff/Workspace/projects/multi-stream-asr && .venv313/bin/python -m pytest tests/test_lane_runtime.py -v 2>&1 | tail -80`
Expected: 全部 PASS（含既有全部测试 + 新增 4 个）。

- [ ] **Step 5: 跑全量测试确认无回归**

Run: `cd /home/liangff/Workspace/projects/multi-stream-asr && .venv313/bin/python -m pytest tests/ 2>&1 | tail -15`
Expected: 除已知与本改动无关的 `test_setup_script.py::test_setup_script_installs_models_from_verified_archives_and_recovers_downloads`（缺 python3.13 apt 环境导致）外全部 PASS。特别确认 `tests/test_config.py`（`metrics.json` 相关）和 `tests/test_process_manager.py` 全部通过，证明聚合 schema 和多进程路径未受影响。

- [ ] **Step 6: Commit**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr
git add src/multi_stream_asr/pipeline/lane_runtime.py tests/test_lane_runtime.py
git commit -m "feat: gate verbose metrics detail files behind config.verbose and loop_audio"
```

---

### Task 4: 文档更新

**Files:**
- Modify: `docs/thread-model.md`

**Interfaces:**
- 无代码接口，纯文档。

- [ ] **Step 1: 更新文件归属说明**

在 `docs/thread-model.md` 第 139-150 行的"文件归属"小节里，把：

```markdown
## 文件归属

每次运行使用一个共享 run 目录，写入责任固定：

- lane 写 `results/laneX.txt`，保存该路最终完整识别结果；
- lane 写 `model_outputs/laneX.jsonl`，每条严格只含模型、音频时长、RTF、路 ID、句 ID
  和识别结果六个字段；
- 主进程写 `config.json`；
- 主进程合并各 lane 指标摘要并写 `metrics.json`。

`metrics.json` 使用 `metrics_schema_version = 3`，包含全局及逐 lane 的共享 Qwen timing
摘要。主进程不会重写 lane-owned 结果文件。
```

改为：

```markdown
## 文件归属

每次运行使用一个共享 run 目录，写入责任固定：

- lane 写 `results/laneX.txt`，保存该路最终完整识别结果；
- lane 写 `model_outputs/laneX.jsonl`，每条严格只含模型、音频时长、RTF、路 ID、句 ID
  和识别结果六个字段；
- 主进程写 `config.json`；
- 主进程合并各 lane 指标摘要并写 `metrics.json`。

`metrics.json` 使用 `metrics_schema_version = 3`，包含全局及逐 lane 的共享 Qwen timing
摘要。主进程不会重写 lane-owned 结果文件。

### verbose 详细指标记录

`--verbose` 开启且该路输入未循环（`loop_audio=False`）时，每个 lane 额外流式写入
`metrics_detail/{frame_latency,event_loop_lag,rtf,shared_timing}/laneX.jsonl`，
每行一条原始记录，字段含 `t_ms`（该路 run 开始到本次事件的毫秒偏移）及对应指标的
原始值（`rtf`/`shared_timing` 额外带 `sentence_id`，可与 `model_outputs/laneX.jsonl`
关联）。普通模式或循环识别时不产生这些文件，`metrics.json` 的聚合值不受影响。
```

- [ ] **Step 2: Commit**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr
git add docs/thread-model.md
git commit -m "docs: document verbose metrics detail file layout"
```
