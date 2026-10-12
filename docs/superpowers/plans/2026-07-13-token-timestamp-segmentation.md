# Token 时间戳音频段切分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 Zipformer 最后 token 的 run 内时间戳切分离线音频，保留下一句字头，并隔离 warmup、手动停止和前一次识别的时间轴。

**Architecture:** 每次识别开始时在现有 Zipformer recognizer 上创建新的 Sherpa stream，不重载 ONNX session；Zipformer 将最后 token 的 run 内时间透传给 lane pipeline。`SegmentCache` 维护绝对样本区间并提供无重叠 `cut()`，pipeline 负责时间戳校验、整缓存降级、空 endpoint 尾部保留和自然 finish 冲刷。

**Tech Stack:** Python 3.13、NumPy、sherpa-onnx、asyncio、pytest、SpaceMIT K3 `.venv313`

## Global Constraints

- `cut_margin_seconds` 默认固定为 `0.1`，且必须满足 `0 <= cut_margin_seconds < rule2_min_trailing_silence`。
- 空文本 endpoint 保留缓存尾部 `rule2_min_trailing_silence` 秒，不提交 Qwen。
- 自然 finish 冲刷非空残句；手动停止和 shutdown 丢弃未 endpoint 残段。
- 每次 run 创建新的 Sherpa stream，但复用已加载 recognizer、模型和 ONNX session。
- 新 stream 的创建、Zipformer 推理和 reset 都必须在现有 Zipformer consumer 线程执行。
- 不修改 sherpa-onnx、进程模型、`OnlineFrameQueue(maxsize=1)`、模型/provider/亲和配置、前端、IPC 和结果文件格式。
- 远程设备只使用 Python 3.13 和项目 `.venv313`；安装依赖前必须激活环境并导出 Spacemit/Aliyun package index 变量。
- 正确性优先；目标维持 online 平均延迟约 `50ms`、offline RTF 约 `0.4`，且不新增丢帧。

---

## File Structure

- `src/multi_stream_asr/config.py`: endpoint 切分 margin 默认值与启动校验。
- `src/multi_stream_asr/models.py`: lane 内部在线结果增加最后 token 的 run 内时间。
- `src/multi_stream_asr/asr/online/base.py`: 在线 recognizer 生命周期增加 `start_run()`。
- `src/multi_stream_asr/asr/online/zipformer.py`: 新建 run stream，读取完整 Sherpa 结果并计算 token 时间。
- `src/multi_stream_asr/buffer/segment_cache.py`: 维护绝对样本坐标并执行无重叠切分。
- `src/multi_stream_asr/pipeline/pipeline.py`: endpoint、空 endpoint、自然 finish 和降级策略。
- `src/multi_stream_asr/pipeline/lane_runtime.py`: 传入 endpoint 配置并让内置 fake 支持 run 生命周期。
- `src/multi_stream_asr/pipeline/manager.py`: 旧 manager 构造 pipeline 时传入同一 endpoint 配置。
- `tests/conftest.py`: 在线结果和 recognizer 测试替身支持 token 时间与 `start_run()`。
- `tests/test_config.py`: 默认值与非法 margin。
- `tests/test_buffers_metrics.py`: 样本坐标和 `cut()` 边界。
- `tests/test_zipformer.py`: Sherpa 完整结果、兼容降级和 stream 生命周期。
- `tests/test_pipeline.py`: PCM 边界、事件、自然 finish 和手动停止。
- `tests/test_lane_runtime.py`: consumer owner thread 上的 run 初始化。

---

### Task 1: 配置与内部接口契约

**Files:**
- Modify: `src/multi_stream_asr/config.py:104-109`
- Modify: `src/multi_stream_asr/models.py:48-56`
- Modify: `src/multi_stream_asr/asr/online/base.py:8-11`
- Modify: `tests/conftest.py:26-79`
- Modify: `tests/test_config.py:71-102`
- Test: `tests/test_models_state.py`

**Interfaces:**
- Produces: `EndpointConfig.cut_margin_seconds: float`
- Produces: `OnlineRecognition.last_token_stream_s: float | None`
- Produces: `OnlineRecognizer.start_run() -> None`
- Produces: `make_online(..., last_token_stream_s: float | None = None)`

- [ ] **Step 1: Write failing config and model tests**

Add these assertions and cases to `tests/test_config.py`:

```python
def test_default_config_values():
    config = AppConfig()
    assert config.endpoint == EndpointConfig(
        rule1_min_trailing_silence=1.0,
        rule2_min_trailing_silence=0.5,
        rule3_min_utterance_length=10.0,
        cut_margin_seconds=0.1,
    )
    assert config.segment_cache_seconds == 15.0


@pytest.mark.parametrize("margin", [-0.001, 0.5, 0.6, float("nan"), float("inf")])
def test_endpoint_rejects_invalid_cut_margin(margin: float):
    with pytest.raises(ValueError, match="cut_margin_seconds"):
        EndpointConfig(cut_margin_seconds=margin)


def test_endpoint_accepts_margin_strictly_below_rule2():
    endpoint = EndpointConfig(
        rule2_min_trailing_silence=0.25,
        cut_margin_seconds=0.249,
    )
    assert endpoint.cut_margin_seconds == 0.249
```

Add to `tests/test_models_state.py`:

```python
def test_online_recognition_defaults_token_time_to_none():
    result = OnlineRecognition(
        stream_id=0,
        sentence_id=0,
        revision=0,
        text="你好",
        timestamp_ms=100,
    )
    assert result.last_token_stream_s is None
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest \
  tests/test_config.py \
  tests/test_models_state.py -q
```

Expected: failures report the missing `cut_margin_seconds` and `last_token_stream_s` fields.

- [ ] **Step 3: Implement the config and result fields**

In `src/multi_stream_asr/config.py`, replace `EndpointConfig` with:

```python
@dataclass(frozen=True)
class EndpointConfig:
    rule1_min_trailing_silence: float = 1.0
    rule2_min_trailing_silence: float = 0.5
    rule3_min_utterance_length: float = 10.0
    cut_margin_seconds: float = 0.1

    def __post_init__(self) -> None:
        if not math.isfinite(self.cut_margin_seconds) or not (
            0.0 <= self.cut_margin_seconds < self.rule2_min_trailing_silence
        ):
            raise ValueError(
                "cut_margin_seconds must be finite, non-negative, and less than "
                "rule2_min_trailing_silence"
            )
```

Add `import math` with the standard-library imports. In `src/multi_stream_asr/models.py`, append the defaulted field after `is_endpoint`:

```python
@dataclass(frozen=True)
class OnlineRecognition:
    stream_id: int
    sentence_id: int
    revision: int
    text: str
    timestamp_ms: int
    is_endpoint: bool = False
    last_token_stream_s: float | None = None
```

In `src/multi_stream_asr/asr/online/base.py`, define the run lifecycle explicitly:

```python
class OnlineRecognizer(Protocol):
    async def start_run(self) -> None: ...

    async def accept(self, frame: AudioFrame) -> OnlineRecognition | None: ...

    async def reset(self) -> None: ...
```

Update `tests/conftest.py` so `make_online()` accepts and forwards the new field:

```python
def make_online(
    *,
    stream_id: int = 0,
    sentence_id: int = 0,
    revision: int = 0,
    text: str,
    timestamp_ms: int,
    is_endpoint: bool = False,
    last_token_stream_s: float | None = None,
) -> OnlineRecognition:
    return OnlineRecognition(
        stream_id=stream_id,
        sentence_id=sentence_id,
        revision=revision,
        text=text,
        timestamp_ms=timestamp_ms,
        is_endpoint=is_endpoint,
        last_token_stream_s=last_token_stream_s,
    )
```

Give the shared fake a countable lifecycle method:

```python
class FakeOnlineRecognizer:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.start_run_count = 0
        self.reset_count = 0
        self.accepted_frames: list[AudioFrame] = []

    async def start_run(self):
        self.start_run_count += 1

    async def accept(self, frame):
        self.accepted_frames.append(frame)
        if frame.is_finish:
            return None
        return self.outputs.pop(0) if self.outputs else None

    async def reset(self):
        self.reset_count += 1
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest \
  tests/test_config.py \
  tests/test_models_state.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add src/multi_stream_asr/config.py \
  src/multi_stream_asr/models.py \
  src/multi_stream_asr/asr/online/base.py \
  tests/conftest.py tests/test_config.py tests/test_models_state.py
git commit -m "feat: define token segment timing contract"
```

---

### Task 2: SegmentCache 绝对样本坐标与无重叠切分

**Files:**
- Modify: `src/multi_stream_asr/buffer/segment_cache.py`
- Modify: `tests/test_buffers_metrics.py:146-164`

**Interfaces:**
- Produces: `SegmentCache.start_time_s: float`
- Produces: `SegmentCache.end_time_s: float`
- Produces: `SegmentCache.cut(stream_time_s: float) -> np.ndarray`
- Preserves: `put()`, `audio()`, `reset()` public behavior

- [ ] **Step 1: Write failing sample-coordinate tests**

Add these tests to `tests/test_buffers_metrics.py`:

```python
def test_segment_cache_cuts_across_chunks_without_overlap_or_gap():
    cache = SegmentCache(sample_rate=10, capacity_seconds=10.0)
    cache.put(make_frame(0, 0, 1.0, 3))
    cache.put(make_frame(0, 1, 2.0, 4))
    cache.put(make_frame(0, 2, 3.0, 3))

    first = cache.cut(0.5)
    second = cache.cut(0.8)
    tail = cache.audio()

    np.testing.assert_array_equal(first, np.asarray([1, 1, 1, 2, 2], dtype=np.float32))
    np.testing.assert_array_equal(second, np.asarray([2, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(tail, np.asarray([3, 3], dtype=np.float32))
    np.testing.assert_array_equal(
        np.concatenate((first, second, tail)),
        np.asarray([1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=np.float32),
    )
    assert cache.start_time_s == 0.8
    assert cache.end_time_s == 1.0


def test_segment_cache_rounds_and_clamps_cut_time():
    cache = SegmentCache(sample_rate=10, capacity_seconds=10.0)
    cache.put(make_frame(0, 0, 1.0, 5))

    np.testing.assert_array_equal(cache.cut(0.26), np.ones(3, dtype=np.float32))
    assert cache.cut(-100.0).size == 0
    np.testing.assert_array_equal(cache.cut(100.0), np.ones(2, dtype=np.float32))
    assert cache.start_time_s == cache.end_time_s == 0.5


def test_segment_cache_capacity_trim_advances_absolute_start():
    cache = SegmentCache(sample_rate=10, capacity_seconds=0.5)
    cache.put(make_frame(0, 0, 1.0, 4))
    cache.put(make_frame(0, 1, 2.0, 4))

    assert cache.start_time_s == 0.3
    assert cache.end_time_s == 0.8
    np.testing.assert_array_equal(
        cache.audio(), np.asarray([1, 2, 2, 2, 2], dtype=np.float32)
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_segment_cache_rejects_non_finite_cut_time(value: float):
    cache = SegmentCache(sample_rate=10, capacity_seconds=1.0)
    with pytest.raises(ValueError, match="stream_time_s"):
        cache.cut(value)


def test_segment_cache_reset_preserves_monotonic_clock():
    cache = SegmentCache(sample_rate=10, capacity_seconds=1.0)
    cache.put(make_frame(0, 0, 1.0, 4))
    cache.reset()
    assert cache.start_time_s == cache.end_time_s == 0.4
    cache.put(make_frame(0, 1, 2.0, 2))
    assert cache.start_time_s == 0.4
    assert cache.end_time_s == 0.6
```

- [ ] **Step 2: Run tests and confirm missing API failures**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest \
  tests/test_buffers_metrics.py -k segment_cache -q
```

Expected: new tests fail because `cut`, `start_time_s`, and `end_time_s` do not exist.

- [ ] **Step 3: Replace SegmentCache with coordinate-aware implementation**

Implement `src/multi_stream_asr/buffer/segment_cache.py` as:

```python
from __future__ import annotations

import math
from collections import deque

import numpy as np

from multi_stream_asr.models import AudioFrame


class SegmentCache:
    def __init__(self, sample_rate: int, capacity_seconds: float) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if capacity_seconds <= 0:
            raise ValueError("capacity_seconds must be positive")
        self.sample_rate = sample_rate
        self.capacity_seconds = capacity_seconds
        self.capacity_samples = int(sample_rate * capacity_seconds)
        self._chunks: deque[np.ndarray] = deque()
        self._samples = 0
        self._start_sample = 0
        self._end_sample = 0

    @property
    def start_time_s(self) -> float:
        return self._start_sample / self.sample_rate

    @property
    def end_time_s(self) -> float:
        return self._end_sample / self.sample_rate

    def put(self, frame: AudioFrame) -> None:
        if frame.is_finish:
            return
        chunk = np.asarray(frame.pcm, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        self._chunks.append(chunk)
        size = int(chunk.size)
        self._samples += size
        self._end_sample += size
        self._trim_left()

    def _trim_left(self) -> None:
        while self._samples > self.capacity_samples and self._chunks:
            overflow = self._samples - self.capacity_samples
            first = self._chunks[0]
            if overflow >= first.size:
                removed = int(first.size)
                self._chunks.popleft()
                self._samples -= removed
                self._start_sample += removed
                continue
            self._chunks[0] = first[overflow:]
            self._samples -= int(overflow)
            self._start_sample += int(overflow)
            break

    def audio(self) -> np.ndarray:
        if not self._chunks:
            return np.asarray([], dtype=np.float32)
        if len(self._chunks) == 1:
            return np.asarray(self._chunks[0], dtype=np.float32)
        return np.concatenate(tuple(self._chunks)).astype(np.float32, copy=False)

    def cut(self, stream_time_s: float) -> np.ndarray:
        if not math.isfinite(stream_time_s):
            raise ValueError("stream_time_s must be finite")
        requested = int(round(stream_time_s * self.sample_rate))
        cut_sample = max(self._start_sample, min(requested, self._end_sample))
        head_samples = cut_sample - self._start_sample
        full = self.audio()
        head = full[:head_samples]
        tail = full[head_samples:]
        self._chunks.clear()
        if tail.size:
            self._chunks.append(tail)
        self._samples = int(tail.size)
        self._start_sample = cut_sample
        return head

    def reset(self) -> None:
        self._chunks.clear()
        self._samples = 0
        self._start_sample = self._end_sample
```

- [ ] **Step 4: Run cache and full buffer tests**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest tests/test_buffers_metrics.py -q
```

Expected: all tests in `tests/test_buffers_metrics.py` pass.

- [ ] **Step 5: Commit cache behavior**

```bash
git add src/multi_stream_asr/buffer/segment_cache.py tests/test_buffers_metrics.py
git commit -m "feat: cut segment cache by stream time"
```

---

### Task 3: Zipformer run stream 与 token 时间戳

**Files:**
- Modify: `src/multi_stream_asr/asr/online/zipformer.py:61-88`
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py:90-116`
- Modify: `tests/test_zipformer.py:142-192`

**Interfaces:**
- Consumes: `OnlineRecognition.last_token_stream_s`
- Produces: `ZipformerRecognizer.start_run() -> None`
- Produces: `last_token_stream_s = start_time + timestamps[-1]` for finite Sherpa results
- Preserves: `reset()` as same-stream reset for endpoint and warmup cleanup

- [ ] **Step 1: Write failing Zipformer lifecycle and timing tests**

Add this complete helper to `tests/test_zipformer.py`:

```python
def build_fake_zipformer(
    monkeypatch,
    tmp_path: Path,
    *,
    result: object | None = None,
    text_result: object = "legacy text",
    expose_get_result_all: bool = True,
) -> ZipformerRecognizer:
    write_reference_zipformer_files(tmp_path)

    class FakeStream:
        def accept_waveform(self, sample_rate, pcm):
            self.sample_rate = sample_rate
            self.pcm = pcm

    class BaseFakeRecognizer:
        def create_stream(self):
            return FakeStream()

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):
            raise AssertionError("decode should not run")

        def get_result(self, stream):
            return text_result

        def is_endpoint(self, stream):
            return False

        def reset(self, stream):
            self.reset_stream = stream

    class TimedFakeRecognizer(BaseFakeRecognizer):
        def get_result_all(self, stream):
            return result

    fake = TimedFakeRecognizer() if expose_get_result_all else BaseFakeRecognizer()
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(
            OnlineRecognizer=types.SimpleNamespace(from_transducer=lambda **kwargs: fake)
        ),
    )
    return ZipformerRecognizer(tmp_path, EndpointConfig())
```

Then add:

```python
@pytest.mark.asyncio
async def test_zipformer_reads_token_time_before_endpoint_reset(monkeypatch, tmp_path: Path):
    write_reference_zipformer_files(tmp_path)
    calls: list[str] = []

    class FakeStream:
        def accept_waveform(self, sample_rate, pcm):
            calls.append("accept")

    class FakeResult:
        text = "decoded text"
        start_time = 1.2
        timestamps = [0.04, 0.36]

    class FakeRecognizer:
        def create_stream(self):
            calls.append("create")
            return FakeStream()

        def is_ready(self, stream):
            return False

        def get_result_all(self, stream):
            calls.append("result")
            return FakeResult()

        def is_endpoint(self, stream):
            return True

        def reset(self, stream):
            calls.append("reset")

    fake = FakeRecognizer()
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(
            OnlineRecognizer=types.SimpleNamespace(from_transducer=lambda **kwargs: fake)
        ),
    )
    recognizer = ZipformerRecognizer(tmp_path, EndpointConfig())
    await recognizer.start_run()
    result = await recognizer.accept(
        AudioFrame(0, 0, np.asarray([0.1], dtype=np.float32), 100)
    )

    assert result is not None
    assert result.text == "decoded text"
    assert result.last_token_stream_s == pytest.approx(1.56)
    assert calls.count("create") == 2
    assert calls.index("result") < calls.index("reset")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start_time,timestamps",
    [(0.0, []), (float("nan"), [0.1]), (0.0, [float("inf")])],
)
async def test_zipformer_omits_invalid_token_time(
    monkeypatch, tmp_path: Path, start_time: float, timestamps: list[float]
):
    recognizer = build_fake_zipformer(
        monkeypatch,
        tmp_path,
        result=types.SimpleNamespace(
            text="text", start_time=start_time, timestamps=timestamps
        ),
    )
    result = await recognizer.accept(AudioFrame(0, 0, np.ones(1, dtype=np.float32), 100))
    assert result is not None
    assert result.last_token_stream_s is None


@pytest.mark.asyncio
async def test_zipformer_falls_back_when_get_result_all_is_unavailable(
    monkeypatch, tmp_path: Path
):
    recognizer = build_fake_zipformer(
        monkeypatch,
        tmp_path,
        text_result="legacy text",
        expose_get_result_all=False,
    )
    result = await recognizer.accept(AudioFrame(0, 0, np.ones(1, dtype=np.float32), 100))
    assert result is not None
    assert result.text == "legacy text"
    assert result.last_token_stream_s is None
```

- [ ] **Step 2: Run Zipformer tests and confirm failure**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest tests/test_zipformer.py -q
```

Expected: failures report missing `start_run()` and missing token timing.

- [ ] **Step 3: Implement new stream creation and compatible result extraction**

Add `import math`. Replace the online result section in `ZipformerRecognizer` with:

```python
    async def start_run(self) -> None:
        self._stream = self._recognizer.create_stream()

    async def accept(self, frame: AudioFrame) -> OnlineRecognition | None:
        if frame.is_finish:
            return None

        self._stream.accept_waveform(self._sample_rate, frame.pcm)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        get_result_all = getattr(self._recognizer, "get_result_all", None)
        if get_result_all is None:
            raw_result = self._recognizer.get_result(self._stream)
            text = raw_result.text if hasattr(raw_result, "text") else str(raw_result)
            last_token_stream_s = None
        else:
            raw_result = get_result_all(self._stream)
            text = raw_result.text if hasattr(raw_result, "text") else str(raw_result)
            timestamps = list(getattr(raw_result, "timestamps", ()))
            start_time = getattr(raw_result, "start_time", None)
            last_token_stream_s = None
            if timestamps and start_time is not None:
                candidate = float(start_time) + float(timestamps[-1])
                if math.isfinite(candidate):
                    last_token_stream_s = candidate

        is_endpoint = self._recognizer.is_endpoint(self._stream)
        if is_endpoint:
            self._recognizer.reset(self._stream)

        return OnlineRecognition(
            stream_id=frame.stream_id,
            sentence_id=0,
            revision=0,
            text=text.strip(),
            timestamp_ms=frame.timestamp_ms,
            is_endpoint=is_endpoint,
            last_token_stream_s=last_token_stream_s,
        )

    async def reset(self) -> None:
        self._recognizer.reset(self._stream)
```

Add a no-op `start_run()` to `lane_runtime._FakeOnlineRecognizer`:

```python
    async def start_run(self) -> None:
        return None
```

- [ ] **Step 4: Run Zipformer and lane-load tests**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest \
  tests/test_zipformer.py \
  tests/test_lane_runtime.py -k 'warmup or load or owner_thread' -q
```

Expected: selected tests pass; existing model/provider assertions remain unchanged.

- [ ] **Step 5: Commit the Zipformer adapter**

```bash
git add src/multi_stream_asr/asr/online/zipformer.py \
  src/multi_stream_asr/pipeline/lane_runtime.py \
  tests/test_zipformer.py
git commit -m "feat: expose zipformer token stream timing"
```

---

### Task 4: Endpoint 按时间切分与空 endpoint 尾部保留

**Files:**
- Modify: `src/multi_stream_asr/pipeline/pipeline.py:25-50,300-402`
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py:322-338`
- Modify: `src/multi_stream_asr/pipeline/manager.py:146-160`
- Modify: `tests/test_pipeline.py:248-335,595-641`
- Modify: `tests/test_lane_runtime.py`

**Interfaces:**
- Consumes: `EndpointConfig`, `OnlineRecognition.last_token_stream_s`, `SegmentCache.cut()`
- Produces: endpoint PCM `[cache_start, last_token_stream_s + margin)`
- Produces: empty endpoint retention window equal to `rule2_min_trailing_silence`
- Preserves: partial/pending/final revisions, sentence IDs, Qwen prompt, model output format

- [ ] **Step 1: Write failing endpoint PCM tests**

Add to `tests/test_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_token_timestamps_cut_adjacent_offline_segments_without_overlap():
    source = AcknowledgedAudioSource(
        [
            make_frame(frame_index=0, samples=[0, 1, 2, 3], timestamp_ms=400),
            make_frame(frame_index=1, samples=[4, 5, 6, 7], timestamp_ms=800),
            AudioFrame.finish(0, 2, 900),
        ]
    )
    online = AcknowledgingOnlineRecognizer(
        [
            make_online(
                text="one", timestamp_ms=400, is_endpoint=True,
                last_token_stream_s=0.2,
            ),
            make_online(
                text="two", timestamp_ms=800, is_endpoint=True,
                last_token_stream_s=0.6,
            ),
        ],
        source,
    )
    offline = FakeOfflineRecognizer()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
        endpoint=EndpointConfig(cut_margin_seconds=0.1),
    )

    await pipeline.run(source)

    assert online.start_run_count == 1
    np.testing.assert_array_equal(
        offline.requests[0][0], np.asarray([0, 1, 2], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        offline.requests[1][0], np.asarray([3, 4, 5, 6], dtype=np.float32)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("token_time", [None, float("nan"), -1.0])
async def test_invalid_token_time_falls_back_to_complete_cache(token_time: float | None):
    source = FakeAudioSource(
        [make_frame(frame_index=0, samples=[1, 2, 3], timestamp_ms=300), AudioFrame.finish(0, 1, 400)],
        pause_between_frames=True,
    )
    online = FakeOnlineRecognizer(
        [make_online(text="one", timestamp_ms=300, is_endpoint=True, last_token_stream_s=token_time)]
    )
    offline = FakeOfflineRecognizer()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
        endpoint=EndpointConfig(),
    )

    await pipeline.run(source)
    np.testing.assert_array_equal(
        offline.requests[0][0], np.asarray([1, 2, 3], dtype=np.float32)
    )
```

Replace `test_empty_endpoint_clears_cache_and_does_not_submit_offline` with:

```python
@pytest.mark.asyncio
async def test_empty_endpoint_retains_rule2_tail_for_next_sentence():
    source = AcknowledgedAudioSource(
        [
            make_frame(
                frame_index=0,
                samples=[0, 1, 2, 3, 4, 5, 6, 7],
                timestamp_ms=800,
            ),
            make_frame(frame_index=1, samples=[8, 9], timestamp_ms=1000),
            AudioFrame.finish(0, 2, 1100),
        ]
    )
    online = AcknowledgingOnlineRecognizer(
        [
            make_online(text="", timestamp_ms=800, is_endpoint=True),
            make_online(text="next", timestamp_ms=1000, is_endpoint=True),
        ],
        source,
    )
    offline = FakeOfflineRecognizer()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
        endpoint=EndpointConfig(
            rule2_min_trailing_silence=0.5,
            cut_margin_seconds=0.1,
        ),
    )

    await pipeline.run(source)

    assert len(offline.requests) == 1
    np.testing.assert_array_equal(
        offline.requests[0][0],
        np.asarray([3, 4, 5, 6, 7, 8, 9], dtype=np.float32),
    )
    assert offline.requests[0][1] == "next"
```

- [ ] **Step 2: Run endpoint tests and confirm old full-cache behavior**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest tests/test_pipeline.py \
  -k 'token_timestamps or invalid_token_time or empty_endpoint' -q
```

Expected: timestamp tests fail because full cache is submitted and the empty endpoint test fails because the cache is cleared.

- [ ] **Step 3: Add endpoint config and start-run lifecycle to LanePipeline**

Import `math` and `EndpointConfig`. Extend `LanePipeline.__init__`:

```python
        endpoint: EndpointConfig | None = None,
```

Store:

```python
        self.endpoint = endpoint or EndpointConfig()
```

At the start of `_consume_frames_thread()`, before the `while` loop, invoke `start_run()` on the owner thread while retaining compatibility with minimal test adapters:

```python
        start_run = getattr(self.online_recognizer, "start_run", None)
        if start_run is not None:
            self._run_async_in_thread(start_run())
```

Pass `endpoint=self.app_config.endpoint` from `LaneRuntime._run_started()` and `endpoint=self._config.endpoint` from `PipelineManager.start_recognition()`.

- [ ] **Step 4: Implement protected endpoint cutting**

Add these helpers to `LanePipeline`:

```python
    def _take_endpoint_audio(self, recognition: OnlineRecognition) -> np.ndarray:
        full = self.segment_cache.audio()
        token_time = recognition.last_token_stream_s
        if token_time is None or not math.isfinite(token_time):
            self.segment_cache.reset()
            return full

        cut_at = token_time + self.endpoint.cut_margin_seconds
        if cut_at <= self.segment_cache.start_time_s:
            self.segment_cache.reset()
            return full

        pcm = self.segment_cache.cut(cut_at)
        if pcm.size == 0 and full.size != 0:
            self.segment_cache.reset()
            return full
        return pcm

    def _advance_sentence(self) -> None:
        self._sentence_id += 1
        self._revision = 0
        self._current_online_text = ""
```

Change `_handle_online_result()` so the endpoint branch is:

```python
        endpoint_text = self._current_online_text
        if endpoint_text:
            pending_revision = self._revision + 1
            self._revision = pending_revision
            pcm = self._take_endpoint_audio(recognition)
            self._publish_event(
                sentence_id=self._sentence_id,
                revision=pending_revision,
                status=EventStatus.PENDING,
                text=endpoint_text,
                timestamp_ms=recognition.timestamp_ms,
            )
            self._record_model_output(
                model="zipformer",
                sentence_id=self._sentence_id,
                status="endpoint",
                text=endpoint_text,
            )
            if pcm.size:
                self._schedule_offline_threadsafe(
                    sentence_id=self._sentence_id,
                    revision=pending_revision + 1,
                    prompt=endpoint_text,
                    pcm=pcm,
                    timestamp_ms=recognition.timestamp_ms,
                )
                self._increment_metrics(offline_requests=1)
        else:
            self.segment_cache.cut(
                self.segment_cache.end_time_s
                - self.endpoint.rule2_min_trailing_silence
            )

        self._advance_sentence()
```

Do not call `segment_cache.reset()` after a successful timestamp cut; its retained tail is the next sentence prefix.

- [ ] **Step 5: Update owner-thread lifecycle test**

In `test_real_lane_warms_on_owner_thread_before_lane_loaded`, add a `start_run_threads` list and implement this method on its `FakeOnlineRecognizer`:

```python
        async def start_run(self) -> None:
            owner_threads.append(threading.get_ident())
            start_run_threads.append(threading.get_ident())
```

Replace its terminal `stop` command with a real run and arrange shutdown after lane completion:

```python
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
```

After `runtime.run()`, assert:

```python
    assert len(start_run_threads) == 1
    assert start_run_threads[0] == owner_threads[0]
    assert len(set(owner_threads)) == 1
```

This proves new stream creation stays on the Zipformer owner thread.

- [ ] **Step 6: Run endpoint and runtime tests**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest \
  tests/test_pipeline.py \
  tests/test_lane_runtime.py -q
```

Expected: all selected tests pass; existing sentence ID and model-output assertions remain valid.

- [ ] **Step 7: Commit endpoint cutting**

```bash
git add src/multi_stream_asr/pipeline/pipeline.py \
  src/multi_stream_asr/pipeline/lane_runtime.py \
  src/multi_stream_asr/pipeline/manager.py \
  tests/test_pipeline.py tests/test_lane_runtime.py
git commit -m "feat: cut endpoint audio at token timestamps"
```

---

### Task 5: 自然 finish 冲刷与手动停止保持快速

**Files:**
- Modify: `src/multi_stream_asr/pipeline/pipeline.py:310-316,513-526`
- Modify: `tests/test_pipeline.py:210-245,540-592`

**Interfaces:**
- Consumes: current online text and remaining `SegmentCache.audio()`
- Produces: natural finish `pending -> final -> LANE_FINISHED`
- Preserves: manual stop does not create a new offline request

- [ ] **Step 1: Write failing natural-finish and manual-stop tests**

Replace the existing non-endpoint finish expectation with:

```python
@pytest.mark.asyncio
async def test_natural_finish_flushes_non_empty_online_remainder():
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[1, 2], timestamp_ms=100),
            make_frame(frame_index=1, samples=[3, 4], timestamp_ms=200),
            AudioFrame.finish(0, 2, 300),
        ],
        pause_between_frames=True,
    )
    online = FakeOnlineRecognizer(
        [
            make_online(text="hel", timestamp_ms=100),
            make_online(text="hello", timestamp_ms=200),
        ]
    )
    offline = FakeOfflineRecognizer(text="offline hello")
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
    )

    await pipeline.run(source)
    events = await next_events(bus, 5)

    assert [(event.status, event.text) for event in events] == [
        (EventStatus.PARTIAL, "hel"),
        (EventStatus.PARTIAL, "hello"),
        (EventStatus.PENDING, "hello"),
        (EventStatus.FINAL, "offline hello"),
        (EventStatus.LANE_FINISHED, ""),
    ]
    np.testing.assert_array_equal(
        offline.requests[0][0], np.asarray([1, 2, 3, 4], dtype=np.float32)
    )
```

Add a stoppable source test:

```python
@pytest.mark.asyncio
async def test_manual_stop_discards_unendpointed_remainder():
    source = StoppableBlockedSource()
    online = FakeOnlineRecognizer(
        [make_online(text="unfinished", timestamp_ms=100)]
    )
    offline = FakeOfflineRecognizer()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    run_task = asyncio.create_task(pipeline.run(source))
    async with asyncio.timeout(0.5):
        while not online.accepted_frames:
            await asyncio.sleep(0.001)
    pipeline.request_stop()
    await source.stop()
    await asyncio.wait_for(run_task, timeout=0.5)

    assert offline.requests == []
```

- [ ] **Step 2: Run finish tests and confirm old behavior**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest tests/test_pipeline.py \
  -k 'natural_finish or manual_stop or finish_resets' -q
```

Expected: the natural-finish test fails because no pending/offline request is created; manual-stop remains passing.

- [ ] **Step 3: Flush the final sentence on the event loop before draining offline work**

Keep the consumer finish branch resetting the recognizer on its owner thread, then call `_finish_after_offline()` as it does now. At the beginning of `_finish_after_offline()`, add:

```python
        endpoint_text = self._current_online_text
        if endpoint_text:
            pcm = self.segment_cache.audio()
            self.segment_cache.reset()
            pending_revision = self._revision + 1
            self._revision = pending_revision
            self._publish_event(
                sentence_id=self._sentence_id,
                revision=pending_revision,
                status=EventStatus.PENDING,
                text=endpoint_text,
                timestamp_ms=timestamp_ms,
            )
            self._record_model_output(
                model="zipformer",
                sentence_id=self._sentence_id,
                status="endpoint",
                text=endpoint_text,
            )
            if pcm.size:
                self._schedule_offline(
                    sentence_id=self._sentence_id,
                    revision=pending_revision + 1,
                    prompt=endpoint_text,
                    pcm=pcm,
                    timestamp_ms=timestamp_ms,
                )
                self._increment_metrics(offline_requests=1)
            self._advance_sentence()
        else:
            self.segment_cache.reset()
```

This code runs on the lane event loop because `_finish_after_offline()` is submitted with `asyncio.run_coroutine_threadsafe`; call `_schedule_offline()` directly so the task enters `_offline_tasks` before the subsequent gather.

- [ ] **Step 4: Run all pipeline tests**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest tests/test_pipeline.py -q
```

Expected: all pipeline tests pass, including event order and offline drain behavior.

- [ ] **Step 5: Commit finish semantics**

```bash
git add src/multi_stream_asr/pipeline/pipeline.py tests/test_pipeline.py
git commit -m "feat: flush final segment on natural finish"
```

---

### Task 6: 全量回归与 K3 真实双运行验收

**Files:**
- Verify: all tracked source and tests changed by Tasks 1-5
- Inspect: `runs/process-*/config.json`
- Inspect: `runs/process-*/metrics.json`
- Inspect: `runs/process-*/model_outputs/lane*.jsonl`
- Inspect: `runs/process-*/results/lane*.txt`

**Interfaces:**
- Verifies: two runs reuse loaded models but receive independent Sherpa streams and cache clocks
- Verifies: Zipformer and Qwen sentence IDs align per lane
- Verifies: no missing first characters or duplicated adjacent sentence prefixes

- [ ] **Step 1: Run the complete local suite**

Run:

```bash
PYTHONPATH=. .venv313/bin/python -m pytest -q
```

Expected: all tests pass. Record the exact pass count in the implementation handoff.

- [ ] **Step 2: Review the complete diff**

Run:

```bash
git diff 1195953..HEAD --check
git diff 1195953..HEAD --stat
git status --short
```

Expected: `git diff --check` exits 0; only planned tracked files and the pre-existing untracked files are present. Do not add `TODO`, `docs/endpoint-segment-alignment.html`, `src/multi_stream_asr.egg-info/`, or `uv.lock`.

- [ ] **Step 3: Sync only this feature's tracked changes to K3**

Run locally:

```bash
git diff --name-only 1195953..HEAD > /tmp/token-timestamp-segmentation-files.txt
rsync -az --files-from=/tmp/token-timestamp-segmentation-files.txt ./ \
  K3:~/Workspace/projects/multi-stream-asr/
```

Expected: source, tests, spec, and plan are updated without deleting remote models, `.venv313`, runs, or untracked debug files.

- [ ] **Step 4: Run K3 tests under the required Python environment**

Run:

```bash
ssh K3 'cd ~/Workspace/projects/multi-stream-asr && \
  source .venv313/bin/activate && \
  export UV_EXTRA_INDEX_URL="https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple" && \
  export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple && \
  export PIP_INDEX_URL=https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple && \
  export PIP_EXTRA_INDEX_URL="https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple https://mirrors.aliyun.com/pypi/simple" && \
  python --version && PYTHONPATH=. python -m pytest -q'
```

Expected: Python reports 3.13.x and the complete test suite passes.

- [ ] **Step 5: Run two recognitions without unloading models between runs**

Run on K3 from the activated `.venv313`:

```bash
ssh K3 'cd ~/Workspace/projects/multi-stream-asr && \
  source .venv313/bin/activate && \
  PYTHONPATH=. python - <<'"'"'PY'"'"'
import asyncio
from pathlib import Path

from multi_stream_asr.config import AppConfig
from multi_stream_asr.log.recorder import RunRecorder
from multi_stream_asr.pipeline.process_manager import ProcessPipelineManager
from multi_stream_asr.web.event_bus import FrontendEventBus


async def main() -> None:
    config = AppConfig(lanes=4, input_dir=Path.home() / "Music/wav16k")
    manager = ProcessPipelineManager(config=config, event_bus=FrontendEventBus())
    recorder = RunRecorder()
    run_dirs: list[Path] = []
    await manager.load_models()
    try:
        for repetition in (1, 2):
            await manager.start_recognition()
            await manager.wait_for_completion()
            assert manager.run_dir is not None
            recorder.save_process_run(
                manager.run_dir,
                config=config,
                lane_summaries=manager.lane_summaries,
            )
            run_dirs.append(manager.run_dir)
            print(f"repeat={repetition} run_dir={manager.run_dir}")
    finally:
        await manager.unload_models()
    Path("/tmp/token-timestamp-run-dirs.txt").write_text(
        "\n".join(str(path) for path in run_dirs) + "\n",
        encoding="utf-8",
    )


asyncio.run(main())
PY'
```

Expected: both runs complete in the same loaded manager session and print two distinct run directories.

- [ ] **Step 6: Validate artifacts and metrics for both runs**

Run this K3 check; it reads both exact paths written by Step 5:

```bash
ssh K3 'cd ~/Workspace/projects/multi-stream-asr && source .venv313/bin/activate && \
  PYTHONPATH=. python - <<'"'"'PY'"'"'
import json
import math
from pathlib import Path

run_dirs = [
    Path(line)
    for line in Path("/tmp/token-timestamp-run-dirs.txt").read_text(encoding="utf-8").splitlines()
    if line
]
assert len(run_dirs) == 2
for run in run_dirs:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert config["endpoint"]["cut_margin_seconds"] == 0.1
    for key in ("average_ms", "max_ms"):
        assert math.isfinite(float(metrics["online_frame_latency"][key]))
    for key in ("average", "max"):
        assert math.isfinite(float(metrics["offline_rtf"][key]))
    for lane in range(4):
        records = [
            json.loads(line)
            for line in (run / "model_outputs" / f"lane{lane}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        zip_ids = {
            row["sentence_id"] for row in records if row["model"] == "zipformer"
        }
        qwen_ids = {
            row["sentence_id"] for row in records if row["model"] == "qwen3-asr"
        }
        assert zip_ids == qwen_ids, (lane, zip_ids, qwen_ids)
        assert (run / "results" / f"lane{lane}.txt").is_file()
    drops = [lane["dropped_frames"] for lane in metrics["lanes"]]
    assert sum(drops) < 10, drops
    print(json.dumps({
        "run": str(run),
        "online": metrics["online_frame_latency"],
        "offline": metrics["offline_rtf"],
        "drops": drops,
    }, ensure_ascii=False, indent=2))
PY'
```

Expected for each run: all four lane ID-set assertions pass, result files exist, total dropped frames remain below 10, and metrics print finite average/max values. Compare the two runs' sentence counts and transcripts side by side; the second run must not lose first characters or shift all cuts to cache tails.

- [ ] **Step 7: Compare performance with the latest four-lane baseline**

Use the baseline in `docs/reports/2026-07-12-qwen3-lane-matrix-performance.md`:

```text
4 lanes, Qwen3-ASR 0.6B q40:
online average 41.627ms
online max 208.492ms
offline RTF average 0.448518
dropped frames total average 0.667
```

Expected: segmentation does not create a systematic online latency increase, total drops stay below 10 per run, and offline RTF is of the same order as the baseline. Segment durations change, so treat small RTF movement as input-boundary variance; investigate only a repeatable regression.

- [ ] **Step 8: Final repository verification**

Run locally:

```bash
git status --short --branch
git log --oneline 1195953..HEAD
```

Expected: Tasks 1-5 appear as focused commits, no implementation changes remain unstaged, and pre-existing untracked files remain untouched.
