# ASR Model Call Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align real model invocation with the known-good reference implementation, force Zipformer int8 model files, and add model-output diagnostics so 4-lane headless runs prove content correctness.

**Architecture:** Keep the existing lane pipeline and event semantics. Fix model adapters at their boundaries, add a small model-output recorder shared by pipeline/headless/recorder, and validate with unit tests plus a real K3 headless run.

**Tech Stack:** Python 3.13, pytest, numpy, httpx, sherpa-onnx, Qwen3-ASR via `llama-server`, FastAPI event pipeline, JSONL diagnostics.

## Global Constraints

- Remote development, validation, and inference must use Python 3.13 and the project-local `.venv313`.
- Remote dependency installation must happen only after activating `.venv313`.
- Remote dependency installation must export the Spacemit and Aliyun package index variables first.
- `sherpa-onnx` is available from `https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple`.
- Real validation target remains 4 lanes with headless mode.
- Target metrics remain online frame latency about 50 ms and whole-segment offline RTF about 0.4; correctness is evaluated before performance tuning.
- Do not change frontend `pending` or `final` replacement semantics.
- Do not change endpointing rules or segment cache policy.
- Do not add Python-side explicit CPU binding.
- Zipformer must use `encoder-epoch-99-avg-1.int8.onnx`, `decoder-epoch-99-avg-1.onnx`, `joiner-epoch-99-avg-1.int8.onnx`, and `tokens.txt`.

---

## File Structure

- Modify `src/multi_stream_asr/asr/offline/base.py`: add optional raw response storage to `OfflineRecognitionResult`.
- Modify `src/multi_stream_asr/asr/offline/qwen3asr.py`: build the reference Qwen request body, encode WAV bytes like the reference project, parse responses, and return raw response JSON.
- Modify `tests/test_qwen3asr.py`: lock Qwen WAV encoding, payload shape, cleanup, parsing, and raw response behavior.
- Modify `src/multi_stream_asr/asr/online/zipformer.py`: require exact int8 model filenames and extract `.text` from sherpa result objects.
- Modify `tests/test_zipformer.py`: lock exact model filenames and result-object extraction.
- Create `src/multi_stream_asr/log/model_output.py`: small model-output recorder that prints concise diagnostics and stores JSONL-ready records.
- Create `tests/test_model_output.py`: cover printed diagnostic format and JSON serialization shape.
- Modify `src/multi_stream_asr/pipeline/pipeline.py`: record Zipformer endpoint output and Qwen final output without changing ASR events.
- Modify `src/multi_stream_asr/pipeline/manager.py`: pass the optional recorder into each lane pipeline.
- Modify `src/multi_stream_asr/headless/runner.py`: own a recorder per run and pass records to `RunRecorder`.
- Modify `src/multi_stream_asr/log/recorder.py`: write compact per-lane `model_outputs/lane<N>.jsonl` and per-lane `results/lane<N>.txt`.
- Modify `tests/test_pipeline.py`, `tests/test_manager_headless.py`, and `tests/test_buffers_metrics.py`: cover diagnostic wiring and persistence.

---

### Task 1: Align Qwen3-ASR Request And Response Handling

**Files:**
- Modify: `src/multi_stream_asr/asr/offline/base.py`
- Modify: `src/multi_stream_asr/asr/offline/qwen3asr.py`
- Test: `tests/test_qwen3asr.py`

**Interfaces:**
- Consumes: `LaneConfig`, `OfflineServerConfig`, `np.ndarray` PCM.
- Produces: `build_chat_request(pcm: np.ndarray, max_tokens: int = 256) -> dict[str, object]`, `build_slot_erase_url(base_url: str, slot_id: int = 0) -> str`, and `OfflineRecognitionResult.raw_response: dict[str, object] | None`.

- [ ] **Step 1: Write failing Qwen payload tests**

Add these imports and tests to `tests/test_qwen3asr.py`:

```python
from multi_stream_asr.asr.offline.qwen3asr import (
    Qwen3AsrClient,
    build_chat_request,
    build_slot_erase_url,
    clean_qwen_text,
    encode_wav_base64,
)


def test_build_chat_request_uses_reference_input_audio_payload():
    pcm = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)

    payload = build_chat_request(pcm)

    assert payload["id_slot"] == 0
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 256
    assert payload["messages"][0] == {
        "role": "system",
        "content": "You are a speech recognition model.",
    }
    user_message = payload["messages"][1]
    assert user_message["role"] == "user"
    assert user_message["content"][0] == {
        "type": "input_audio",
        "input_audio": {
            "data": encode_wav_base64(pcm),
            "format": "wav",
        },
    }
    assert user_message["content"][1] == {"type": "text", "text": ""}
    assert "audio" not in payload


def test_build_slot_erase_url_uses_slot_zero_by_default():
    assert build_slot_erase_url("http://127.0.0.1:8063") == "http://127.0.0.1:8063/slots/0?action=erase"
```

Update `FakeAsyncClient.post()` in `tests/test_qwen3asr.py` so cleanup calls can omit a JSON body:

```python
    async def post(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,
        timeout: float,
    ) -> FakeResponse:
        self.calls.append(("POST", url, json or {}, timeout))
        if not self.post_responses:
            raise AssertionError("unexpected post")
        return self.post_responses.pop(0)
```

Update `test_recognize_sends_chat_completions_payload_and_erases_slot()` assertions:

```python
    assert request_call[2] == build_chat_request(pcm)
    assert cleanup_call[1] == "http://127.0.0.1:9002/slots/0?action=erase"
    assert cleanup_call[3] == 5.678
    assert result.raw_response == {"choices": [{"message": {"content": "<asr_text> hello world </asr_text>"}}]}
```

- [ ] **Step 2: Run Qwen tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_qwen3asr.py -v
```

Expected: FAIL because `build_chat_request` and `build_slot_erase_url` are missing and current requests still contain top-level `audio`.

- [ ] **Step 3: Implement Qwen request helpers and raw response storage**

In `src/multi_stream_asr/asr/offline/base.py`, change `OfflineRecognitionResult` to:

```python
from typing import Any


@dataclass(frozen=True)
class OfflineRecognitionResult:
    text: str
    elapsed_ms: float
    audio_seconds: float
    rtf: float
    raw_response: dict[str, Any] | None = None
```

In `src/multi_stream_asr/asr/offline/qwen3asr.py`, add constants and helpers:

```python
from typing import Any


QWEN_SLOT_ID = 0
QWEN_SYSTEM_PROMPT = "You are a speech recognition model."
QWEN_MAX_TOKENS = 256


def encode_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    array = np.asarray(pcm, dtype=np.float32)
    clipped = np.clip(array, -1.0, 0.999969)
    pcm16 = (clipped * 32768.0).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def encode_wav_base64(pcm: np.ndarray, sample_rate: int = 16000) -> str:
    return base64.b64encode(encode_wav_bytes(pcm, sample_rate=sample_rate)).decode("ascii")


def build_chat_request(pcm: np.ndarray, max_tokens: int = QWEN_MAX_TOKENS) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encode_wav_base64(pcm),
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": ""},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "id_slot": QWEN_SLOT_ID,
    }


def build_slot_erase_url(base_url: str, slot_id: int = QWEN_SLOT_ID) -> str:
    return base_url.rstrip("/") + f"/slots/{slot_id}?action=erase"
```

Update `recognize()` to use the helper payload and preserve raw response:

```python
        payload = build_chat_request(pcm)
        raw_response: dict[str, Any] | None = None

        async with httpx.AsyncClient() as client:
            try:
                request_sent = True
                response = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    timeout=request_timeout_seconds,
                )
                response.raise_for_status()
                raw_response = response.json()
                text = clean_qwen_text(raw_response)
            finally:
                if request_sent:
                    try:
                        cleanup_response = await client.post(
                            build_slot_erase_url(self._base_url),
                            json={},
                            timeout=slot_cleanup_timeout_seconds,
                        )
                        cleanup_response.raise_for_status()
                    except Exception:
                        logger.warning("slot cleanup failed for lane offline request", exc_info=True)
```

Return:

```python
        return OfflineRecognitionResult(
            text=text,
            elapsed_ms=elapsed_ms,
            audio_seconds=audio_seconds,
            rtf=rtf,
            raw_response=raw_response,
        )
```

- [ ] **Step 4: Run Qwen tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_qwen3asr.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Qwen fix**

Run:

```bash
git add src/multi_stream_asr/asr/offline/base.py src/multi_stream_asr/asr/offline/qwen3asr.py tests/test_qwen3asr.py
git commit -m "fix: align qwen asr request payload"
```

---

### Task 2: Force Zipformer Int8 Model Files And Result Text Extraction

**Files:**
- Modify: `src/multi_stream_asr/asr/online/zipformer.py`
- Test: `tests/test_zipformer.py`

**Interfaces:**
- Consumes: model directory path and `EndpointConfig`.
- Produces: exact model file selection and text extraction from either `result.text` or string-like sherpa results.

- [ ] **Step 1: Write failing Zipformer tests**

Add imports to `tests/test_zipformer.py`:

```python
import numpy as np
import pytest

from multi_stream_asr.models import AudioFrame
```

Replace the current generic model-file setup in `tests/test_zipformer.py` with exact filenames:

```python
def write_reference_zipformer_files(model_dir: Path) -> None:
    for name in (
        "tokens.txt",
        "encoder-epoch-99-avg-1.int8.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.int8.onnx",
    ):
        (model_dir / name).write_text("stub", encoding="utf-8")
```

Update `test_zipformer_recognizer_constructs_sherpa_with_one_thread()`:

```python
    write_reference_zipformer_files(tmp_path)
```

Expected assertions:

```python
    assert captured["encoder"] == str(tmp_path / "encoder-epoch-99-avg-1.int8.onnx")
    assert captured["decoder"] == str(tmp_path / "decoder-epoch-99-avg-1.onnx")
    assert captured["joiner"] == str(tmp_path / "joiner-epoch-99-avg-1.int8.onnx")
```

Add a missing-file test:

```python
def test_zipformer_recognizer_fails_when_reference_int8_files_are_missing(monkeypatch, tmp_path: Path):
    (tmp_path / "tokens.txt").write_text("stub", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(
            OnlineRecognizer=types.SimpleNamespace(from_transducer=lambda **kwargs: None),
        ),
    )

    with pytest.raises(FileNotFoundError, match="encoder-epoch-99-avg-1.int8.onnx"):
        ZipformerRecognizer(tmp_path, EndpointConfig())
```

Add a result-object extraction test:

```python
@pytest.mark.asyncio
async def test_zipformer_accept_uses_text_attribute_from_sherpa_result(monkeypatch, tmp_path: Path):
    write_reference_zipformer_files(tmp_path)

    class FakeStream:
        def accept_waveform(self, sample_rate, pcm):
            self.sample_rate = sample_rate
            self.pcm = pcm

    class FakeResult:
        text = "decoded text"

    class FakeRecognizer:
        def create_stream(self):
            return FakeStream()

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):
            raise AssertionError("decode should not run")

        def get_result(self, stream):
            return FakeResult()

        def is_endpoint(self, stream):
            return True

        def reset(self, stream):
            self.reset_stream = stream

    class FakeOnlineRecognizerFactory:
        @staticmethod
        def from_transducer(**kwargs):
            return FakeRecognizer()

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(OnlineRecognizer=FakeOnlineRecognizerFactory),
    )
    recognizer = ZipformerRecognizer(tmp_path, EndpointConfig())

    result = await recognizer.accept(
        AudioFrame(stream_id=1, frame_index=0, pcm=np.asarray([0.1], dtype=np.float32), timestamp_ms=100)
    )

    assert result is not None
    assert result.text == "decoded text"
    assert result.is_endpoint is True
```

- [ ] **Step 2: Run Zipformer tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_zipformer.py -v
```

Expected: FAIL because current code accepts broad `encoder*.onnx` and may stringify sherpa result objects.

- [ ] **Step 3: Implement exact int8 selection and text extraction**

In `src/multi_stream_asr/asr/online/zipformer.py`, replace broad model lookup with exact files:

```python
        tokens = self._require_file(model_path / "tokens.txt")
        encoder = self._require_file(model_path / "encoder-epoch-99-avg-1.int8.onnx")
        decoder = self._require_file(model_path / "decoder-epoch-99-avg-1.onnx")
        joiner = self._require_file(model_path / "joiner-epoch-99-avg-1.int8.onnx")
```

Replace result extraction in `accept()`:

```python
        raw_result = self._recognizer.get_result(self._stream)
        text = raw_result.text if hasattr(raw_result, "text") else str(raw_result)
```

Remove `_find_model_file()` if it is no longer used.

- [ ] **Step 4: Run Zipformer tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_zipformer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Zipformer fix**

Run:

```bash
git add src/multi_stream_asr/asr/online/zipformer.py tests/test_zipformer.py
git commit -m "fix: use reference zipformer int8 model files"
```

---

### Task 3: Add Model Output Recorder

**Files:**
- Create: `src/multi_stream_asr/log/model_output.py`
- Create: `tests/test_model_output.py`

**Interfaces:**
- Produces: `ModelOutputRecord` dataclass and `ModelOutputRecorder.record(record: ModelOutputRecord) -> None`.
- Produces: `ModelOutputRecorder.as_dicts() -> list[dict[str, object]]` for JSONL persistence.

- [ ] **Step 1: Write failing recorder tests**

Create `tests/test_model_output.py`:

```python
from __future__ import annotations

import io

from multi_stream_asr.log.model_output import ModelOutputRecord, ModelOutputRecorder


def test_model_output_recorder_prints_concise_line_and_stores_record():
    stream = io.StringIO()
    recorder = ModelOutputRecorder(stream=stream)
    record = ModelOutputRecord(
        model="qwen3-asr",
        stream_id=2,
        sentence_id=5,
        status="final",
        text="这是模型识别出来的一段文字",
        audio_seconds=1.25,
        elapsed_ms=500.0,
        rtf=0.4,
        raw_response={"text": "<asr_text>这是模型识别出来的一段文字</asr_text>"},
    )

    recorder.record(record)

    printed = stream.getvalue()
    assert "[model-output]" in printed
    assert "model=qwen3-asr" in printed
    assert "lane=2" in printed
    assert "sentence=5" in printed
    assert "status=final" in printed
    assert "text_len=13" in printed
    assert "rtf=0.400" in printed
    assert "这是模型识别出来的一段文字" in printed

    stored = recorder.as_dicts()
    assert stored == [
        {
            "model": "qwen3-asr",
            "stream_id": 2,
            "sentence_id": 5,
            "status": "final",
            "text": "这是模型识别出来的一段文字",
            "text_preview": "这是模型识别出来的一段文字",
            "audio_seconds": 1.25,
            "elapsed_ms": 500.0,
            "rtf": 0.4,
            "raw_response": {"text": "<asr_text>这是模型识别出来的一段文字</asr_text>"},
        }
    ]


def test_model_output_recorder_can_store_without_printing():
    stream = io.StringIO()
    recorder = ModelOutputRecorder(stream=stream, emit_stdout=False)

    recorder.record(
        ModelOutputRecord(
            model="zipformer",
            stream_id=1,
            sentence_id=0,
            status="endpoint",
            text="online endpoint",
        )
    )

    assert stream.getvalue() == ""
    assert recorder.as_dicts()[0]["text"] == "online endpoint"
```

- [ ] **Step 2: Run recorder tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_model_output.py -v
```

Expected: FAIL because `multi_stream_asr.log.model_output` does not exist.

- [ ] **Step 3: Implement recorder**

Create `src/multi_stream_asr/log/model_output.py`:

```python
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any, TextIO


@dataclass(frozen=True)
class ModelOutputRecord:
    model: str
    stream_id: int | None
    sentence_id: int | None
    status: str
    text: str
    audio_seconds: float | None = None
    elapsed_ms: float | None = None
    rtf: float | None = None
    raw_response: dict[str, Any] | None = None

    @property
    def text_preview(self) -> str:
        return self.text[:120]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["text_preview"] = self.text_preview
        return payload


class ModelOutputRecorder:
    def __init__(self, *, stream: TextIO | None = None, emit_stdout: bool = True) -> None:
        self._stream = stream or sys.stdout
        self._emit_stdout = emit_stdout
        self._records: list[ModelOutputRecord] = []

    def record(self, record: ModelOutputRecord) -> None:
        self._records.append(record)
        if not self._emit_stdout:
            return

        audio = "-" if record.audio_seconds is None else f"{record.audio_seconds:.3f}"
        elapsed = "-" if record.elapsed_ms is None else f"{record.elapsed_ms:.1f}"
        rtf = "-" if record.rtf is None else f"{record.rtf:.3f}"
        print(
            "[model-output] "
            f"model={record.model} "
            f"lane={record.stream_id} "
            f"sentence={record.sentence_id} "
            f"status={record.status} "
            f"text_len={len(record.text)} "
            f"audio_s={audio} "
            f"elapsed_ms={elapsed} "
            f"rtf={rtf} "
            f"text={record.text_preview}",
            file=self._stream,
            flush=True,
        )

    def as_dicts(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self._records]
```

- [ ] **Step 4: Run recorder tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_model_output.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit recorder**

Run:

```bash
git add src/multi_stream_asr/log/model_output.py tests/test_model_output.py
git commit -m "feat: add model output recorder"
```

---

### Task 4: Wire Diagnostics Into Pipeline, Headless, And Persistence

**Files:**
- Modify: `src/multi_stream_asr/pipeline/pipeline.py`
- Modify: `src/multi_stream_asr/pipeline/manager.py`
- Modify: `src/multi_stream_asr/headless/runner.py`
- Modify: `src/multi_stream_asr/log/recorder.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_manager_headless.py`
- Test: `tests/test_buffers_metrics.py`

**Interfaces:**
- Consumes: `ModelOutputRecorder.record(ModelOutputRecord)`.
- Produces: `model_outputs/lane<N>.jsonl` and `results/lane<N>.txt` in headless run directories.

- [ ] **Step 1: Write failing pipeline diagnostics test**

Add to `tests/test_pipeline.py`:

```python
from multi_stream_asr.asr.offline.base import OfflineRecognitionResult
from multi_stream_asr.log.model_output import ModelOutputRecorder


class RawResponseOfflineRecognizer(FakeOfflineRecognizer):
    async def recognize(self, pcm, prompt):
        self.requests.append((pcm.copy(), prompt))
        return OfflineRecognitionResult(
            text="offline final",
            elapsed_ms=400.0,
            audio_seconds=1.0,
            rtf=0.4,
            raw_response={"text": "<asr_text>offline final</asr_text>"},
        )


@pytest.mark.asyncio
async def test_pipeline_records_zipformer_endpoint_and_qwen_final_model_outputs():
    recorder = ModelOutputRecorder(emit_stdout=False)
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [
                make_online(text="online endpoint", timestamp_ms=100, is_endpoint=True),
            ]
        ),
        offline_recognizer=RawResponseOfflineRecognizer(),
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        model_output_recorder=recorder,
    )
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            AudioFrame.finish(stream_id=0, frame_index=1, timestamp_ms=200),
        ]
    )

    await pipeline.run(source)

    records = recorder.as_dicts()
    assert records[0]["model"] == "zipformer"
    assert records[0]["stream_id"] == 0
    assert records[0]["sentence_id"] == 0
    assert records[0]["status"] == "endpoint"
    assert records[0]["text"] == "online endpoint"
    assert records[1]["model"] == "qwen3-asr"
    assert records[1]["status"] == "final"
    assert records[1]["text"] == "offline final"
    assert records[1]["audio_seconds"] == 1.0
    assert records[1]["elapsed_ms"] == 400.0
    assert records[1]["rtf"] == 0.4
    assert records[1]["raw_response"] == {"text": "<asr_text>offline final</asr_text>"}
```

- [ ] **Step 2: Write failing recorder persistence test**

Add to `tests/test_buffers_metrics.py`:

```python
def test_run_recorder_writes_per_lane_compact_model_outputs_jsonl(tmp_path: Path):
    recorder = RunRecorder()
    config = AppConfig(output_dir=tmp_path)

    run_dir = recorder.save(
        tmp_path / "run",
        config=config,
        model_outputs=[
            {
                "model": "qwen3-asr",
                "stream_id": 0,
                "sentence_id": 1,
                "text": "recognized",
                "audio_seconds": 1.0,
                "rtf": 0.4,
            }
        ],
    )

    assert not (run_dir / "model_outputs.jsonl").exists()
    lines = (run_dir / "model_outputs" / "lane0.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "model": "qwen3-asr",
        "stream_id": 0,
        "sentence_id": 1,
        "audio_seconds": 1.0,
        "rtf": 0.4,
        "text": "recognized",
    }
```

- [ ] **Step 3: Run diagnostics tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_pipeline.py::test_pipeline_records_zipformer_endpoint_and_qwen_final_model_outputs tests/test_buffers_metrics.py::test_run_recorder_writes_per_lane_compact_model_outputs_jsonl -v
```

Expected: FAIL because pipeline and recorder do not accept model-output records yet.

- [ ] **Step 4: Implement pipeline diagnostic recording**

In `src/multi_stream_asr/pipeline/pipeline.py`, import:

```python
from multi_stream_asr.log.model_output import ModelOutputRecord, ModelOutputRecorder
```

Add an optional constructor parameter:

```python
        model_output_recorder: ModelOutputRecorder | None = None,
```

Store it:

```python
        self.model_output_recorder = model_output_recorder
```

Add a helper:

```python
    def _record_model_output(
        self,
        *,
        model: str,
        sentence_id: int,
        status: str,
        text: str,
        audio_seconds: float | None = None,
        elapsed_ms: float | None = None,
        rtf: float | None = None,
        raw_response: dict[str, object] | None = None,
    ) -> None:
        if self.model_output_recorder is None:
            return
        self.model_output_recorder.record(
            ModelOutputRecord(
                model=model,
                stream_id=self.stream_id,
                sentence_id=sentence_id,
                status=status,
                text=text,
                audio_seconds=audio_seconds,
                elapsed_ms=elapsed_ms,
                rtf=rtf,
                raw_response=raw_response,
            )
        )
```

In `_handle_online_result()`, immediately before scheduling offline for non-empty endpoint text:

```python
            self._record_model_output(
                model="zipformer",
                sentence_id=self._sentence_id,
                status="endpoint",
                text=endpoint_text,
            )
```

In `_schedule_offline().run_offline()`, after `self.rtf_tracker.add(...)` and before publishing final:

```python
            self._record_model_output(
                model="qwen3-asr",
                sentence_id=sentence_id,
                status="final",
                text=result.text,
                audio_seconds=result.audio_seconds,
                elapsed_ms=result.elapsed_ms,
                rtf=result.rtf,
                raw_response=result.raw_response,
            )
```

- [ ] **Step 5: Implement manager and headless wiring**

In `src/multi_stream_asr/pipeline/manager.py`, import:

```python
from multi_stream_asr.log.model_output import ModelOutputRecorder
```

Add constructor parameter and field:

```python
        model_output_recorder: ModelOutputRecorder | None = None,
```

```python
        self._model_output_recorder = model_output_recorder
```

Pass it into each lane pipeline:

```python
                    model_output_recorder=self._model_output_recorder,
```

In `src/multi_stream_asr/headless/runner.py`, import:

```python
from multi_stream_asr.log.model_output import ModelOutputRecorder
```

Add constructor parameter and field:

```python
        model_output_recorder: ModelOutputRecorder | None = None,
```

```python
        self._model_output_recorder = model_output_recorder or ModelOutputRecorder(emit_stdout=not fake)
```

Pass the recorder into each `PipelineManager(...)` creation:

```python
                model_output_recorder=self._model_output_recorder,
```

and:

```python
            model_output_recorder=self._model_output_recorder,
```

In `_save_results()`, pass:

```python
            model_outputs=self._model_output_recorder.as_dicts(),
```

- [ ] **Step 6: Implement JSONL persistence**

In `src/multi_stream_asr/log/recorder.py`, extend `save()`:

```python
        model_outputs: Any = None,
```

After writing metrics, write per-lane result and model-output files:

```python
        if model_outputs is not None:
            _write_lane_model_outputs(output_dir, config=config, model_outputs=model_outputs)
```

- [ ] **Step 7: Run diagnostics tests and verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_pipeline.py::test_pipeline_records_zipformer_endpoint_and_qwen_final_model_outputs tests/test_buffers_metrics.py::test_run_recorder_writes_model_outputs_jsonl tests/test_manager_headless.py::test_fake_headless_mode_emits_events_saves_output_unloads_and_returns_zero -v
```

Expected: PASS.

- [ ] **Step 8: Commit diagnostics wiring**

Run:

```bash
git add src/multi_stream_asr/log/model_output.py src/multi_stream_asr/pipeline/pipeline.py src/multi_stream_asr/pipeline/manager.py src/multi_stream_asr/headless/runner.py src/multi_stream_asr/log/recorder.py tests/test_model_output.py tests/test_pipeline.py tests/test_buffers_metrics.py tests/test_manager_headless.py
git commit -m "feat: record model recognition outputs"
```

---

### Task 5: Full Local And Remote Validation

**Files:**
- Modify: `README.md` only if the final diagnostics command needs to be documented after validation.
- No source changes expected unless validation exposes a bug.

**Interfaces:**
- Consumes: local `.venv`, remote `bianbu@10.0.91.51`, remote project path `~/Workspace/projects/multi-stream-asr`, remote `.venv313`, remote WAV directory `~/Music/wav16k`.
- Produces: local pytest result, remote pytest result, real 4-lane headless run directory, correctness summary.

- [ ] **Step 1: Run full local tests**

Run:

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Check worktree before remote sync**

Run:

```bash
git status --short
```

Expected: only intentional changes are present. The existing untracked `docs/superpowers/specs/design.md` may remain untracked and must not be added.

- [ ] **Step 3: Sync code to remote K3 project**

Run from local workspace:

```bash
rsync -a --delete \
  --exclude .git \
  --exclude .venv \
  --exclude .venv313 \
  --exclude runs \
  ./ bianbu@10.0.91.51:~/Workspace/projects/multi-stream-asr/
```

Expected: command exits with status 0.

- [ ] **Step 4: Run remote pytest under Python 3.13**

Run:

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr && . .venv313/bin/activate && python --version && python -m pytest -v'
```

Expected: Python version is 3.13.x and all tests PASS.

- [ ] **Step 5: Run real 4-lane headless recognition**

Run:

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr && . .venv313/bin/activate && python -m multi_stream_asr.main --headless --lanes 4 --input-dir ~/Music/wav16k'
```

Expected: command exits with status 0. Stdout includes `[model-output]` lines for Zipformer endpoints and Qwen final results.

- [ ] **Step 6: Inspect latest real run artifacts**

Run:

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr && latest=$(ls -td runs/headless-* | head -1) && echo "$latest" && LATEST_RUN="$latest" python - <<'"'"'PY'"'"'
import json
import os
from pathlib import Path

run = Path(os.environ["LATEST_RUN"])
metrics = json.loads((run / "metrics.json").read_text())
model_outputs = [
    json.loads(line)
    for path in sorted((run / "model_outputs").glob("lane*.jsonl"))
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
results = [
    path.read_text(encoding="utf-8")
    for path in sorted((run / "results").glob("lane*.txt"))
]

nonempty_results = [text for text in results if text.strip()]
qwen_finals = [record for record in model_outputs if record["model"] == "qwen3-asr"]
nonempty_qwen = [record for record in qwen_finals if record["text"]]

print("lane_states", [lane["final_state"] for lane in metrics["lanes"]])
print("processed_frames", [lane["processed_frames"] for lane in metrics["lanes"]])
print("dropped_frames", [lane["dropped_frames"] for lane in metrics["lanes"]])
print("offline_requests", [lane["offline_requests"] for lane in metrics["lanes"]])
print("online_avg_ms", metrics["online_frame_latency"]["average_ms"])
print("offline_rtf_avg", metrics["offline_rtf"]["average"])
print("lane_result_count", len(results))
print("nonempty_lane_result_count", len(nonempty_results))
print("qwen_final_count", len(qwen_finals))
print("nonempty_qwen_count", len(nonempty_qwen))
for lane in range(4):
    texts = [record["text_preview"] for record in qwen_finals if record["stream_id"] == lane and record["text"]]
    print("lane", lane, "first_qwen_preview", texts[0] if texts else "")
PY'
```

Expected:

- `lane_states` is `['finished', 'finished', 'finished', 'finished']`.
- `dropped_frames` is `[0, 0, 0, 0]` for the normal WAV run.
- `nonempty_final_count` is greater than 0 and matches `final_count` unless Qwen returns a valid empty sentence for silence.
- `nonempty_qwen_count` is greater than 0 and close to `qwen_final_count`.
- Lane previews are recognizable Chinese text matching the topic of the reference final files.

- [ ] **Step 7: Compare against known-good reference text**

Run:

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr && latest=$(ls -td runs/headless-* | head -1) && LATEST_RUN="$latest" python - <<'"'"'PY'"'"'
import json
import os
from pathlib import Path

run = Path(os.environ["LATEST_RUN"])
records = [
    json.loads(line)
    for path in sorted((run / "model_outputs").glob("lane*.jsonl"))
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
for lane in range(4):
    text = "\n".join(
        record["text"]
        for record in records
        if record["model"] == "qwen3-asr" and record["stream_id"] == lane
    )
    print(f"--- lane {lane} qwen text preview ---")
    print(text[:600])
PY'
```

Manually compare the lane previews with:

```text
/home/liangff/Workspace/projects/4-asr-python/run_logs/asr-results-20260709-142934/lane0_qwen_final.txt
/home/liangff/Workspace/projects/4-asr-python/run_logs/asr-results-20260709-142934/lane1_qwen_final.txt
/home/liangff/Workspace/projects/4-asr-python/run_logs/asr-results-20260709-142934/lane2_qwen_final.txt
/home/liangff/Workspace/projects/4-asr-python/run_logs/asr-results-20260709-142934/lane3_qwen_final.txt
```

Expected: each lane contains the same source topic as the corresponding reference lane. Exact punctuation and segment boundaries do not need byte-identical equality.

- [ ] **Step 8: Confirm no leftover `llama-server`**

Run:

```bash
ssh bianbu@10.0.91.51 'pgrep -a llama-server || true'
```

Expected: no app-started `llama-server` remains.

- [ ] **Step 9: Commit validation documentation if README changed**

If `README.md` was changed, run:

```bash
git add README.md
git commit -m "docs: document model output diagnostics"
```

If `README.md` was not changed, do not create a commit for this step.

---

## Plan Self-Review

- Spec coverage: Qwen payload alignment is covered by Task 1; Zipformer int8 selection is covered by Task 2; model output printing and persistence are covered by Tasks 3 and 4; local and remote real 4-lane validation is covered by Task 5.
- Placeholder scan: no placeholder sections remain in this plan.
- Type consistency: `ModelOutputRecord`, `ModelOutputRecorder`, `OfflineRecognitionResult.raw_response`, and `model_outputs` are named consistently across tasks.
