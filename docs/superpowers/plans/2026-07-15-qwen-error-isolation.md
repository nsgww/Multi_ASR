# Qwen Request Error Isolation and Fatal Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve each failed Qwen request's diagnostics and Zipformer pending text, continue later Qwen requests, and stop all recognition only when the model-loading session reaches three cumulative Qwen failures.

**Architecture:** Lane processes convert Qwen failures into structured IPC without leaving failed offline tasks in the serial chain. A main-process `QwenErrorCoordinator` owns the process-log checkpoint, lazy run artifacts, and failure count across recognition runs; `ProcessPipelineManager` publishes nonfatal `qwen_error` events and invokes the existing fatal path at the third failure. Fatal commands cancel queued and in-flight offline work so teardown does not wait for the Qwen timeout.

**Tech Stack:** Python 3.13, asyncio, multiprocessing IPC, httpx, FastAPI WebSocket events, JSONL artifacts, pytest.

## Global Constraints

- Use Python 3.13 and the project-local `.venv313` on the K3 device.
- Do not save PCM samples or base64 audio request bodies.
- Limit saved and printed Qwen response bodies to 64 KiB and mark truncation.
- Do not retry a failed Qwen request.
- Preserve the failed sentence's Zipformer `pending`; do not emit a Qwen `final` for it.
- Count failures across all lanes and recognition runs since model load; successful requests and run completion do not reset the count.
- Escalate exactly the third cumulative failure to global fatal; reset only when models unload.
- Persist error artifacts before publishing `qwen_error` or escalating fatal.
- Keep normal manual stop, natural completion, and successful Qwen behavior unchanged.
- On remote devices, activate `.venv313` and follow `~/.codex/docs/k3/remote-device-python.md` before running Python.

## File Map

- Modify `src/multi_stream_asr/asr/offline/qwen3asr.py`: structured request errors and bounded response-body capture.
- Modify `src/multi_stream_asr/models.py`: `qwen_error` status and JSON-ready sentence failure context.
- Modify `src/multi_stream_asr/pipeline/pipeline.py`: consume per-request errors and continue the serialized offline chain.
- Modify `src/multi_stream_asr/pipeline/lane_runtime.py`: send Qwen success/error IPC and cancel offline work on fatal stop.
- Modify `src/multi_stream_asr/pipeline/process_manager.py`: validate Qwen IPC, coordinate logging/counting, publish events, and trigger fatal.
- Create `src/multi_stream_asr/log/qwen_errors.py`: main-process session counter, log checkpoint, lazy JSONL/process-log artifacts, and stderr output.
- Modify `src/multi_stream_asr/web/server.py`: record fatal runs when completion raises.
- Modify `src/multi_stream_asr/web/static/index.html`: render `qwen_error` separately from pending.
- Modify `tests/test_qwen3asr.py`, `tests/test_pipeline.py`, `tests/test_lane_runtime.py`, `tests/test_process_manager.py`, `tests/test_web_server.py`.
- Create `tests/test_qwen_errors.py`: focused coordinator and artifact tests.

---

### Task 1: Structured Qwen Request Errors

**Files:**
- Modify: `src/multi_stream_asr/asr/offline/qwen3asr.py`
- Test: `tests/test_qwen3asr.py`

**Interfaces:**
- Produces: `MAX_ERROR_RESPONSE_BYTES = 64 * 1024`.
- Produces: `QwenRequestFailure.to_payload() -> dict[str, object]`.
- Produces: `QwenRequestError.failure: QwenRequestFailure`.
- Preserves: `Qwen3AsrClient.recognize(pcm, prompt) -> OfflineRecognitionResult` on success.

- [ ] **Step 1: Write failing HTTP and transport error tests**

Add tests using `httpx.MockTransport` and monkeypatch `httpx.AsyncClient` so no real server is used:

```python
@pytest.mark.asyncio
async def test_recognize_http_500_preserves_bounded_response_body(monkeypatch):
    body = b"server detail:" + b"x" * (64 * 1024 + 10)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=body, request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        qwen3asr.httpx,
        "AsyncClient",
        lambda: real_client(transport=transport),
    )
    client = Qwen3AsrClient(shared=SharedQwenConfig(request_timeout_ms=1234))

    with pytest.raises(QwenRequestError) as raised:
        await client.recognize(np.zeros(16000, dtype=np.float32), prompt="pending text")

    failure = raised.value.failure
    assert failure.category == "http_status"
    assert failure.status_code == 500
    assert len(failure.response_body.encode("utf-8")) <= 64 * 1024
    assert failure.response_body_truncated is True
    assert failure.method == "POST"
    assert failure.url.endswith("/v1/chat/completions")
    assert failure.timeout_ms == 1234
    assert failure.audio_seconds == 1.0
    assert failure.sample_count == 16000


@pytest.mark.asyncio
async def test_recognize_timeout_has_no_response_body(monkeypatch):
    class TimeoutClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return None
        async def post(self, *args, **kwargs):
            raise httpx.ReadTimeout("timed out", request=httpx.Request("POST", args[0]))

    monkeypatch.setattr(qwen3asr.httpx, "AsyncClient", TimeoutClient)
    client = Qwen3AsrClient(shared=SharedQwenConfig())

    with pytest.raises(QwenRequestError) as raised:
        await client.recognize(np.zeros(8000, dtype=np.float32), prompt="pending")

    assert raised.value.failure.category == "transport"
    assert raised.value.failure.status_code is None
    assert raised.value.failure.response_body is None
    assert raised.value.failure.sample_count == 8000
```

Also add one HTTP 200 invalid-JSON test and one HTTP 200 missing-text test; assert categories `invalid_json` and `invalid_response` respectively.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
source .venv313/bin/activate
python -m pytest tests/test_qwen3asr.py -k "preserves_bounded or timeout_has_no or invalid_json or missing_text" -v
```

Expected: FAIL because `QwenRequestError` and `QwenRequestFailure` do not exist and HTTP bodies are discarded by `raise_for_status()`.

- [ ] **Step 3: Implement structured failure capture**

Add these public types and helper near the Qwen constants:

```python
from dataclasses import asdict, dataclass

MAX_ERROR_RESPONSE_BYTES = 64 * 1024


@dataclass(frozen=True)
class QwenRequestFailure:
    category: str
    error_type: str
    message: str
    method: str
    url: str
    timeout_ms: int
    audio_seconds: float
    sample_count: int
    status_code: int | None = None
    response_body: str | None = None
    response_body_truncated: bool = False

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


class QwenRequestError(RuntimeError):
    def __init__(self, failure: QwenRequestFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


def _bounded_response_body(response: httpx.Response) -> tuple[str, bool]:
    content = response.content
    truncated = len(content) > MAX_ERROR_RESPONSE_BYTES
    return content[:MAX_ERROR_RESPONSE_BYTES].decode("utf-8", errors="replace"), truncated
```

Refactor `recognize()` to build common request metadata once, catch `httpx.HTTPError`, inspect `response.is_success` before JSON parsing, and wrap JSON/text parsing errors. Do not use a broad `BaseException` catch; `asyncio.CancelledError` must pass through unchanged.

Use this ordering:

```python
response = await client.post(url, json=payload, timeout=request_timeout_seconds)
if not response.is_success:
    body, truncated = _bounded_response_body(response)
    raise QwenRequestError(QwenRequestFailure(
        category="http_status",
        error_type="HTTPStatusError",
        message=f"Qwen request returned HTTP {response.status_code}",
        status_code=response.status_code,
        response_body=body,
        response_body_truncated=truncated,
        **request_metadata,
    ))
try:
    raw_response = response.json()
except ValueError as exc:
    body, truncated = _bounded_response_body(response)
    raise QwenRequestError(QwenRequestFailure(
        category="invalid_json",
        error_type=type(exc).__name__,
        message=str(exc),
        status_code=response.status_code,
        response_body=body,
        response_body_truncated=truncated,
        **request_metadata,
    )) from exc
```

Wrap `clean_qwen_text()` failures as `invalid_response` with the same bounded response body. Wrap only `httpx.HTTPError` around `client.post()` as category `transport`.

- [ ] **Step 4: Run client tests**

Run:

```bash
source .venv313/bin/activate
python -m pytest tests/test_qwen3asr.py -v
```

Expected: all `tests/test_qwen3asr.py` tests PASS, including existing successful request/timing tests.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/asr/offline/qwen3asr.py tests/test_qwen3asr.py
git commit -m "feat: preserve qwen request error details"
```

### Task 2: Isolate a Failed Sentence from the Offline Task Chain

**Files:**
- Modify: `src/multi_stream_asr/models.py`
- Modify: `src/multi_stream_asr/pipeline/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `QwenRequestError.failure.to_payload()` from Task 1.
- Produces: `EventStatus.QWEN_ERROR = "qwen_error"`.
- Produces: `QwenSentenceFailure(stream_id, sentence_id, revision, pending_text, timestamp_ms, request)`.
- Produces constructor callbacks `on_qwen_success: Callable[[int], None] | None` and `on_qwen_error: Callable[[QwenSentenceFailure], None] | None`.
- Produces: `LanePipeline.cancel_offline_tasks() -> None`.

- [ ] **Step 1: Write failing pipeline continuation tests**

Add a fake offline recognizer that fails once and then succeeds:

```python
class FailOnceOfflineRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    async def recognize(self, pcm, prompt):
        self.calls += 1
        if self.calls == 1:
            raise QwenRequestError(QwenRequestFailure(
                category="http_status", error_type="HTTPStatusError",
                message="HTTP 500", method="POST", url="http://qwen/v1/chat/completions",
                timeout_ms=120000, audio_seconds=1.0, sample_count=16000,
                status_code=500, response_body="decoder failed",
            ))
        return OfflineRecognitionResult(
            text="second final", elapsed_ms=100.0, audio_seconds=1.0, rtf=0.1,
        )
```

Schedule two offline requests through `_schedule_offline()`, await
`wait_for_offline_tail(timeout_seconds=1)`, and assert:

```python
assert recognizer.calls == 2
assert [failure.sentence_id for failure in failures] == [1]
assert successes == [2]
assert pipeline._offline_error is None
assert any(event.status is EventStatus.PENDING and event.sentence_id == 1 for event in events)
assert not any(event.status is EventStatus.FINAL and event.sentence_id == 1 for event in events)
```

Add a cancellation test that creates a blocking offline request, calls
`pipeline.cancel_offline_tasks()`, and verifies the task is cancelled and
`wait_for_offline_tail()` returns without waiting for its configured timeout.

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
source .venv313/bin/activate
python -m pytest tests/test_pipeline.py -k "qwen_failure or cancel_offline" -v
```

Expected: FAIL because callbacks, `QwenSentenceFailure`, and cancellation support do not exist; the second task inherits the first exception.

- [ ] **Step 3: Add the status and failure context**

In `models.py` add:

```python
class EventStatus(str, Enum):
    PARTIAL = "partial"
    PENDING = "pending"
    FINAL = "final"
    QWEN_ERROR = "qwen_error"
    FATAL_ERROR = "fatal_error"
    LANE_FINISHED = "lane_finished"
    STATS_UPDATED = "stats_updated"


@dataclass(frozen=True)
class QwenSentenceFailure:
    stream_id: int
    sentence_id: int
    revision: int
    pending_text: str
    timestamp_ms: int
    request: dict[str, object]
```

Update `ASREvent.asr_result()` so `qwen_error` uses `ResultType.OFFLINE` but remains `is_final=False`:

```python
is_final = status is EventStatus.FINAL
result_type = (
    ResultType.OFFLINE
    if status in {EventStatus.FINAL, EventStatus.QWEN_ERROR}
    else ResultType.ONLINE
)
```

- [ ] **Step 4: Consume Qwen errors inside each offline task**

Extend `LanePipeline.__init__()` with optional callbacks and save them. In
`run_offline()` catch only `QwenRequestError`:

```python
try:
    result = await self.offline_recognizer.recognize(pcm=pcm, prompt=prompt)
except QwenRequestError as exc:
    if self.on_qwen_error is not None:
        self.on_qwen_error(QwenSentenceFailure(
            stream_id=self.stream_id,
            sentence_id=sentence_id,
            revision=revision,
            pending_text=prompt,
            timestamp_ms=timestamp_ms,
            request=exc.failure.to_payload(),
        ))
    return

if self.on_qwen_success is not None:
    self.on_qwen_success(sentence_id)
```

Do not publish `qwen_error` in the lane. Main-process persistence must happen first.

Add event-loop-only cancellation:

```python
def cancel_offline_tasks(self) -> None:
    for task in tuple(self._offline_tasks):
        if not task.done():
            task.cancel()
```

Keep `_record_offline_task_done()` discarding cancelled tasks. A handled
`QwenRequestError` returns normally, so later tasks can await it safely.

- [ ] **Step 5: Run pipeline and model tests**

```bash
source .venv313/bin/activate
python -m pytest tests/test_pipeline.py tests/test_models_state.py tests/test_ipc.py -v
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/multi_stream_asr/models.py src/multi_stream_asr/pipeline/pipeline.py tests/test_pipeline.py
git commit -m "fix: isolate failed qwen sentences"
```

### Task 3: Report Qwen Completion Through Lane IPC

**Files:**
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py`
- Test: `tests/test_lane_runtime.py`

**Interfaces:**
- Consumes: pipeline callbacks and `QwenSentenceFailure` from Task 2.
- Produces IPC type `qwen_success` with `run_id`, `sentence_id`.
- Produces IPC type `qwen_error` with `run_id`, sentence context, and structured request fields.

- [ ] **Step 1: Write failing lane IPC tests**

Extend the fake offline recognizer used by lane runtime tests so one endpoint
raises `QwenRequestError`. Assert the emitted IPC sequence contains:

```python
qwen_error = next(message for message in emitted if message.type == "qwen_error")
assert qwen_error.payload["run_id"] == run_dir.name
assert qwen_error.payload["sentence_id"] == 1
assert qwen_error.payload["pending_text"] == "fake endpoint lane 0"
assert qwen_error.payload["request"]["status_code"] == 500
assert not any(message.type == "fatal" for message in emitted)
```

Add a success case asserting exactly one `qwen_success` message with the same
run ID and sentence ID.

- [ ] **Step 2: Run tests and verify failure**

```bash
source .venv313/bin/activate
python -m pytest tests/test_lane_runtime.py -k "qwen_error_ipc or qwen_success_ipc" -v
```

Expected: FAIL because `LaneRuntime` does not wire Qwen callbacks.

- [ ] **Step 3: Wire callbacks into the lane pipeline**

In `_run_started()`, define callbacks that close over `run_dir`:

```python
def report_qwen_success(sentence_id: int) -> None:
    self.send("qwen_success", {
        "run_id": run_dir.name,
        "sentence_id": sentence_id,
    })

def report_qwen_error(failure: QwenSentenceFailure) -> None:
    self.send("qwen_error", {
        "run_id": run_dir.name,
        "sentence_id": failure.sentence_id,
        "revision": failure.revision,
        "pending_text": failure.pending_text,
        "timestamp_ms": failure.timestamp_ms,
        "request": failure.request,
    })
```

Pass them as `on_qwen_success` and `on_qwen_error` to `LanePipeline`. The IPC
envelope already supplies `stream_id` and wall-clock `time_ms`; do not duplicate
audio payload data.

- [ ] **Step 4: Run lane runtime tests**

```bash
source .venv313/bin/activate
python -m pytest tests/test_lane_runtime.py -v
```

Expected: all lane runtime tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/pipeline/lane_runtime.py tests/test_lane_runtime.py
git commit -m "feat: report qwen request outcomes over ipc"
```

### Task 4: Main-Process Qwen Error Coordinator and Artifacts

**Files:**
- Create: `src/multi_stream_asr/log/qwen_errors.py`
- Create: `tests/test_qwen_errors.py`

**Interfaces:**
- Produces: `QwenFailureDecision(run_sequence: int, session_sequence: int, should_fatal: bool)`.
- Produces: `QwenErrorCoordinator.reset_for_model_load()`, `start_run(run_dir)`, `record_success()`, `record_failure(...)`, `finish_run()`, and `reset_for_unload()`.
- Creates lazily: `errors/qwen_requests.jsonl` and `errors/qwen_server.log`.

- [ ] **Step 1: Write failing coordinator tests**

Create tests with a temporary server log and `io.StringIO` stderr:

```python
def failure_payload(status=500, body="decoder failed"):
    return {
        "run_id": "process-1", "sentence_id": 34, "revision": 5,
        "pending_text": "pending text", "timestamp_ms": 232700,
        "request": {
            "category": "http_status", "error_type": "HTTPStatusError",
            "message": "HTTP 500", "method": "POST",
            "url": "http://127.0.0.1:8090/v1/chat/completions",
            "timeout_ms": 120000, "audio_seconds": 2.6,
            "sample_count": 41600, "status_code": status,
            "response_body": body, "response_body_truncated": False,
        },
    }


def test_success_advances_checkpoint_and_failure_saves_only_new_bytes(tmp_path):
    server_log = tmp_path / "llama.log"
    server_log.write_bytes(b"startup\n")
    stderr = io.StringIO()
    coordinator = QwenErrorCoordinator(server_log_path=server_log, stderr=stderr)
    coordinator.reset_for_model_load()
    run_dir = tmp_path / "process-1"
    coordinator.start_run(run_dir)

    server_log.write_bytes(b"startup\nsuccess noise\n")
    coordinator.record_success()
    server_log.write_bytes(b"startup\nsuccess noise\nerror detail\n")
    decision = coordinator.record_failure(stream_id=2, time_ms=1000, payload=failure_payload())

    assert decision == QwenFailureDecision(1, 1, False)
    assert "error detail" in (run_dir / "errors/qwen_server.log").read_text()
    assert "success noise" not in (run_dir / "errors/qwen_server.log").read_text()
    record = json.loads((run_dir / "errors/qwen_requests.jsonl").read_text())
    assert record["stream_id"] == 2
    assert record["pending_text"] == "pending text"
    assert record["run_failure_sequence"] == 1
    assert record["session_failure_sequence"] == 1
```

Add tests for no-new-log marker, no duplicate bytes across two failures,
cross-run count preservation, exactly the third decision returning
`should_fatal=True`, unload reset, lazy file creation, and stderr content.

- [ ] **Step 2: Run tests and verify failure**

```bash
source .venv313/bin/activate
python -m pytest tests/test_qwen_errors.py -v
```

Expected: FAIL because `multi_stream_asr.log.qwen_errors` does not exist.

- [ ] **Step 3: Implement the coordinator**

Use a focused synchronous class; it runs only on the main event loop and does
not need locks:

```python
@dataclass(frozen=True)
class QwenFailureDecision:
    run_sequence: int
    session_sequence: int
    should_fatal: bool


class QwenErrorCoordinator:
    def __init__(self, *, server_log_path: Path, stderr: TextIO = sys.stderr) -> None:
        self.server_log_path = server_log_path
        self.stderr = stderr
        self._checkpoint = 0
        self._session_failures = 0
        self._run_failures = 0
        self._run_dir: Path | None = None
        self._fatal_latched = False

    def reset_for_model_load(self) -> None:
        self.finish_run()
        self._session_failures = 0
        self._run_failures = 0
        self._fatal_latched = False
        self._checkpoint = self._server_log_size()

    def start_run(self, run_dir: Path) -> None:
        self.finish_run()
        self._run_dir = run_dir
        self._run_failures = 0

    def record_success(self) -> None:
        self._checkpoint = self._server_log_size()
```

`record_failure()` must calculate the next counters, lazily create the two files,
write and flush both records, then commit counters/checkpoint. Return
`should_fatal=True` only when `session_sequence >= 3` and `_fatal_latched` was
false; latch immediately after successful writes. Read server logs as bytes and
decode copied bytes with `errors="replace"`. Emit `no new server log bytes` when
the range is empty.

If opening, reading, writing, or flushing either required artifact raises, let
the exception propagate so the process manager can enter immediate fatal.

- [ ] **Step 4: Run coordinator tests**

```bash
source .venv313/bin/activate
python -m pytest tests/test_qwen_errors.py -v
```

Expected: all coordinator tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/log/qwen_errors.py tests/test_qwen_errors.py
git commit -m "feat: persist qwen request failures"
```

### Task 5: Integrate Global Counting, Events, and Threshold Fatal

**Files:**
- Modify: `src/multi_stream_asr/pipeline/process_manager.py`
- Test: `tests/test_process_manager.py`

**Interfaces:**
- Consumes: `QwenErrorCoordinator` and `QwenFailureDecision` from Task 4.
- Consumes IPC: `qwen_success`, `qwen_error` from Task 3.
- Publishes: sentence-level `EventStatus.QWEN_ERROR` after persistence.
- Triggers: `_record_fatal()` exactly once at session failure three.

- [ ] **Step 1: Write failing IPC validation and threshold tests**

Add `qwen_success` and `qwen_error` to fake controller emissions. Test exact
payload validation: matching run ID, integer sentence/revision/timestamp,
nonempty pending text allowed to be empty only when Zipformer text is empty,
and a mapping request payload with required request fields.

Use a recording coordinator in manager tests:

```python
class RecordingQwenErrors:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.successes = 0
        self.failures = []
        self.started = []
        self.finished = 0
        self.reset_load = 0
        self.reset_unload = 0
    def reset_for_model_load(self): self.reset_load += 1
    def start_run(self, run_dir): self.started.append(run_dir)
    def record_success(self): self.successes += 1
    def record_failure(self, **kwargs):
        self.failures.append(kwargs)
        return next(self.decisions)
    def finish_run(self): self.finished += 1
    def reset_for_unload(self): self.reset_unload += 1
```

Assert failures one and two publish `qwen_error` without manager failure or stop
commands. Assert failure three publishes both `qwen_error` and `fatal_error`,
sets `PipelineState.FATAL`, and sends `{"type": "stop", "reason": "sibling_fatal"}`
to all active controllers.

Add a test where `record_failure()` raises `OSError("disk full")`; assert immediate
fatal regardless of prior Qwen count.

- [ ] **Step 2: Run focused process manager tests and verify failure**

```bash
source .venv313/bin/activate
python -m pytest tests/test_process_manager.py -k "qwen_success or qwen_error or threshold or artifact_write" -v
```

Expected: FAIL because message types and coordinator integration do not exist.

- [ ] **Step 3: Add coordinator dependency and lifecycle hooks**

Extend `ProcessPipelineManager.__init__()` with an optional injected coordinator:

```python
qwen_error_coordinator: QwenErrorCoordinator | None = None,
```

Initialize the default with `SHARED_LOG_PATH`. Call:

- `reset_for_model_load()` after the shared server is healthy and before lanes run;
- `start_run(self._run_dir)` immediately after `_new_run_dir()`;
- `finish_run()` in every normal, stopped, and fatal completion path;
- `reset_for_unload()` after shared Qwen shutdown completes.

Do not call `reset_for_model_load()` from `_reset_for_run()`.

- [ ] **Step 4: Validate and handle Qwen IPC**

Add both message types to `_ALLOWED_MESSAGE_TYPES`. Validate payloads before
dispatch. In `_handle_message()`:

```python
elif message.type == "qwen_success":
    self._qwen_errors.record_success()
elif message.type == "qwen_error":
    try:
        decision = self._qwen_errors.record_failure(
            stream_id=stream_id,
            time_ms=message.time_ms,
            payload=payload,
        )
    except Exception as exc:
        await self._record_fatal(
            stream_id,
            f"failed to persist Qwen request error: {type(exc).__name__}: {exc}",
        )
    else:
        request = dict(payload["request"])
        status = request.get("status_code")
        text = f"Qwen request failed: {request['error_type']}"
        if status is not None:
            text += f" (HTTP {status})"
        self._event_bus.publish(ASREvent.asr_result(
            stream_id=stream_id,
            sentence_id=int(payload["sentence_id"]),
            revision=int(payload["revision"]),
            status=EventStatus.QWEN_ERROR,
            text=text,
            timestamp_ms=int(payload["timestamp_ms"]),
        ))
        if decision.should_fatal:
            await self._record_fatal(
                stream_id,
                f"Qwen request failure threshold reached ({decision.session_sequence}/3)",
            )
```

The coordinator prints the detailed stderr record; do not duplicate full response
bodies in `ProcessPipelineManager`.

- [ ] **Step 5: Run process manager tests**

```bash
source .venv313/bin/activate
python -m pytest tests/test_process_manager.py -v
```

Expected: all process manager tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/multi_stream_asr/pipeline/process_manager.py tests/test_process_manager.py
git commit -m "feat: escalate cumulative qwen failures"
```

### Task 6: Make Fatal Stop Cancel Offline Work and Persist Fatal Runs

**Files:**
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py`
- Modify: `src/multi_stream_asr/web/server.py`
- Test: `tests/test_lane_runtime.py`
- Test: `tests/test_process_manager.py`
- Test: `tests/test_web_server.py`

**Interfaces:**
- Consumes: `LanePipeline.cancel_offline_tasks()` from Task 2.
- Treats stop reasons `sibling_fatal` and `qwen_failure_threshold` as fatal cancellation.
- Guarantees fatal run recording even when `wait_for_completion()` raises.

- [ ] **Step 1: Write failing fatal cancellation tests**

Create a fake offline recognizer whose request waits forever after setting an
`asyncio.Event`. Send `{"type": "stop", "reason": "sibling_fatal"}` and assert
the lane reaches `lane_stopped` within one second and the recognizer observes
`CancelledError`. Add a control test proving `user_requested` retains normal
tail-drain behavior.

In `test_web_server.py`, make `wait_for_completion()` raise after exposing a run
directory and lane summaries. Assert `_record_results()` still calls
`save_process_run()` exactly once for the fatal run.

- [ ] **Step 2: Run tests and verify failure**

```bash
source .venv313/bin/activate
python -m pytest tests/test_lane_runtime.py tests/test_web_server.py -k "fatal_cancel or fatal_run" -v
```

Expected: FAIL because fatal stop waits for offline tail and web completion returns before recording.

- [ ] **Step 3: Cancel offline work for fatal commands**

In `_watch_runtime_commands()` inspect the reason before stopping the source:

```python
reason = str(command.get("reason", ""))
fatal_stop = command_type == "shutdown" or reason in {
    "sibling_fatal",
    "qwen_failure_threshold",
}
if self._pipeline is not None:
    self._pipeline.request_stop()
    if fatal_stop:
        self._pipeline.cancel_offline_tasks()
```

Ensure `cancel_offline_tasks()` is invoked on the lane event-loop thread. After
cancellation, `wait_for_offline_tail()` must see no pending request and return
without waiting for `request_timeout_ms`.

- [ ] **Step 4: Record process runs on exceptional completion**

Change `_record_after_completion()` to record in `finally` when the same manager
still owns a run and its state is `LOADED` or `FATAL`:

```python
async def _record_after_completion(app: FastAPI, manager: Any) -> None:
    try:
        await manager.wait_for_completion()
    except Exception:
        pass
    finally:
        if (
            app.state.manager is manager
            and manager.state in {PipelineState.LOADED, PipelineState.FATAL}
            and getattr(manager, "run_dir", None) is not None
        ):
            _record_results(app)
```

The existing `recorded_process_run_dirs` guard prevents duplicate saves during
later unload.

- [ ] **Step 5: Run lifecycle and web tests**

```bash
source .venv313/bin/activate
python -m pytest tests/test_lane_runtime.py tests/test_process_manager.py tests/test_web_server.py -v
```

Expected: all selected suites PASS; fatal cancellation completes under the bounded assertion.

- [ ] **Step 6: Commit**

```bash
git add src/multi_stream_asr/pipeline/lane_runtime.py src/multi_stream_asr/web/server.py tests/test_lane_runtime.py tests/test_process_manager.py tests/test_web_server.py
git commit -m "fix: stop offline work on fatal errors"
```

### Task 7: Keep Qwen Errors Separate from Pending in the Frontend

**Files:**
- Modify: `src/multi_stream_asr/web/static/index.html`
- Test: `tests/test_web_server.py`

**Interfaces:**
- Consumes WebSocket status `qwen_error` from Task 5.
- Preserves transcript revision key `${streamId}:${sentenceId}`.
- Stores error key `${streamId}:${sentenceId}:qwen_error`.

- [ ] **Step 1: Write failing HTML contract tests**

Add assertions against the served HTML:

```python
assert 'event.status === "qwen_error"' in response.text
assert '`${streamId}:${sentenceId}:qwen_error`' in response.text
assert '.result.qwen_error' in response.text
assert 'const tracksRevision = ["partial", "pending", "final"].includes(event.status);' in response.text
```

Extract the `renderLaneEvent` source and assert it uses a status-aware storage
key while revision tracking still uses the transcript-only key.

- [ ] **Step 2: Run the focused test and verify failure**

```bash
source .venv313/bin/activate
python -m pytest tests/test_web_server.py -k qwen_error -v
```

Expected: FAIL because the frontend has no separate Qwen error key or style.

- [ ] **Step 3: Implement separate event keying and restrained error styling**

Add:

```javascript
function eventStorageKey(event) {
  if (event.status === "qwen_error") {
    return `${event.stream_id}:${event.sentence_id}:qwen_error`;
  }
  return laneKey(event.stream_id, event.sentence_id);
}
```

Use `eventStorageKey(event)` only for `lane.events.set(...)`; continue using
`laneKey()` for transcript revisions. Render the Qwen error as a separate row
after the pending row. Add a compact amber/red error style that wraps long
response summaries and does not change lane dimensions:

```css
.result.qwen_error {
  color: #9a3412;
  font-size: 12px;
  overflow-wrap: anywhere;
}
```

Do not display the full response body in the frontend; it remains in run logs
and stderr.

- [ ] **Step 4: Run frontend server tests**

```bash
source .venv313/bin/activate
python -m pytest tests/test_web_server.py -v
```

Expected: all web server tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/web/static/index.html tests/test_web_server.py
git commit -m "feat: display nonfatal qwen request errors"
```

### Task 8: Full Regression and K3 Acceptance

**Files:**
- Modify only if verification reveals a defect in files already listed above.
- Verify: `docs/superpowers/specs/2026-07-15-qwen-error-isolation-design.md`

**Interfaces:**
- Verifies all interfaces and acceptance criteria from Tasks 1-7.

- [ ] **Step 1: Run the complete local test suite**

```bash
source .venv313/bin/activate
python -m pytest -q
```

Expected: exit code 0 with no failed tests.

- [ ] **Step 2: Run static repository checks**

```bash
git diff --check
source .venv313/bin/activate
python -m compileall -q src scripts tests
```

Expected: both commands exit 0 with no output indicating an error.

- [ ] **Step 3: Synchronize committed code to the K3 test checkout**

Following the repository workflow, commit locally first, then on K3:

```bash
ssh K3 'cd /home/bianbu/Workspace/projects/multi-stream-asr && git pull --ff-only'
```

Expected: K3 checkout reaches the same commit as local `git rev-parse HEAD`.

- [ ] **Step 4: Run deterministic failure tests on K3 with Python 3.13**

```bash
ssh K3 'cd /home/bianbu/Workspace/projects/multi-stream-asr && \
  export UV_EXTRA_INDEX_URL="https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple" && \
  export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple && \
  export PIP_INDEX_URL=https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple && \
  export PIP_EXTRA_INDEX_URL="https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple https://mirrors.aliyun.com/pypi/simple" && \
  source .venv313/bin/activate && python --version && \
  python -m pytest tests/test_qwen3asr.py tests/test_qwen_errors.py tests/test_pipeline.py tests/test_lane_runtime.py tests/test_process_manager.py tests/test_web_server.py -q'
```

Expected: Python reports 3.13 and all selected tests PASS. These tests inject
HTTP 500 deterministically, verify failures one and two continue, verify the
third fatal cancels work, and verify unload reset.

- [ ] **Step 5: Run a real shared-Qwen smoke request**

Ensure the web service is stopped so port 8090 is free. The smoke script starts
and stops the configured shared Qwen process itself, then run:

```bash
ssh K3 'cd /home/bianbu/Workspace/projects/multi-stream-asr && \
  source .venv313/bin/activate && \
  python scripts/smoke_shared_qwen.py \
    src/multi_stream_asr/assets/wav16k/00_simple_news_16000hz_1ch.wav \
    --seconds 10'
```

Expected: nonempty recognized text, HTTP success, and no new
`errors/qwen_requests.jsonl` for a successful-only run.

- [ ] **Step 6: Inspect retained run artifacts after injected failures**

For the generated failure run:

```bash
ssh K3 'cd /home/bianbu/Workspace/projects/multi-stream-asr && \
  RUN=$(find runs -maxdepth 1 -type d -name "process-*" | sort | tail -1) && \
  test -s "$RUN/errors/qwen_requests.jsonl" && \
  test -s "$RUN/errors/qwen_server.log" && \
  python3.13 -c "import json,sys; rows=[json.loads(x) for x in open(sys.argv[1])]; assert rows and all(\"pending_text\" in x and \"response_body\" in x for x in rows)" "$RUN/errors/qwen_requests.jsonl"'
```

Expected: both files are nonempty and every JSONL record contains pending text
and the captured response body field.

- [ ] **Step 7: Final review and commit any verification-only correction**

```bash
git status --short
git log --oneline -8
```

Expected: only known pre-existing untracked files remain. If verification
required a scoped correction, return to that component's task, rerun its focused
test and the full suite, then commit only the source and matching tests named in
that task. Do not add unrelated untracked workspace files.
