# Qwen Request Error Isolation and Fatal Threshold Design

## Goal

Make an individual Qwen3-ASR request failure observable without poisoning later
offline recognition requests. Preserve the failed sentence's Zipformer pending
text, write diagnostic artifacts immediately, and escalate to a real global
fatal state only after three Qwen request failures in the same model-loading
session.

## Current Failure Mode

Each lane serializes Qwen requests by awaiting the previous offline task. When
one task raises, every later task awaits and inherits that exception without
sending another HTTP request. The exception remains in `_offline_error` until
recognition ends, at which point the lane becomes fatal.

The client also calls `response.raise_for_status()` before preserving the HTTP
response body. A run can therefore prove that Qwen returned HTTP 500 without
retaining the server's error response. The shared Qwen process log is stored in
`/tmp`, is overwritten on the next model load, and may not contain request-level
diagnostics.

## Error Semantics

Add a nonfatal event status named `qwen_error` for a single failed offline
request. A `qwen_error` event:

- belongs to the lane and sentence that failed;
- does not replace or remove that sentence's Zipformer `pending` event;
- does not produce a Qwen `final` result;
- does not stop its lane or any sibling lane;
- allows the next sentence to issue its Qwen request normally.

The failed request is not retried automatically.

All Qwen request failures count toward the threshold, including:

- HTTP non-success responses;
- connection failures and request timeouts;
- invalid response JSON;
- responses without recognized text.

Cancellation caused by a user stop or a global fatal stop is not a Qwen
failure and does not increment the counter.

## Structured Client Error

`Qwen3AsrClient` will convert request-level failures into a structured
`QwenRequestError`. It carries the information available for the failure:

- request URL and method;
- configured timeout;
- audio duration and sample count, but no PCM or base64 audio body;
- error category, exception type, and message;
- HTTP status code when a response exists;
- response body when a response exists;
- whether the response body was truncated.

The response body is captured before raising for HTTP status. Stored and
printed response bodies are limited to 64 KiB. Larger bodies are truncated and
marked with `response_body_truncated=true`.

`asyncio.CancelledError` remains cancellation and must not be wrapped as
`QwenRequestError`.

## Lane Behavior

Each lane keeps its existing sequential Qwen ordering. The offline task catches
`QwenRequestError` inside that sentence's task, reports the structured failure,
and then returns normally. A handled request failure must not set
`_offline_error`, leave a failed `_offline_tail`, or fail shutdown tail
processing.

On success, the lane reports a lightweight Qwen success IPC message. On
failure, it reports a Qwen error IPC message containing the structured client
error plus:

- run identifier;
- stream ID, sentence ID, and revision;
- Zipformer pending text;
- request completion time.

The lane does not publish `qwen_error` directly. The main process publishes it
after the error artifacts have been written, without replacing the sentence's
pending transcript. This guarantees one event and ensures the frontend cannot
observe an error that was not persisted. Result files continue to contain only
successful Qwen final text; the failed sentence remains observable through
pending model output, the frontend event, and the error artifact.

## Main-Process Coordinator

The main process owns a model-session Qwen error coordinator. It is the only
owner of:

- the cumulative Qwen failure count;
- the Qwen process-log byte checkpoint;
- the active run's Qwen error writers;
- the threshold-to-fatal transition.

This avoids shared counters, cross-process file locks, and duplicate process-log
slices. Lane IPC messages are handled serially by the main event loop. When
multiple lanes complete at nearly the same time, their IPC receive order defines
the checkpoint update order.

The coordinator lifecycle is tied to model loading:

- successful model load initializes the failure count to zero and sets the
  initial process-log checkpoint to the current end of the Qwen log;
- starting recognition attaches error writers for the new run without resetting
  the count;
- ending recognition closes the run writers but preserves the count and log
  checkpoint;
- unloading models clears the count, checkpoint, and fatal latch.

## Immediate Run Artifacts

The current run creates error artifacts only after its first Qwen error:

### `errors/qwen_requests.jsonl`

Append and flush one JSON object per failed request. Each record contains:

- schema version and run ID;
- wall-clock error time;
- stream ID, sentence ID, and revision;
- URL, HTTP method, and timeout;
- audio duration and sample count;
- Zipformer pending text;
- error category, exception type, and message;
- HTTP status and bounded response body;
- `response_body_truncated`;
- failure sequence within the current run;
- cumulative failure sequence within the model-loading session.

### `errors/qwen_server.log`

For every failed request:

1. Read the current size of
   `/tmp/multi-stream-asr/shared-qwen/llama-multi-asr.log`.
2. Append a delimiter containing time, lane, sentence, and byte range.
3. Copy the raw bytes from the previous checkpoint to the current end.
4. If there are no new bytes, append `no new server log bytes`.
5. Flush the artifact and advance the checkpoint to the current end.

Every successful request also advances the checkpoint to the current Qwen log
end, but does not create or write a run artifact. The client JSONL is the
authoritative request-error record because the Qwen binary may not emit a
request-level process log.

Each failure also prints an immediate stderr summary with lane, sentence,
cumulative count, exception type, HTTP status, and the bounded response body.

Failure to write required error artifacts is an infrastructure failure and
causes immediate global fatal escalation rather than silently continuing.

## Failure Threshold and Fatal Stop

The threshold is three cumulative Qwen request failures across all lanes and
all recognition runs since the current model load. Successful requests and
recognition completion do not reset it.

For every failure, the main process performs these actions in order:

1. Calculate the next run and model-session failure sequence values.
2. Persist and flush the client and process-log diagnostics with those values.
3. Commit the updated counters.
4. Publish the sentence-level nonfatal `qwen_error` event.
5. Print the command-line diagnostic.
6. If the model-session count reaches three, trigger global fatal exactly once.

The third failure remains a recorded `qwen_error` and additionally causes a
global `fatal_error`. The process manager transitions to `fatal` and sends a
fatal stop command to every active lane.

Fatal stop differs from normal tail draining:

- stop audio producers and wake online frame queues;
- stop scheduling new offline requests;
- cancel in-flight and queued Qwen tasks;
- do not wait for the ordinary 120-second offline request timeout;
- still collect terminal lane metrics and artifact paths;
- finish `config.json`, `metrics.json`, and error-artifact finalization.

Failures already in flight when the threshold is crossed are still persisted
and counted if their IPC messages arrive. They do not trigger another fatal
transition. Non-Qwen fatal errors retain their existing immediate-fatal
behavior.

After a threshold fatal, the model session remains fatal until models are
unloaded. Unloading models resets the Qwen failure count and related state.

## Frontend Behavior

The frontend recognizes `qwen_error` as a nonfatal, non-transcript event. It
shows the error for the affected sentence while keeping the existing pending
text visible. The first two errors do not change the application state from
running. The third error is followed by the existing global fatal presentation.

The frontend event keying must keep `qwen_error` separate from the transcript
entry for the same lane and sentence so it cannot overwrite pending text.

## Testing

### Client Unit Tests

- HTTP 500 captures status and response body before raising.
- Response bodies over 64 KiB are truncated and marked.
- Timeout and connection failures preserve request metadata without inventing a
  response body.
- Invalid JSON and missing recognized text become structured Qwen errors.
- Cancellation remains cancellation and is not counted.

### Pipeline Unit Tests

- Sentence N failure does not prevent sentence N+1 from making a real Qwen
  call.
- The failed sentence retains pending and receives no final.
- `qwen_error` does not replace pending.
- A handled error does not poison `_offline_tail`, `_offline_tasks`, or finish
  processing.

### Coordinator and Artifact Tests

- Success advances the Qwen log checkpoint without creating artifacts.
- Failure appends one JSONL record and exactly the new process-log byte range.
- Consecutive failures do not duplicate old process-log bytes.
- Empty process-log slices are explicitly marked.
- Response truncation fields and command-line output are correct.
- Counts survive recognition completion and a later run.
- Model unload resets counts and the checkpoint.
- Exactly the third failure triggers one global fatal.
- Artifact write failure triggers immediate fatal.

### Process and Frontend Tests

- The first two errors do not stop any lane.
- The third error stops every lane and cancels offline work without waiting for
  the request timeout.
- Fatal cleanup still writes metrics and paths.
- `qwen_error` is visible while pending text remains visible.
- The frontend enters fatal only after the threshold escalation.

### K3 Acceptance

Use a controllable HTTP stub to inject failures deterministically:

1. Inject the first two failures across two recognition runs without unloading
   models; verify later sentences continue and the count persists.
2. Inject the third failure; verify immediate global stop and complete run
   artifacts.
3. Verify both client and Qwen process-log artifacts and command-line output.
4. Unload and reload models; verify the next failure is counted as one.
5. Run a normal request against the real shared Qwen service; verify the success
   path is unchanged and no error files are created.

## Out of Scope

- Automatic retry of a failed sentence.
- Saving PCM or base64 request audio in error logs.
- Modifying the `llama-multi-asr` binary to add server-side request logging.
- Resetting the failure count after a successful request or recognition run.
