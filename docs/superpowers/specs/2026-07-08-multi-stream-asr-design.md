# Multi-Stream ASR Design Spec

## Purpose

Build a Python 3.13 application that runs 2, 4, 6, or 8 independent ASR lanes. Each lane streams 16 kHz mono audio through sherpa-onnx zipformer for low-latency online recognition, then submits endpoint-delimited segments to a lane-local Qwen3-ASR `llama-server` for offline correction. Results are rendered through the same event semantics in the browser, headless mode, logs, and saved outputs.

The project owns application orchestration only: configuration, audio input, pipeline scheduling, ASR adapters, model process lifecycle, FastAPI/WebSocket service, headless execution, logging, metrics, and tests. It does not implement sherpa-onnx internals, Qwen3-ASR internals, or `llama-server` internals.

## Scope

In scope:

- Python package and CLI entry point under `src/multi_stream_asr`.
- FastAPI app with WebSocket updates and static HTML UI.
- Headless mode using the same service objects and event bus as Web mode.
- Local WAV input and microphone input abstractions.
- Online sherpa-onnx zipformer adapter.
- Offline Qwen3-ASR HTTP adapter and `llama-server` process manager.
- Lane-local pipeline orchestration, segment cache, state machine, event queue, logging, metrics, and result persistence.
- pytest unit tests and fake-adapter integration smoke tests.
- Final remote-device headless validation with real models.

Out of scope:

- Implementing model kernels, decoding algorithms, or VAD internals.
- Rewriting sherpa-onnx or `llama-server`.
- Explicit Python-side Linux thread affinity. A100 binding is handled by sherpa-onnx and `llama-server`/SMT backends; this project only generates and passes the required configuration.

## Runtime Environment

Development, validation, and inference on the remote K3 device must use Python 3.13 and a project-local `.venv313`.

Commands that install dependencies or run project code on the remote device must activate `.venv313` first. Dependency installation must also export package index variables first:

```bash
python3.13 -m venv .venv313
. .venv313/bin/activate
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Subsequent runs must begin with:

```bash
. .venv313/bin/activate
```

## Architecture

The implementation should use a layered Python application, with fake adapters available for tests and real adapters available for deployment.

Primary modules:

- `config.py`: typed app configuration, defaults, model paths, lane count, endpoint rules, ports, input settings, and output paths.
- `models.py`: structured data models for audio frames, lane configuration, ASR events, metrics, and recognition results.
- `pipeline/manager.py`: serializes model load/unload and recognition start/stop across all lanes.
- `pipeline/pipeline.py`: owns one lane's audio queue, online consumer, segment cache, offline submission, sentence IDs, revisions, and lane state.
- `pipeline/state.py`: legal state transitions for unloaded, loading, loaded, starting, running, stopping, stopped, unloading, and fatal states.
- `audio/base.py`, `audio/local_file.py`, `audio/mic.py`, `audio/device.py`: audio input abstraction, WAV validation/reading, microphone support, and device enumeration.
- `asr/online/base.py`, `asr/online/zipformer.py`: online recognizer interface and sherpa-onnx implementation.
- `asr/offline/base.py`, `asr/offline/qwen3asr.py`, `asr/offline/llama_server.py`: offline recognizer interface, Qwen3-ASR HTTP client, and `llama-server` lifecycle manager.
- `buffer/audio_queue.py`, `buffer/segment_cache.py`: bounded online queue behavior and offline segment audio cache.
- `web/server.py`, `web/event_bus.py`, `web/static/index.html`: FastAPI routes, WebSocket endpoint, broadcast queue, connection management, and UI.
- `headless/runner.py`: headless control flow that uses the same app services as Web mode.
- `log/recorder.py`, `log/metrics.py`: result persistence, warnings, per-lane latency, offline RTF, frame counts, dropped frames, and final state.
- `utils/paths.py`, `utils/time.py`, `utils/process.py`: path expansion, time helpers, subprocess helpers, and process cleanup.

## Audio Input

All lanes use 16 kHz mono audio. The default frame duration is 100 ms and must be configurable.

Each audio source produces `AudioFrame` records with:

- `stream_id`: zero-based lane ID.
- `frame_index`: monotonically increasing lane-local frame index.
- `pcm`: float32 mono PCM at 16 kHz.
- `timestamp_ms`: production timestamp.
- `is_finish`: true only for the final finish frame.

All configured lanes must initialize first. After microphone devices are ready and local files are validated, producers start together and send 100 ms frames into their lane pipelines. Producers must not block on ASR inference or WebSocket sending. When a local file ends or a microphone lane receives a stop signal, the producer sends exactly one finish frame.

Local file input requirements:

- Files must be 16 kHz mono PCM WAV.
- Invalid format is an input error.
- Headless mode validates all input files before recognition starts.
- Default directory is `~/Music/wav16k`.
- Default selection is the first N files by filename sort; if fewer than N files exist, lanes are assigned by cycling through available files.

## Online Recognition and Segmentation

Each lane has a bounded online queue with default capacity 1. If the queue is full, the pipeline may drop replaceable online intermediate audio work and record a warning, but it must not block audio production. Offline final results must never be dropped.

Each lane has one online consumer that calls sherpa-onnx zipformer. The default online model path is:

```text
~/Workspace/sm-sdk/models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
```

Each online recognizer uses `num_threads=1`.

Default zipformer endpoint rules:

- `rule1_min_trailing_silence = 1.0s`
- `rule2_min_trailing_silence = 0.5s`
- `rule3_min_utterance_length = 10.0s`

These values must be configurable. The offline segment cache capacity is `1.5 * rule3_min_utterance_length`, so the default capacity is 15.0 seconds of audio. The cache stores PCM frames that have already been sent to online recognition since the previous endpoint. The endpoint signal is a sentence-level trigger; it does not need sample-accurate VAD boundaries, and offline audio may contain small leading or trailing silence. No extra pre-padding or post-padding is added.

When online text changes, the lane publishes a `partial` ASR result event. When `is_endpoint` is true:

- If the current sentence has not produced non-empty online text, the lane clears the segment cache and does not submit an offline request.
- If the current sentence has produced non-empty online text, the lane publishes a `pending` ASR result event, submits the cached segment to the offline worker asynchronously, and clears the segment cache.
- The online text is passed as prompt/reference text for offline correction.
- The lane increments `sentence_id` only after the endpoint handling for the current sentence is scheduled.

`pending` is a frontend-visible and headless-visible event. It updates the current sentence and means "online has cut this sentence; offline correction is still pending." The later `final` event for the same `stream_id + sentence_id` replaces it. An empty string from offline recognition is a legal final result and must still replace the pending text.

A finish frame resets the online recognizer and emits a lane finished event after all already-submitted offline work for that lane is accounted for.

## Offline Recognition

The offline model is Qwen3-ASR 0.6B. The default model directory is:

```text
~/Workspace/sm-sdk/models/asr/qwen3asr
```

Each lane has one dedicated `llama-server`, with ports assigned from `8063` upward. Since each service is lane-local, every request uses `id_slot=0`. Requests for the same lane are processed serially. Requests for different lanes run concurrently against independent ports and independent `llama-server` processes.

Before starting offline servers, `LlamaServerManager` must clean residual process and TCM state:

```bash
pkill -x llama-server || true
sleep 1
pgrep -x llama-server && pkill -9 -x llama-server || true
spacemit-tcm-smi -c
```

For each lane, `LlamaServerManager` generates a temporary `config.json` under a per-port temporary directory. It must not modify the model directory's original `config.json`. Each server is started with `--smt-config-dir` pointing at its temporary config directory, `--media-backend smt`, `--slot-save-path /tmp`, and `--warmup`.

After starting each process, the manager polls `http://127.0.0.1:<port>/health` until it returns `"status":"ok"` or times out. The model load state may switch to loaded only after all lane servers are healthy. If any lane times out, the manager stops all already-started processes and reports load failure.

## K3/A100 Configuration

A100 binding is performed by sherpa-onnx and `llama-server`/SMT backends. This Python application must produce the correct per-lane configuration and environment.

Default A100 CPU set:

```text
8,9,10,11,12,13,14,15
```

Lane allocation:

| Lanes | Core allocation |
| --- | --- |
| 2 | `8,9,10,11` / `12,13,14,15` |
| 4 | `8,9` / `10,11` / `12,13` / `14,15` |
| 6 | `10` / `11` / `12` / `13` / `14` / `15` |
| 8 | `8` / `9` / `10` / `11` / `12` / `13` / `14` / `15` |

For each `llama-server`:

- `SPACEMIT_PERFER_CORE_ID` uses comma-separated cores, for example `8,9`.
- `llama-server -t` and `-tb` equal the number of assigned cores.
- SMT ONNX `ep_config.SPACEMIT_EP_INTRA_THREAD_NUM` equals the number of assigned cores.
- SMT ONNX `ep_config.SPACEMIT_EP_INTER_THREAD_NUM` is `1`.
- SMT ONNX `ep_config.SPACEMIT_EP_INTRA_THREAD_AFFINITY` uses semicolon-separated cores, for example `8;9`.

## Qwen3-ASR Request, Cleanup, and Parsing

The offline client request flow:

1. Encode endpoint PCM float32 mono 16 kHz audio as mono 16 kHz WAV bytes.
2. Base64-encode the WAV bytes.
3. POST to `http://127.0.0.1:<port>/v1/chat/completions`.
4. If the request was sent, POST to `http://127.0.0.1:<port>/slots/0?action=erase` after the request completes or fails.
5. Parse the response text.
6. Emit `final` on success or a lane error event on hard failure.

Default request timeout is 300000 ms. Slot cleanup timeout is 30000 ms.

Request body:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a speech recognition model."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64-wav>",
            "format": "wav"
          }
        },
        {
          "type": "text",
          "text": ""
        }
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "id_slot": 0
}
```

Hard failures:

- HTTP connection failure.
- HTTP non-2xx response.
- Request timeout.
- JSON parse failure.
- Missing recognizable text field.
- Slot cleanup failure.

Response parsing:

- Prefer `choices[0].message.content`.
- Fall back to top-level `content`.
- Fall back to top-level `text`.
- If `<asr_text>` exists, remove everything before and including that tag.
- If `</asr_text>` exists, remove that tag and everything after it.
- Strip leading and trailing whitespace.
- Empty cleaned text is valid and must be emitted as `final`.

Stop recognition and model unload must perform best-effort slot cleanup for each started `llama-server`.

## Events and WebSocket

All user-visible and headless-visible updates use one event model. Pipelines never write directly to WebSocket connections. They publish into a global `FrontendEventBus`, and a single async broadcaster sends events to all connected clients. Slow browsers, broken sockets, or network stalls must not block audio input or ASR inference.

ASR result event shape:

```json
{
  "type": "asr_result",
  "stream_id": 0,
  "sentence_id": 12,
  "revision": 3,
  "status": "partial",
  "result_type": "online",
  "is_final": false,
  "text": "当前流式识别文本",
  "timestamp_ms": 123456
}
```

Statuses:

- `partial`: streaming online intermediate result.
- `pending`: endpointed online sentence awaiting offline correction.
- `final`: offline corrected sentence result.
- `fatal_error`: fatal lane or global error.
- `lane_finished`: one lane input has ended.
- `stats_updated`: metrics update.

The UI and headless state store locate text by `stream_id + sentence_id`. Each result carries a lane-local `revision`; consumers accept only newer revisions for the same sentence, preventing out-of-order messages from overwriting newer text. Offline `final` uses a higher revision than the corresponding `pending` event.

If the frontend event queue backs up, the event bus may merge replaceable `partial` events for the same lane and sentence, keeping only the latest revision. It must not drop `pending`, `final`, `fatal_error`, `lane_finished`, or `stats_updated` events.

## Web UI

The FastAPI service serves a static single-page HTML UI. The page should not scroll as a whole. The top control bar includes:

- Lane count selector: 2, 4 default, 6, 8.
- Load and unload model buttons.
- Start and stop recognition buttons.

State behavior:

- Recognition requires loaded models.
- Loading disables unload/start/stop controls until it completes.
- Loaded model configuration is disabled.
- Recognition start clears previous recognition display and metrics.
- Running recognition disables model and input configuration.
- Stop terminates audio streams and enters the normal stop flow.
- Unload requires recognition to be stopped first.

Display layout by lane count:

- 2 lanes: `2x1`
- 4 lanes: `2x2`
- 6 lanes: `3x2`
- 8 lanes: `4x2`

Each lane panel shows input configuration and recognition text. Unselected configuration controls are hidden. Online `partial` text appears in gray. `pending` appears as the endpointed online text awaiting correction. Offline `final` appears in black and replaces the corresponding online/pending text for the same sentence.

## Headless Mode

Headless mode is for remote-board validation and batch runs. It does not open a browser, but it still starts the same service objects:

- FastAPI/uvicorn service.
- `FrontendEventBus`.
- `PipelineManager`.
- `LlamaServerManager`.

Headless mode uses internal control calls to simulate frontend operations:

1. Build app services.
2. Resolve lane count and local file inputs.
3. Validate all input files.
4. Load models and wait for all `llama-server` health checks.
5. Start recognition.
6. Wait for all local files to finish and all submitted offline tasks to complete.
7. Stop recognition.
8. Save final results, configuration, logs, and metrics.
9. Release online model, clean slots, stop `llama-server`, close event bus, and shut down Web service.

Default CLI:

```bash
python -m multi_stream_asr.main --headless
python -m multi_stream_asr.main --headless --lanes 8 --input-dir ~/Music/wav16k
```

Headless defaults:

- Lane count: 4.
- Input source: local files only.
- Input directory: `~/Music/wav16k`.
- File assignment: sorted first N files, cycling if needed.

If no browser is connected, `FrontendEventBus` still consumes events and updates the internal result state. Logs and saved results come from internal state, not from DOM state.

## Lifecycle and Error Handling

`PipelineManager` serializes lifecycle operations:

- `load_models()`: start online model resources and per-lane `llama-server` processes; transition to loaded only when all health checks pass.
- `start_recognition()`: clear old state, initialize inputs, start producers, online consumers, offline workers, metrics, and event processing.
- `stop_recognition()`: stop producers, send finish frames, wait for online reset and submitted offline tasks, save results, and mark lanes finished/stopped.
- `unload_models()`: best-effort slot cleanup, release online resources, stop offline servers, and clear model state.

Normal completion requires:

- All input streams ended.
- All submitted offline tasks completed.
- All pipelines and offline workers stopped.
- Final results, configuration, and performance statistics saved.
- Online model released and all `llama-server` processes stopped.

Fatal completion:

- Any lane fatal error triggers global stop.
- Producers stop reading new audio.
- Current unsubmitted segment caches may be discarded.
- Already-sent Qwen3-ASR requests are allowed to return or time out.
- Final logs include error source and collected metrics.

## Metrics and Logging

Per lane metrics:

- Processed audio frame count.
- Dropped frame count.
- Online average frame processing latency.
- Offline request count.
- Offline average request latency.
- Offline RTF.
- Final lane state: `finished`, `stopped`, or `fatal`.

Logging requirements:

- Queue full and online frame drop warnings.
- Offline request timeout warnings/errors.
- Slot cleanup failure errors.
- Online per-frame latency in verbose mode.
- Offline request latency and RTF in verbose mode.
- Final per-lane recognition results under `results/lane<N>.txt`, config, metrics, model-output diagnostics under `model_outputs/lane<N>.jsonl`, and final state.

## Testing and Validation

pytest is required despite the original design note saying tests were not needed. Tests are part of the accepted scope for this spec.

Unit tests must cover:

- Default config values, including lane count, model paths, endpoint rules, ports, timeouts, and frame duration.
- Lane count validation for only 2, 4, 6, and 8 lanes.
- A100 core allocation and environment/config rendering for each lane count.
- `llama-server` temporary config generation without modifying model directory config.
- `llama-server` command construction and health-check success/failure behavior.
- Qwen3-ASR request body construction, WAV/base64 encoding, response parsing, empty final handling, hard failure handling, and slot cleanup.
- Local WAV validation for 16 kHz mono PCM.
- Segment cache capacity and reset behavior.
- Event bus ordering, partial merging, and non-dropping of pending/final/error events.
- State machine legal and illegal transitions.
- Pipeline endpoint behavior for empty online text, non-empty pending submission, final replacement, and finish frames.

Integration smoke tests must run with fake audio, fake online recognizer, and fake offline recognizer. They must prove that headless mode can load, start, emit partial/pending/final events, finish all lanes, save output state, and unload without real model dependencies.

Final real-device validation must be run after all development is complete:

```bash
. .venv313/bin/activate
python -m multi_stream_asr.main --headless --lanes 4 --input-dir ~/Music/wav16k
```

Expected final validation:

- The command completes without fatal errors.
- All four local WAV lanes finish.
- All submitted offline tasks complete.
- Final recognition results, config, per-lane model-output diagnostics, logs, and metrics are saved.
- All `llama-server` processes started by the app are stopped.
- No zombie `llama-server` remains.
- Ideal online average frame processing latency is approximately 50 ms.
- Ideal offline RTF is approximately 0.4.

The performance values are targets, not unit-test thresholds. They should be reported from final headless metrics and used to guide tuning.

## Open Decisions Resolved

- Tests are required: use pytest for unit and fake-adapter integration coverage.
- `pending` is a frontend-visible and headless-visible event and must refresh the displayed/internal sentence state.
- A100 affinity is not implemented directly by project Python threads; the project configures sherpa-onnx and `llama-server`/SMT correctly.
- Default zipformer endpoint values are rule1 1.0s, rule2 0.5s, and rule3 10.0s.
