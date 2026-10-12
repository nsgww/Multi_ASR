# Looping Local Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded-memory looping for default/local WAV inputs while preserving the existing Zipformer and Qwen workflow.

**Architecture:** Add a `loop_audio` flag to lane input configuration and IPC commands. `LocalWavSource` reuses one prepared PCM buffer and emits monotonic frames until stopped. Loop mode disables result/model-output content writes and activates bounded UI/event retention; metrics use exact streaming aggregates.

**Tech Stack:** Python 3.13, dataclasses, asyncio, NumPy, FastAPI/Pydantic, browser JavaScript, pytest.

## Global Constraints

- Looping applies to `Default files` and `Local WAV`, never to microphones.
- Loop mode ends only after a manual stop.
- Zipformer and Qwen request order, segmentation, scheduling, and inference calls remain unchanged.
- Metrics remain full-run exact summaries with `count`, `total`, `average`, and `max`.
- Existing non-loop behavior and artifact contents remain unchanged.

---

### Task 1: Add Loop Configuration and Source Behavior

**Files:**
- Modify: `src/multi_stream_asr/config.py`
- Modify: `src/multi_stream_asr/audio/local_file.py`
- Modify: `src/multi_stream_asr/pipeline/process_manager.py`
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py`
- Test: `tests/test_config.py`
- Test: `tests/test_local_file_audio.py`
- Test: `tests/test_process_manager.py`
- Test: `tests/test_lane_runtime.py`

- [ ] Add failing tests for `loop_audio` on file inputs, rejection on microphones, two complete loops with monotonic frame/timestamp values, manual stop, and empty-file rejection.
- [ ] Run the focused tests and confirm failure comes from the missing field/behavior.
- [ ] Add `loop_audio: bool = False` to `LaneInputConfig`, validate it only for file kinds, serialize it into start commands, and pass it into `LocalWavSource`.
- [ ] Implement `LocalWavSource.frames()` as an outer loop over the existing prepared `_samples`; emit `AudioFrame.finish()` only when looping is disabled or stop is requested. Reject zero-sample looping in `prepare()`.
- [ ] Run focused tests and confirm non-loop tests and new loop tests pass.

### Task 2: Stream Metrics Without Changing Inference

**Files:**
- Modify: `src/multi_stream_asr/log/metrics.py`
- Test: `tests/test_buffers_metrics.py`
- Test: `tests/test_pipeline.py`

- [ ] Add failing tests that add many values and assert exact full-run summaries while tracker storage stays bounded and no inference call sequence changes.
- [ ] Run focused metrics tests and confirm failure against list-based storage.
- [ ] Replace raw sample accumulation with locked `count`, compensated `total`, and `max` state; keep the existing summary schema and constructor compatibility where tests or callers provide initial samples.
- [ ] Apply the same constant-memory accumulator per shared timing field.
- [ ] Run the metrics and pipeline tests, verifying average equals total/count and max covers all observed values.

### Task 3: Loop Artifact Policy and Bounded Display Retention

**Files:**
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py`
- Modify: `src/multi_stream_asr/pipeline/process_manager.py`
- Modify: `src/multi_stream_asr/web/event_bus.py`
- Modify: `src/multi_stream_asr/web/static/index.html`
- Modify: `src/multi_stream_asr/web/server.py`
- Test: `tests/test_process_manager.py`
- Test: `tests/test_web_server.py`

- [ ] Add failing tests that loop mode sends no result/model-output content, still emits metrics, retains at most 100 events per lane in the process event bus, and retains at most 100 rendered lane events in the browser contract.
- [ ] Run focused tests and confirm failure.
- [ ] Add loop metadata to the start command and keep artifact writers empty in loop mode while preserving their lifecycle/path protocol.
- [ ] Add a bounded result retention mode used for loop runs; leave non-loop artifact and event behavior unchanged.
- [ ] Prune frontend event/revision maps per lane to the latest 100 sentence keys while preserving current scrolling and status rendering.
- [ ] Run focused tests and verify the result files remain empty for loop mode while metrics are still produced.

### Task 4: Frontend Controls and End-to-End Stop Flow

**Files:**
- Modify: `src/multi_stream_asr/web/static/index.html`
- Modify: `src/multi_stream_asr/web/server.py`
- Test: `tests/test_web_server.py`
- Test: `tests/test_manager_headless.py`

- [ ] Add failing HTML/API tests for a loop checkbox on `default_files` and `local_wav`, hidden checkbox on microphone, payload round-trip, and manual stop returning to `loaded`.
- [ ] Run focused tests and confirm failure.
- [ ] Render the checkbox beside file input controls, clear it when switching to microphone, include it in lane input payloads, and keep it editable until recognition starts.
- [ ] Ensure the start/stop handlers retain current lifecycle behavior and do not add a natural-completion path for loop sources.
- [ ] Run Web and manager tests, then the complete local suite and JavaScript syntax check.

### Task 5: Real Validation

**Files:**
- No source changes.

- [ ] Commit the implementation after local verification.
- [ ] Pull it into the K3 board using the project `.venv313` environment and run the complete test suite.
- [ ] Load models, select a local WAV with loop enabled on all file lanes, run for at least 10 minutes, and verify the service remains running, memory is stable, and no natural finish occurs.
- [ ] Manually stop, verify the service returns to `loaded`, and inspect `metrics.json`, `config.json`, empty result/model-output files, processed/dropped frames, and event-loop lag.
