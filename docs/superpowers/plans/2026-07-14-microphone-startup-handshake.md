# Microphone Startup Handshake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serialize only microphone stream initialization using a real input-ready handshake while preserving parallel Zipformer loading, warmup, and recognition.

**Architecture:** The audio producer reports readiness immediately after `source.prepare()`. The lane runtime emits `lane_started` from its event-loop thread, and the process manager waits for each microphone lane before starting the next microphone; non-microphone lanes remain concurrent.

**Tech Stack:** Python 3.13, asyncio, multiprocessing, pytest, sounddevice/PortAudio.

## Global Constraints

- Do not serialize model loading or Zipformer warmup.
- Do not use fixed delays or device-specific conditions.
- Only microphone initialization is serialized.
- Preserve current fatal cleanup and repeat-run behavior.

---

### Task 1: Make lane readiness reflect successful source preparation

**Files:**
- Modify: `src/multi_stream_asr/pipeline/pipeline.py`
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_lane_runtime.py`

**Interfaces:**
- `LanePipeline.run(source, *, on_source_ready=None)` invokes the callback after `source.prepare()`.
- `LaneProcessRuntime` emits `lane_started` only after receiving that callback on its asyncio loop.

- [ ] Add failing tests for callback ordering and prepare failure.
- [ ] Run focused tests and confirm the new assertions fail.
- [ ] Add the readiness callback and move `lane_started` behind it.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Serialize microphone starts in the process manager

**Files:**
- Modify: `src/multi_stream_asr/pipeline/process_manager.py`
- Test: `tests/test_process_manager.py`

**Interfaces:**
- `ProcessPipelineManager._started` records `lane_started` lifecycle events.
- `start_recognition()` sends all non-microphone starts without waiting, then starts microphone lanes one at a time and waits for each readiness event.

- [ ] Add a failing test that blocks the second microphone until the first reports ready while file lanes start immediately.
- [ ] Run the focused test and confirm failure.
- [ ] Implement `_started` lifecycle tracking and microphone sequencing.
- [ ] Run process-manager tests and confirm they pass.

### Task 3: Regression and K3 validation

**Files:**
- Modify only if test-discovered defects require a scoped fix.

- [ ] Run the complete local test suite under `.venv313`.
- [ ] Synchronize the committed branch to K3.
- [ ] Run repeated device 15/16 service start/stop cycles and inspect service/kernel logs.
- [ ] Restore the service to `loaded` and record the measured result.
