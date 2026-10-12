# Hotplug Microphone Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve each lane's physical microphone selection while PortAudio indices change after hotplug.

**Architecture:** The audio-device module refreshes PortAudio and exposes a stable `name, Host API` identity. The web client keeps one current device snapshot, reconciles every microphone lane by identity, and submits both the stable identity and current compatibility index.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, sounddevice/PortAudio, browser JavaScript, pytest.

## Global Constraints

- Existing numeric-only lane input configurations remain supported.
- Device refresh must not affect lane-process microphone streams.
- Missing selected devices block recognition before a run is created.

---

### Task 1: Refreshable Stable Device Inventory

**Files:**
- Modify: `src/multi_stream_asr/audio/device.py`
- Test: `tests/test_local_file_audio.py`

- [ ] Add failing tests that require balanced PortAudio terminate/initialize calls and an `id` containing device name and Host API.
- [ ] Run `python -m pytest -q tests/test_local_file_audio.py` and confirm the new tests fail.
- [ ] Implement locked PortAudio refresh, stable inventory IDs, and resolution by ID or legacy index.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Stable Identity Through Configuration and Start

**Files:**
- Modify: `src/multi_stream_asr/config.py`
- Modify: `src/multi_stream_asr/web/server.py`
- Modify: `src/multi_stream_asr/pipeline/process_manager.py`
- Test: `tests/test_config.py`
- Test: `tests/test_process_manager.py`
- Test: `tests/test_web_server.py`

- [ ] Add failing tests for `microphone_device_id`, stable-ID priority, missing-device failure before start, and no-store response headers.
- [ ] Run the focused tests and confirm the new assertions fail.
- [ ] Add the stable field to API/config status and resolve it before entering `STARTING`; preserve numeric fallback.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Reconcile Every Lane From One Fresh Snapshot

**Files:**
- Modify: `src/multi_stream_asr/web/static/index.html`
- Test: `tests/test_web_server.py`

- [ ] Add failing HTML contract tests for no-store fetch, stable option values, global reconciliation, index synchronization, and unavailable-device blocking.
- [ ] Run the focused tests and confirm the new assertions fail.
- [ ] Replace per-lane device responses with one latest-generation snapshot applied to every microphone lane.
- [ ] Run Web tests and the complete suite.

### Task 4: Board Acceptance

**Files:**
- No source changes.

- [ ] Commit and push the tested implementation.
- [ ] Pull it on the K3 board and run the Python 3.13 test suite from `.venv313`.
- [ ] Exercise device refresh and four-microphone recognition; verify all lanes process frames, report zero device-query failures, and return to `loaded` after stop.
