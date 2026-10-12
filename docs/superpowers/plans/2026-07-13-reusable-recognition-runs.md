# Reusable Recognition Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep lane models reusable across completed or stopped runs while allowing audio inputs to change between runs without unloading models.

**Architecture:** Lane processes become long-lived command workers. Zipformer, its worker thread, and the Qwen client survive each run; the source, pipeline, metrics, and file writers are recreated per start. The manager waits for every lane's per-run `lane_finished` event and returns to `loaded`; the web app independently updates pending lane inputs only in that state.

**Tech Stack:** Python 3.13, asyncio, multiprocessing spawn, FastAPI, Pydantic, pytest, browser JavaScript.

## Global Constraints

- Keep process-per-lane model isolation and the single shared Qwen3 server.
- Normal lifecycle is `loaded -> starting -> running -> stopping -> loaded`.
- Lane count, offline model ID, and model directory are editable only in `unloaded`.
- Lane inputs are editable only in `unloaded` and `loaded`; validate local WAV readability and 16 kHz mono PCM before accepting them.
- Preserve results until the next successful start.
- Do not modify or commit the user-owned untracked `TODO` file.

---

## File Structure

- `src/multi_stream_asr/pipeline/state.py`: legal lifecycle transition.
- `src/multi_stream_asr/pipeline/lane_runtime.py`: reusable per-lane command loop and run cleanup.
- `src/multi_stream_asr/pipeline/process_manager.py`: run completion barrier and pending inputs.
- `src/multi_stream_asr/web/server.py`: lane-input API and run recording after natural completion.
- `src/multi_stream_asr/web/static/index.html`: separate control-locking rules.
- `tests/test_models_state.py`, `tests/test_lane_runtime.py`, `tests/test_process_manager.py`, `tests/test_web_server.py`: regression coverage.

## Task 1: Define The Reusable Idle State

**Files:**
- Modify: `src/multi_stream_asr/pipeline/state.py:18-27`
- Modify: `src/multi_stream_asr/pipeline/manager.py:312-334`
- Modify: `tests/test_models_state.py:59-67`
- Modify: `tests/test_manager_headless.py:272-339`

**Interfaces:**
- Consumes: `assert_transition(current: PipelineState, target: PipelineState) -> None`.
- Produces: `assert_transition(PipelineState.STOPPING, PipelineState.LOADED)`.

- [ ] **Step 1: Write failing transition tests**

```python
def test_loaded_run_returns_to_loaded() -> None:
    assert_transition(PipelineState.LOADED, PipelineState.STARTING)
    assert_transition(PipelineState.STARTING, PipelineState.RUNNING)
    assert_transition(PipelineState.RUNNING, PipelineState.STOPPING)
    assert_transition(PipelineState.STOPPING, PipelineState.LOADED)
```

Change in-process-manager normal stop/completion assertions from `STOPPED` to `LOADED`.

- [ ] **Step 2: Verify the test fails**

Run: `pytest tests/test_models_state.py -v`

Expected: FAIL because `stopping -> loaded` is illegal.

- [ ] **Step 3: Implement the transition**

```python
PipelineState.STOPPING: {PipelineState.LOADED, PipelineState.STOPPED, PipelineState.FATAL},
```

In `PipelineManager._wait_for_run_completion()`, use `STOPPING -> LOADED` for normal runs. Retain `STOPPED` only for compatibility until remaining non-process callers are migrated; the process manager must not enter it.

- [ ] **Step 4: Verify focused tests**

Run: `pytest tests/test_models_state.py tests/test_manager_headless.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/pipeline/state.py src/multi_stream_asr/pipeline/manager.py tests/test_models_state.py tests/test_manager_headless.py && git commit -m "feat: return completed runs to loaded state"
```

## Task 2: Make Lane Runtime Reusable

**Files:**
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py:195-355, 595-625`
- Test: `tests/test_lane_runtime.py`

**Interfaces:**
- Consumes: command dictionaries with `type` in `start`, `stop`, `warmup`, `shutdown`.
- Produces: `lane_finished` exactly once per completed or stopped run; `lane_stopped` only on `shutdown`.

- [ ] **Step 1: Write failing two-run runtime tests**

Queue `warmup`, two start commands with distinct run directories, then `shutdown`. Assert two finishes, one shutdown, and persistent recognizer identity:

```python
assert lifecycle_states == [
    "lane_loading", "lane_model_loaded", "lane_loaded",
    "lane_started", "lane_finished",
    "lane_started", "lane_finished", "lane_stopped",
]
assert first_recognizer is runtime._online_recognizer
```

Add a second test that sends `stop` during the first run, verifies a `lane_finished`, then sends a second start before shutdown.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_lane_runtime.py -v`

Expected: FAIL because `LaneRuntime.run()` shuts down after one start.

- [ ] **Step 3: Implement the persistent command loop**

Replace the one-shot portion of `run()` with:

```python
command = await self._load()
while True:
    command = command or await self._wait_for_command({"start", "stop", "shutdown"})
    if command["type"] == "shutdown":
        await self._shutdown()
        return 0
    if command["type"] == "stop":
        command = None
        continue
    await self._run_started(command)
    command = None
```

Before `_run_started()`, reset only run-scoped fields: source, pipeline, stop flag, terminal metrics, artifact flag, latency/RTF/shared-timing/event-loop-lag trackers, and writer references. Do not replace or close the recognizers, online worker, or Qwen client.

After natural completion or a user stop, drain the offline tail, send final metrics and artifact paths, close the source/writers, and send `lane_finished`. Preserve `lane_failed` and fatal shutdown behavior for exceptions. Keep `lane_stopped` in `_shutdown()` only.

- [ ] **Step 4: Verify runtime tests**

Run: `pytest tests/test_lane_runtime.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/pipeline/lane_runtime.py tests/test_lane_runtime.py && git commit -m "feat: reuse lane models across recognition runs"
```

## Task 3: Finalize A Run Without Killing Lane Processes

**Files:**
- Modify: `src/multi_stream_asr/pipeline/process_manager.py:183-502, 650-740`
- Modify: `tests/test_process_manager.py:35-115, 780-870, 1320-1375`

**Interfaces:**
- Consumes: `lane_finished`, `metrics_final`, and `paths` IPC messages.
- Produces: `ProcessPipelineManager.set_lane_inputs(lane_inputs: tuple[LaneInputConfig, ...]) -> None`; manager state `loaded` after all lanes finish.

- [ ] **Step 1: Write failing manager tests**

Extend `_ScriptedController.send()` so start scripts may emit final metrics, paths, and `lane_finished` while remaining alive. Add:

```python
await manager.load_models()
await manager.start_recognition()
await manager.wait_for_completion()
assert manager.state is PipelineState.LOADED
assert all(controller.is_alive() for controller in _ScriptedController.instances)

await manager.start_recognition()
await manager.wait_for_completion()
assert all(sum(c["type"] == "start" for c in controller.commands) == 2
           for controller in _ScriptedController.instances)
```

Add a stop test: `stop_recognition()` waits for all `lane_finished` messages, returns loaded, and sends no `shutdown`. Add a pending-input test that calls `set_lane_inputs()` in `loaded`, starts, and confirms local-WAV and microphone fields in start commands. Assert wrong lane count and active-state calls raise `ValueError`.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_process_manager.py -v`

Expected: FAIL because completion waits for process exits and `set_lane_inputs` does not exist.

- [ ] **Step 3: Implement the completion barrier and pending inputs**

Initialize `self._lane_inputs = config.lane_inputs`. Add:

```python
def set_lane_inputs(self, lane_inputs: tuple[LaneInputConfig, ...]) -> None:
    if self._state is not PipelineState.LOADED:
        raise ValueError(f"cannot update lane inputs from {self._state.value}")
    if len(lane_inputs) != self._config.lanes:
        raise ValueError("lane_inputs must match configured lanes")
    self._lane_inputs = lane_inputs
```

Use `_lane_inputs` in `_resolve_input_commands()`, retaining default-file behavior when it is empty. At start clear `_finished`, final-metric state, summaries, paths, run metrics, and completion-task state. Make `_monitor_run_completion()` wait for every lane's `lane_finished` plus final metrics and artifact paths, without reader/process teardown; its normal path is `running -> stopping -> loaded`.

Make `stop_recognition()` send `stop` and await the completion task, but never call `_join_or_escalate()`. Retain joining readers/processes and shared-Qwen shutdown for unload and fatal paths only.

- [ ] **Step 4: Verify manager tests**

Run: `pytest tests/test_process_manager.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/pipeline/process_manager.py tests/test_process_manager.py && git commit -m "feat: retain loaded lane processes after a run"
```

## Task 4: Add Independent Lane Input Updates To The Web API

**Files:**
- Modify: `src/multi_stream_asr/web/server.py:55-105, 210-350, 430-480`
- Modify: `tests/test_web_server.py:60-105, 620-710, 1010-1320`

**Interfaces:**
- Consumes: `POST /api/lane-inputs` with `{"lane_inputs": [LaneInputRequest, ...]}`.
- Produces: updated status payload; HTTP 422 for invalid inputs; HTTP 409 outside `loaded`; once-only process-run recording after completion.

- [ ] **Step 1: Write failing API tests**

Extend `RecordingManager` with `set_lane_inputs()`. Add a valid request test:

```python
response = client.post("/api/lane-inputs", json={"lane_inputs": [
    {"kind": "local_wav", "local_wav_path": str(valid_wav)},
    {"kind": "microphone", "microphone_device": 2},
]})
assert response.status_code == 200
assert response.json()["lane_inputs"][0]["local_wav_path"] == str(valid_wav)
```

Assert invalid WAV returns 422 without mutating app config, and `running` returns 409. Update stop endpoint expectations to `loaded`. Add a natural-completion recorder test with a completed `run_dir` and lane summaries, and assert exactly one `save_process_run()` call.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_web_server.py -v`

Expected: FAIL because the route and completion monitor do not exist.

- [ ] **Step 3: Implement request conversion, API, and recording monitor**

Add:

```python
class LaneInputsRequest(BaseModel):
    lane_inputs: list[LaneInputRequest]
```

Extract conversion/validation of lane-input requests into a helper used by model load and this route. It requires exact lane count and calls `validate_load_paths(replace(config, lane_inputs=lane_inputs))` before any mutation.

Implement `POST /api/lane-inputs`: require `PipelineState.LOADED`, validate, call `manager.set_lane_inputs(lane_inputs)`, then atomically `replace()` both application configs with the new input tuple.

After successful start, create one tracked task that awaits `manager.wait_for_completion()` and calls `_record_results(app)` only if the manager ended in `loaded`. Cancel/await it during app shutdown. The stop route may also record after it awaits completion because `recorded_process_run_dirs` makes this idempotent.

- [ ] **Step 4: Verify web tests**

Run: `pytest tests/test_web_server.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/web/server.py tests/test_web_server.py && git commit -m "feat: update lane inputs without reloading models"
```

## Task 5: Separate Frontend Model And Input Locks

**Files:**
- Modify: `src/multi_stream_asr/web/static/index.html:360-790`
- Modify: `tests/test_web_server.py:820-900`

**Interfaces:**
- Consumes: loaded status, `POST /api/lane-inputs`, current lane-input draft.
- Produces: model controls editable only unloaded; lane-input controls editable unloaded/loaded; results cleared only after successful start.

- [ ] **Step 1: Write failing served-HTML contract tests**

```python
assert 'const modelEditable = appState.state === "unloaded";' in response.text
assert 'const laneInputsEditable = ["unloaded", "loaded"].includes(appState.state);' in response.text
assert 'await postJson("/api/lane-inputs", {' in response.text
assert 'startButton.disabled = appState.state !== "loaded";' in response.text
```

Keep the existing ordering assertion that `resetRecognitionView()` follows the successful `/api/recognition/start` call. Assert the former `loaded", "stopped` start rule is absent.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_web_server.py -v -k 'index_html'`

Expected: FAIL because the page uses one `editable` flag.

- [ ] **Step 3: Implement separate availability and input persistence**

Use:

```javascript
const modelEditable = appState.state === "unloaded";
const laneInputsEditable = ["unloaded", "loaded"].includes(appState.state);
```

Apply `modelEditable` to lane count, model type, model directory, and load controls. Apply `laneInputsEditable` to lane input type, microphone selector, local WAV picker, and WAV-dialog buttons. The model-directory dialog remains disabled unless `modelEditable` is true.

When a lane input changes while loaded, post the complete draft to `/api/lane-inputs`. On failure restore the previous draft and show the server error. Do not post while unloaded because load includes the draft. Enable Start only in loaded, Stop only in starting/running, and Unload in loaded/fatal. Keep the current snapshot/restore start behavior unchanged.

- [ ] **Step 4: Verify frontend tests**

Run: `pytest tests/test_web_server.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/web/static/index.html tests/test_web_server.py && git commit -m "feat: unlock lane inputs between recognition runs"
```

## Task 6: Verify The Full Reusable Lifecycle

**Files:**
- Modify: `tests/test_process_manager.py` only if a complete two-run integration assertion is missing.
- Modify: `tests/test_web_server.py` only if a complete web lifecycle assertion is missing.

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: a verified two-run lifecycle with no second model load and two saved run directories.

- [ ] **Step 1: Add a full fake lifecycle test if not already covered**

Cover: load, input update, natural four-lane completion, loaded plus one saved run, second input update, start, manual stop after four lane finishes, loaded plus two saved runs, and one model load:

```python
assert manager.calls == [
    "load_models", "set_lane_inputs", "start_recognition",
    "set_lane_inputs", "start_recognition", "stop_recognition",
]
assert len(recorder.saved_runs) == 2
```

- [ ] **Step 2: Run the focused lifecycle test**

Run: `pytest tests/test_web_server.py -v -k reusable`

Expected: PASS.

- [ ] **Step 3: Run all local tests**

Run: `pytest -q`

Expected: PASS with no collection errors.

- [ ] **Step 4: Run board fake-mode validation**

Follow `~/.codex/docs/k3/remote-device-python.md`; activate the remote project `.venv313` with Python 3.13, then run:

```bash
python -m multi_stream_asr.main --headless --fake --lanes 4
```

Expected: exit code 0, four result files under one run directory, and no remaining lane or shared-Qwen processes after headless shutdown.

- [ ] **Step 5: Commit any final regression-only changes**

```bash
git add tests/test_process_manager.py tests/test_web_server.py && git commit -m "test: cover reusable recognition run lifecycle"
```

Do not create this commit if this task required no changes.

