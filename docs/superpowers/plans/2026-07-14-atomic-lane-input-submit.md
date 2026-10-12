# Atomic Lane Input Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple model loading from audio input selection and submit all Lane inputs exactly once immediately before recognition starts.

**Architecture:** The browser owns an editable Lane-input draft whenever recognition is inactive. Model loading sends only topology and model fields; Start sends the complete draft to the main service, which validates and installs it before starting the already-loaded Lane processes. Input preflight runs before the manager enters `starting`, so failures leave the service in `loaded`.

**Tech Stack:** Python 3.13, FastAPI, asyncio, pytest, browser JavaScript.

## Global Constraints

- Use project-local `.venv313` for Python commands.
- Do not change Zipformer or Qwen model loading and warmup concurrency.
- `starting`, `running`, and `stopping` are the only states that lock Lane-input controls.
- Every transition of an input type to `microphone` must issue a fresh `GET /api/audio-devices` request.
- Input validation failure must not create a run or send a start command to any Lane.
- Preserve untracked workspace files.

## File Map

- Modify `src/multi_stream_asr/web/server.py`: separate model and Lane-input validation; keep model-load requests independent of Lane inputs.
- Modify `src/multi_stream_asr/pipeline/process_manager.py`: preflight all input commands before changing lifecycle state.
- Modify `src/multi_stream_asr/web/static/index.html`: maintain the local draft, refresh microphone devices, and submit inputs from Start.
- Modify `tests/test_web_server.py`: cover API validation boundaries and frontend request/state ordering.
- Modify `tests/test_process_manager.py`: cover preflight failures and absence of Lane start commands.

---

### Task 1: Separate Model And Audio Validation

**Files:**
- Modify: `src/multi_stream_asr/web/server.py:61-69,254-283,338-378,470-551`
- Test: `tests/test_web_server.py:145-176,395-651`

**Interfaces:**
- Produces: `validate_model_paths(config: AppConfig) -> dict[str, list[str]]`.
- Produces: `validate_lane_input_paths(config: AppConfig) -> dict[str, list[str]]`.
- Preserves: `POST /api/lane-inputs` accepts one ordered full list and calls `manager.set_lane_inputs(...)` only after validation.

- [ ] **Step 1: Write failing API tests for the new validation boundary**

Replace the tests that expect `/api/models/load` to reject a bad WAV with tests that prove model loading ignores the existing audio draft and `/api/lane-inputs` rejects it later:

```python
def test_model_load_does_not_validate_existing_lane_inputs(tmp_path: Path):
    config = app_config_with_readable_model_files(tmp_path, lanes=2)
    config = replace(
        config,
        lane_inputs=(
            LaneInputConfig(kind="local_wav", local_wav_path=tmp_path / "missing.wav"),
            LaneInputConfig(),
        ),
    )
    manager = RecordingManager()

    with TestClient(create_app(config=config, manager=manager)) as client:
        response = client.post("/api/models/load")

    assert response.status_code == 200
    assert manager.calls == ["load_models"]


def test_lane_input_submit_rejects_invalid_wav_before_updating_manager(tmp_path: Path):
    config = app_config_with_readable_model_files(tmp_path, lanes=2)
    manager = RecordingManager(state=PipelineState.LOADED)
    invalid_wav = tmp_path / "invalid.wav"
    invalid_wav.write_bytes(b"not wav")

    with TestClient(create_app(config=config, manager=manager)) as client:
        response = client.post(
            "/api/lane-inputs",
            json={"lane_inputs": [
                {"kind": "local_wav", "local_wav_path": str(invalid_wav)},
                {"kind": "default_files"},
            ]},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["lane_inputs"] == [str(invalid_wav.resolve())]
    assert manager.calls == []
```

Import `LaneInputConfig` from `multi_stream_asr.config`. Remove `lane_inputs` from model-load request bodies in model-file tests because it is no longer part of that contract.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
source .venv313/bin/activate
pytest tests/test_web_server.py -k 'model_load_does_not_validate_existing_lane_inputs or lane_input_submit_rejects_invalid_wav' -v
```

Expected: the first test fails with HTTP 422 from combined validation; the second test passes or exposes the current combined helper name.

- [ ] **Step 3: Split validators and remove Lane inputs from model-load parsing**

Change `LoadModelsRequest` to contain only `lanes`, `offline_model_id`, and `offline_model_dir`. Remove the `request.lane_inputs` branch from `_config_from_load_request`; preserve existing Lane inputs only when the Lane count is unchanged, otherwise use an empty tuple.

Split validation as follows:

```python
def validate_model_paths(config: AppConfig) -> dict[str, list[str]]:
    errors: dict[str, list[str]] = {}
    model_directory = config.models.offline_model.model_dir
    try:
        validate_readable_directory(model_directory)
    except (OSError, RuntimeError, ValueError):
        errors["model_directory"] = [_error_path(model_directory)]
    else:
        invalid = validate_offline_model_files(config.models.offline_model)
        if invalid:
            errors["model_files"] = [_error_path(path) for path in invalid]
    return errors


def validate_lane_input_paths(config: AppConfig) -> dict[str, list[str]]:
    invalid: list[str] = []
    for stream_id in range(config.lanes):
        lane_input = config.input_for_lane(stream_id)
        if lane_input.kind != "local_wav":
            continue
        assert lane_input.local_wav_path is not None
        try:
            wav_path = validate_readable_file(lane_input.local_wav_path)
            if wav_path.suffix != ".wav":
                raise ValueError(f"not a WAV file: {wav_path}")
            validate_wav_16k_mono_pcm(wav_path)
        except (OSError, RuntimeError, ValueError, EOFError, wave.Error):
            invalid.append(_error_path(lane_input.local_wav_path))
    return {"lane_inputs": invalid} if invalid else {}
```

Use `validate_model_paths` from `/api/models/validate` and `/api/models/load`. Use `validate_lane_input_paths` from `/api/lane-inputs`.

- [ ] **Step 4: Run API validation tests and verify GREEN**

Run:

```bash
pytest tests/test_web_server.py -k 'model_validation or model_load or lane_input or load_rejects' -v
```

Expected: all selected tests pass with model failures still reported under `model_directory` or `model_files`, and audio failures only under `lane_inputs`.

- [ ] **Step 5: Commit the backend validation boundary**

```bash
git add src/multi_stream_asr/web/server.py tests/test_web_server.py
git commit -m "refactor: decouple audio validation from model loading"
```

---

### Task 2: Preflight Inputs Before Starting A Run

**Files:**
- Modify: `src/multi_stream_asr/pipeline/process_manager.py:326-387,516-573`
- Test: `tests/test_process_manager.py:1050-1170`

**Interfaces:**
- Consumes: `ProcessPipelineManager._lane_inputs` installed by `set_lane_inputs(...)`.
- Produces: `_preflight_input_commands() -> dict[int, dict[str, Any]]` or equivalent ordering where microphone validation and file resolution complete before transition to `STARTING`.

- [ ] **Step 1: Write failing tests for Local WAV and default-file preflight**

Add two tests. The Local WAV case can use a missing path; the default case should monkeypatch `select_input_files` to raise. Both managers start in `LOADED`, have recording controllers, and assert no run or command was created:

```python
@pytest.mark.asyncio
async def test_start_rejects_missing_local_wav_before_entering_starting(tmp_path, monkeypatch):
    from multi_stream_asr.pipeline import process_manager

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    manager = ProcessPipelineManager(
        config=AppConfig(
            lanes=2,
            output_dir=tmp_path / "runs",
            lane_inputs=(
                LaneInputConfig(kind="local_wav", local_wav_path=tmp_path / "missing.wav"),
                LaneInputConfig(kind="local_wav", local_wav_path=tmp_path / "missing-2.wav"),
            ),
        ),
        event_bus=FrontendEventBus(),
    )
    manager._state = PipelineState.LOADED

    with pytest.raises(FileNotFoundError):
        await manager.start_recognition()

    assert manager.state is PipelineState.LOADED
    assert manager.run_dir is None
    assert not any(
        command["type"] == "start"
        for controller in _ScriptedController.instances
        for command in controller.commands
    )


@pytest.mark.asyncio
async def test_start_rejects_missing_default_inputs_before_entering_starting(
    tmp_path, monkeypatch
):
    from multi_stream_asr.pipeline import process_manager

    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    monkeypatch.setattr(
        process_manager,
        "select_input_files",
        lambda _input_dir, _lanes: (_ for _ in ()).throw(
            FileNotFoundError("default inputs unavailable")
        ),
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path / "runs"),
        event_bus=FrontendEventBus(),
    )
    manager._state = PipelineState.LOADED

    with pytest.raises(FileNotFoundError, match="default inputs unavailable"):
        await manager.start_recognition()

    assert manager.state is PipelineState.LOADED
    assert manager.run_dir is None
    assert not any(
        command["type"] == "start"
        for controller in _ScriptedController.instances
        for command in controller.commands
    )
```

- [ ] **Step 2: Run the focused manager tests and verify RED**

Run:

```bash
pytest tests/test_process_manager.py -k 'rejects_missing_local_wav or rejects_missing_default' -v
```

Expected: failures show manager state becomes `fatal`, or a run directory is assigned before input resolution fails.

- [ ] **Step 3: Move all input preflight before lifecycle transition**

At the start of the lifecycle lock, validate microphones and resolve every command before calling `assert_transition` or assigning `_run_dir`:

```python
async with self._lifecycle_lock:
    self._validate_microphone_inputs()
    input_commands = self._resolve_input_commands()
    assert_transition(self._state, PipelineState.STARTING)
    self._state = PipelineState.STARTING
    try:
        self._event_bus.clear_results()
        self._reset_for_run()
        self._run_dir = self._new_run_dir()
        # Existing command dispatch uses input_commands.
```

Do not change serialized microphone startup or parallel file-source startup.

- [ ] **Step 4: Run input-start tests and verify GREEN**

Run:

```bash
pytest tests/test_process_manager.py -k 'start_ and (input or microphone or local or default)' -v
```

Expected: all selected tests pass; failed preflight leaves state `loaded`, `run_dir is None`, and controllers receive no start commands.

- [ ] **Step 5: Commit input preflight behavior**

```bash
git add src/multi_stream_asr/pipeline/process_manager.py tests/test_process_manager.py
git commit -m "fix: preflight audio inputs before recognition start"
```

---

### Task 3: Make Lane Inputs A Browser Draft And Submit On Start

**Files:**
- Modify: `src/multi_stream_asr/web/static/index.html:279-329,377-455,534-655,803-855`
- Test: `tests/test_web_server.py:847-1025`

**Interfaces:**
- Consumes: `POST /api/lane-inputs` with `{lane_inputs: LaneInputRequest[]}`.
- Consumes: `GET /api/audio-devices` on every transition to microphone.
- Produces: Start ordering `POST /api/lane-inputs` then `POST /api/recognition/start`.

- [ ] **Step 1: Replace stale structural tests with atomic-submit assertions**

Update the existing frontend tests to extract the Lane change and Start handler bodies. Assert:

```python
assert "laneInputUpdatePromise" not in response.text
assert "persistLaneInputs" not in response.text
assert "appState.microphoneDevices" not in response.text

start_handler = response.text[start_start:start_end]
submit_call = 'await postJson("/api/lane-inputs", {'
start_call = 'await postJson("/api/recognition/start");'
assert submit_call in start_handler
assert "lane_inputs: appState.draft.laneInputs" in start_handler
assert start_handler.index(submit_call) < start_handler.index(start_call)
```

Add assertions that model load does not contain `lane_inputs`, the Lane lock is based only on active recognition states, and `loadMicrophoneDevices` always executes `fetch("/api/audio-devices")` without a cache guard.

Also add a concurrency test proving the main web process serves the device route while a model-load request is pending. Import `threading`, create a manager whose `load_models()` sets a `threading.Event` and awaits `asyncio.to_thread(release.wait)`, issue `/api/models/load` in a background thread, and assert `/api/audio-devices` returns before releasing model load:

```python
def test_audio_devices_responds_while_models_are_loading(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    load_responses = []

    class PausingLoadManager(RecordingManager):
        async def load_models(self) -> None:
            started.set()
            await asyncio.to_thread(release.wait)
            self.state = PipelineState.LOADED

    monkeypatch.setattr(
        web_server,
        "list_microphone_devices",
        lambda: [{"index": 15, "name": "USB microphone"}],
    )
    app = create_app(
        config=app_config_with_readable_model_files(tmp_path, lanes=2),
        manager=PausingLoadManager(),
    )

    with TestClient(app) as client:
        worker = threading.Thread(
            target=lambda: load_responses.append(client.post("/api/models/load"))
        )
        worker.start()
        assert started.wait(timeout=1.0)
        devices = client.get("/api/audio-devices")
        release.set()
        worker.join(timeout=1.0)

    assert devices.status_code == 200
    assert devices.json()["devices"] == [{"index": 15, "name": "USB microphone"}]
    assert load_responses[0].status_code == 200
```

- [ ] **Step 2: Run frontend tests and verify RED**

Run:

```bash
pytest tests/test_web_server.py -k 'index_html' -v
```

Expected: failures show the update promise, immediate persistence, cached microphone list, and model-load Lane payload are still present.

- [ ] **Step 3: Remove immediate persistence and update the local draft synchronously**

Remove `appState.microphoneDevices`, `laneInputUpdatePromise`, `persistLaneInputs`, and `queueLaneInputUpdate`. In each control handler, resolve the current item by `streamId` and mutate it immediately:

```javascript
kind.addEventListener("change", async () => {
  const current = appState.draft.laneInputs[streamId];
  current.kind = kind.value;
  if (current.kind !== "local_wav") current.local_wav_path = null;
  if (current.kind !== "microphone") current.microphone_device = null;
  updateLaneInputVisibility(row, current);
  if (current.kind === "microphone") {
    try {
      await populateMicrophoneDevices(device, current);
      showError("");
    } catch (error) {
      showError(error.message);
    }
  }
});
```

Introduce `updateLaneInputVisibility(row, input)` as a synchronous function that only toggles the Local WAV output, picker, and microphone select. Path and microphone-device handlers must also look up `appState.draft.laneInputs[streamId]` instead of mutating a captured object. `loadMicrophoneDevices()` must fetch every time.

- [ ] **Step 4: Decouple Lane control availability from model loading**

Use separate model and recognition locks:

```javascript
const modelBusy = ["loading", "unloading"].includes(appState.state);
const recognitionActive = ["starting", "running", "stopping"].includes(appState.state);
const modelEditable = appState.state === "unloaded" && !modelBusy;
const laneInputsEditable = !recognitionActive;
```

Apply `modelEditable` to Lane count and model controls. Apply only `laneInputsEditable` to Lane input controls and Local WAV path-picker controls. Keep Start enabled only in `loaded`.

- [ ] **Step 5: Submit the complete draft before Start**

Remove `lane_inputs` from the `/api/models/load` body. At the beginning of Start, disable the button and lock Lane controls for the pending operation, then submit the full draft before starting:

```javascript
appState.state = "starting";
setOperation("Validating audio inputs...");
renderStatus();
await postJson("/api/lane-inputs", {
  lane_inputs: appState.draft.laneInputs,
});
appState.state = "starting";
setOperation("Starting recognition...");
renderStatus();
await postJson("/api/recognition/start");
```

Keep `resetRecognitionView()` after successful Start only. On failure, restore the recognition view, show the validation error, and call `refreshOperationalState()` rather than `loadHealth()` so the server status does not overwrite the independent Lane draft.

- [ ] **Step 6: Run frontend and API tests and verify GREEN**

Run:

```bash
pytest tests/test_web_server.py -v
```

Expected: all web tests pass, including two-microphone/two-default ordering, fresh device-list requests, and editable Lane controls during model loading.

- [ ] **Step 7: Commit the browser behavior**

```bash
git add src/multi_stream_asr/web/static/index.html tests/test_web_server.py
git commit -m "fix: submit lane input draft atomically on start"
```

---

### Task 4: Full Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes the completed backend, manager, and browser changes.
- Produces test evidence and a clean tracked worktree.

- [ ] **Step 1: Run formatting and full local tests**

```bash
source .venv313/bin/activate
git diff --check
pytest -q
```

Expected: `git diff --check` exits 0 and the full suite passes.

- [ ] **Step 2: Verify the final diff is scoped**

```bash
git status --short
git diff HEAD~3 --stat
git log --oneline -4
```

Expected: only the planned source, tests, spec, and plan are tracked; pre-existing untracked files remain untouched.

- [ ] **Step 3: Perform service-level acceptance when the development board is available**

Following `~/.codex/docs/k3/remote-device-python.md`, sync tracked code, use Python 3.13 with `.venv313`, then verify:

1. Start model loading and switch two Lane inputs to microphone while loading; each switch refreshes the device list.
2. After load, configure Lane 1/2 as distinct microphones and Lane 3/4 as default files.
3. Start recognition and inspect the created run config; it must contain exactly two microphone and two default-file inputs in the selected order.
4. Stop recognition, change a Lane input, and start again; the second run must use the new draft.
5. Select an unavailable microphone; Start must return to `loaded` without creating a run or starting any Lane.

- [ ] **Step 4: Record acceptance evidence**

Append the test commands, commit IDs, run directory names, and observed Lane-input arrays to the implementation report or final response. Do not claim board acceptance if the board was unavailable.
