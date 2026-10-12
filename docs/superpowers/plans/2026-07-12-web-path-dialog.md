# Web Path Dialog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Web path text inputs with a native-dialog picker for readable service-host model directories and local WAV files, and reject unreadable or incomplete submitted paths before model loading.

**Architecture:** FastAPI provides a read-only path-browser API plus shared final validation. The existing vanilla page uses native HTML dialog for a folder/file picker. Load requests remain path-based, so selected models are not uploaded or copied.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, pathlib, os.access, pytest, native HTML dialog, vanilla JavaScript and CSS.

## Global Constraints

- Do not add npm, a JavaScript framework, third-party widgets, or a file-upload endpoint.
- Initial browsing opens Path.home(), but navigation has no root restriction.
- Resolve symlinks strictly before validating type or permissions.
- Directories require os.R_OK | os.X_OK; files require os.R_OK.
- Model mode confirms directories. WAV mode confirms readable files ending in .wav.
- Run final server-side validation before manager_factory, Qwen startup, or lane startup.
- Keep configuration editable only in the unloaded state.
- Preserve the user-owned untracked TODO file.

---

## File Structure

- src/multi_stream_asr/config.py: required-Qwen-file readability validation.
- src/multi_stream_asr/web/server.py: browse endpoint and load-path validation.
- src/multi_stream_asr/web/static/index.html: read-only path displays, picker buttons, native dialog, and browser state.
- tests/test_config.py: required-file readability coverage.
- tests/test_web_server.py: endpoint, pre-load rejection, and static Web contract coverage.

### Task 1: Add Readable Service-Host Path Browsing

**Files:** Modify src/multi_stream_asr/web/server.py. Test tests/test_web_server.py.

**Interfaces:**

- Add validate_readable_directory(path: Path) -> Path: expand user, resolve(strict=True), require is_dir and os.access(path, os.R_OK | os.X_OK).
- Add validate_readable_file(path: Path) -> Path: expand user, resolve(strict=True), require is_file and os.access(path, os.R_OK).
- Add browse_path(path: Path | None, selection: Literal['directory', 'wav']) -> dict[str, object].
- Add GET /api/path-browser with path and selection query parameters.

- [ ] **Step 1: Write failing endpoint tests**

Create a temporary home directory with readable child directory, sample.wav, and ignored.txt. Monkeypatch web_server.Path.home to the temporary directory. Assert GET /api/path-browser?selection=wav returns current_path, parent_path, the directory entry, and only the WAV file entry. Add tests that a nonexistent path, file requested as directory, and an os.access-denied directory return 422.

- [ ] **Step 2: Verify RED**

Run .venv313/bin/python -m pytest tests/test_web_server.py -k path_browser -q. Expected: endpoint is missing.

- [ ] **Step 3: Implement helpers and endpoint**

Import os and Literal. browse_path must default to Path.home(), validate the requested current directory, sort directories before WAV files by lowercase name, and skip inaccessible entries. Each response entry has name, path, and kind values; kind is directory or wav. A directory's parent is null at filesystem root. Convert ValueError, OSError, and RuntimeError to HTTP 422.

- [ ] **Step 4: Verify GREEN**

Run .venv313/bin/python -m pytest tests/test_web_server.py -k path_browser -q. Expected: all new tests pass.

- [ ] **Step 5: Commit**

Run git add src/multi_stream_asr/web/server.py tests/test_web_server.py && git commit -m 'feat: browse readable service paths'.

### Task 2: Validate All Submitted Paths Before Model Loading

**Files:** Modify src/multi_stream_asr/config.py and src/multi_stream_asr/web/server.py. Test tests/test_config.py and tests/test_web_server.py.

**Interfaces:**

- Update validate_offline_model_files(spec: OfflineModelSpec) -> list[Path] so missing, non-file, and unreadable required model files are returned.
- Add validate_load_paths(config: AppConfig) -> dict[str, list[str]] in web.server.
- Error keys are model_directory, model_files, and lane_inputs.

- [ ] **Step 1: Write failing validation tests**

Test a Load Models request with an unreadable model directory and assert 422 detail.model_directory contains the resolved directory and RecordingManager.calls is empty. Test a local_wav lane containing a readable but invalid WAV and assert 422 detail.lane_inputs contains its resolved path and manager calls remain empty. Test an unreadable required Qwen file is reported in detail.model_files.

- [ ] **Step 2: Verify RED**

Run .venv313/bin/python -m pytest tests/test_config.py tests/test_web_server.py -k 'unreadable or invalid_explicit_wav' -q. Expected: Load Models either accepts the input or reports only legacy missing-file data.

- [ ] **Step 3: Implement one validation path**

In config.py import os and treat a required file as invalid unless is_file() and os.access(file, os.R_OK). In server.py, validate the offline directory with validate_readable_directory; collect invalid required files; for every local_wav lane validate_readable_file then call validate_wav_16k_mono_pcm. Return only absolute resolved paths. Invoke validate_load_paths(next_config) before the branch that creates a reconfigured manager. Raise HTTPException(status_code=422, detail=errors) when the mapping is nonempty.

Make POST /api/models/validate use the same directory/file validation so its valid result cannot disagree with Load Models. Preserve missing_files only if existing clients require it; otherwise update its tests and frontend to read model_directory plus model_files consistently.

- [ ] **Step 4: Verify GREEN**

Run .venv313/bin/python -m pytest tests/test_config.py tests/test_web_server.py -k 'model_validation or unreadable or explicit_wav or rejects_' -q. Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Run git add src/multi_stream_asr/config.py src/multi_stream_asr/web/server.py tests/test_config.py tests/test_web_server.py && git commit -m 'feat: validate readable paths before model load'.

### Task 3: Replace Text Inputs with Native Dialog Controls

**Files:** Modify src/multi_stream_asr/web/static/index.html. Test tests/test_web_server.py.

**Interfaces:**

- Add dialog id path-picker-dialog with title, close button, Up button, current-path output, entry list, Cancel button, and Select button.
- Add openPathDialog({ selection, value, onSelect }), loadPathEntries(path), renderPathEntries(payload), and closePathDialog() JavaScript functions.
- Replace offline-model-dir input with an output and id offline-model-picker button.
- Replace local-wav-path input with an output and local-wav-picker button.

- [ ] **Step 1: Write a failing static HTML test**

Assert the rendered page contains dialog id path-picker-dialog, offline-model-picker, local-wav-picker, openPathDialog, fetch('/api/path-browser'), showModal(), and close(). Assert the old editable offline-model-dir input is absent, while its value remains represented by a read-only output.

- [ ] **Step 2: Verify RED**

Run .venv313/bin/python -m pytest tests/test_web_server.py -k 'path_dialog or picker' -q. Expected: dialog and picker controls are absent.

- [ ] **Step 3: Add modal markup and CSS**

Use native dialog with a fixed max width and a fixed-height scrollable entry list. Follow current colors and 6px radius. Use icon-labelled buttons for close, Up, and path picker actions; keep text only on Cancel and Select. Do not put the dialog inside a card.

- [ ] **Step 4: Add picker state and event handlers**

Store selection mode, current path, pending path, parent path, and an onSelect callback in a single pathPicker object. openPathDialog requests /api/path-browser with selection and optional path, renders entries, and calls showModal. Clicking a directory navigates into it. In directory mode Select applies current_path. In WAV mode selecting a WAV sets pendingPath and Select applies it. Cancel, close, and a failed browse request must not mutate the draft.

Wire the model picker callback to appState.draft.offlineModelDir. Wire each lane WAV picker callback to input.local_wav_path. Keep outputs in sync through renderLaneInputControls and loadDraftFromPayload. Existing input kind changes must hide both WAV output and picker unless kind is local_wav.

Extend updateControlAvailability to disable picker buttons and dialog actions outside unloaded state.

- [ ] **Step 5: Verify Web tests**

Run .venv313/bin/python -m pytest tests/test_web_server.py -q. Expected: all Web API and static page tests pass.

- [ ] **Step 6: Commit**

Run git add src/multi_stream_asr/web/static/index.html tests/test_web_server.py && git commit -m 'feat: add native path picker dialog'.

### Task 4: Document and Verify the Complete Flow

**Files:** Modify README.md. Test tests/test_web_server.py.

- [ ] **Step 1: Add ordering regression test**

Use a manager_factory that records configurations. Submit an invalid reconfigured model path. Assert HTTP 422 and that only the initial manager was constructed. This proves final validation runs before manager_factory and model process startup.

- [ ] **Step 2: Verify RED**

Run .venv313/bin/python -m pytest tests/test_web_server.py::test_invalid_path_load_does_not_construct_reconfigured_manager -q. Expected: fail until the validation is placed before the reconfiguration branch.

- [ ] **Step 3: Document operator behavior**

Add one README Web mode paragraph: model directories and local WAV files are selected from readable paths on the service host; the server validates directory access, required model files, and explicit WAV format again before loading.

- [ ] **Step 4: Run final verification**

Run .venv313/bin/python -m pytest -q, then .venv313/bin/python -m multi_stream_asr.main --headless --fake --lanes 4, then git diff --check. Expected: full suite passes, fake mode exits 0, and no whitespace errors.

- [ ] **Step 5: Commit**

Run git add README.md src/multi_stream_asr/web/server.py tests/test_web_server.py && git commit -m 'docs: describe web path selection validation'.

## Self-Review

- Spec coverage: Task 1 covers unrestricted readable browsing; Task 2 covers final pre-load validation; Task 3 covers native dialog behavior; Task 4 covers ordering, documentation, and full regression.
- Placeholder scan: no TBD, TODO, or deferred implementation language remains.
- Type consistency: Task 1 defines browse_path and permission helpers; Task 2 consumes them; Task 3 consumes only the documented API response.

## Execution Handoff

Plan complete and saved to docs/superpowers/plans/2026-07-12-web-path-dialog.md. Two execution options:

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task and review between tasks.

2. Inline Execution - execute tasks in this session using executing-plans with checkpoints.
