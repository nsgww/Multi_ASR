# Web Path Dialog Design

## Goal

Replace the Web configuration page's free-form model-directory and local-WAV path
text fields with a native HTML `dialog` path picker. The picker selects existing
paths on the ASR service host without uploading or copying model files.

## Constraints

- Keep the current FastAPI plus vanilla HTML/CSS/JavaScript stack. Do not add a
  frontend framework, npm build step, or third-party widget.
- Use the browser's native `<dialog>` element for modal behavior and focus
  management; implement the directory list with existing vanilla JavaScript.
- The service and browser are normally run on the same development board. The
  chosen path is still a service-host path, not a browser-uploaded `File`.
- Browsing has no artificial root restriction. The initial directory is the
  current service user's home directory, and the user can navigate to any
  readable location, including `/`.
- The server must apply the current service user's permissions on every browse
  request and final load request. It must never rely on a client-side check.
- Model selection confirms directories only. Local audio selection confirms
  readable `.wav` files only. No upload endpoint is introduced.
- Existing configuration remains editable only while the service is unloaded.
- A failed validation must return HTTP 422, preserve the displayed draft, and
  prevent manager construction, Qwen startup, and lane startup.

## Server Interfaces

Add `GET /api/path-browser` with query parameters:

- `path`: optional service-host path. If omitted, use `Path.home()`.
- `selection`: `directory` or `wav`.

The response contains the resolved current path, its parent path when one
exists, and sorted readable entries. A directory entry is returned for
navigation when it is readable and traversable. In `wav` mode, readable files
whose suffix is `.wav` are also returned. Other files are not returned.

Before returning an entry or accepting it at load time, resolve symlinks with
`strict=True`, confirm the expected type, and check readability with `os.access`:
directories require `R_OK | X_OK`; files require `R_OK`. Invalid, missing,
unreadable, or wrong-type browse paths return HTTP 422. The endpoint does not
attempt to conceal readable paths, because unrestricted browsing is an explicit
requirement.

Extend load validation before `manager_factory(...)` or `load_models()`:

1. The selected offline model directory is a readable, traversable directory.
2. Every required Qwen file exists, is a regular readable file, and belongs to
   the selected model directory.
3. Every `local_wav` lane path is a readable regular file and passes existing
   16 kHz mono PCM WAV validation.

The 422 response groups errors by `model_directory`, `model_files`, and
`lane_inputs`, each containing the affected absolute paths and a concise reason.
`POST /api/models/validate` uses the same model-directory and required-file
validation so its result matches Load Models behavior.

## Frontend Interaction

Replace both editable path inputs with a read-only path display plus a compact
folder button. The model button opens the dialog in `directory` mode; each local
WAV lane button opens it in `wav` mode. The dialog shows the current path,
an Up icon button, a fixed-height list of directories and eligible WAV files,
and Cancel / Select actions.

Selecting a directory highlights it. Activating a directory navigates into it;
the current directory can be selected in model mode. Activating a WAV selects
and confirms it in WAV mode. The dialog keeps its own pending path and changes
the draft only after Select. Failed browse requests and failed Load Models
validation are shown through the existing configuration error area.

The dialog and all folder buttons use the existing unloaded/loading disable
rules. No per-lane path state is added beyond the current `draft.laneInputs`.

## Testing

Server tests cover default-home browsing, directory navigation, WAV filtering,
unreadable/wrong-type paths, symlink resolution, and final load rejection before
manager construction. Frontend static tests cover native dialog markup, model
and lane picker buttons, browser API calls, mode-specific confirmation, and
disabled controls. Existing missing-file validation tests remain and are updated
to assert the new structured error response.
