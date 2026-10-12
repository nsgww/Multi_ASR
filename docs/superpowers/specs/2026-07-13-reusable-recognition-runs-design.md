# Reusable Recognition Runs Design

## Goal

After a recognition run finishes naturally or is stopped by the user, return the
service to the loaded state without reloading models. The next run can use new
lane audio inputs. Model configuration and lane count remain locked until the
models are unloaded.

## State Model

`loaded` means all lane model processes and the shared Qwen service are alive,
Zipformer has completed warmup, and no recognition run is active.

The normal run lifecycle is:

`loaded -> starting -> running -> stopping -> loaded`.

Both natural completion and a user stop use the same terminal sequence. A fatal
error remains fatal and requires unloading before a new model load. `stopped`
is no longer used as the normal idle state.

## Lane Process Lifecycle

Each lane process loads Zipformer and creates its Qwen client once. It then
waits for commands:

1. On `start`, it creates only run-scoped resources: audio source, pipeline,
   segment cache, result writer, and model-output recorder.
2. When the source completes or a `stop` command is received, it drains the
   offline tail, writes final metrics and artifact paths, closes the
   run-scoped resources, and emits a per-run terminal lifecycle event.
3. It keeps model-scoped resources alive and waits for the next `start`.
4. On `shutdown`, it closes recognizers and worker threads, emits
   `lane_stopped`, and exits.

Run-scoped state is reset before every new `start`, including stop flags,
source, pipeline, metrics trackers, and artifact bookkeeping. This prevents
events or metrics from one run being attributed to the next run.

## Manager Completion and Recording

The process manager tracks terminal status per lane for the active run rather
than waiting for child processes to exit. Once all configured lanes have
reported their terminal run lifecycle event, it performs run-finalization:

1. Wait for final metrics and artifact paths from every lane.
2. Make the completed run available to the web recorder.
3. Transition from `stopping` to `loaded`.

The web layer records the completed run exactly once after either a stop
request or natural completion. The recognition result view is not cleared
while finalization runs and remains visible in `loaded`.

## Independent Lane Input Configuration

Lane inputs are separate from model configuration. A dedicated request updates
the complete lane-input list only while the manager is `loaded`.

The server validates every supplied local WAV path for readability and required
16 kHz mono PCM format before accepting it. It updates the manager's pending
inputs and the web application's input configuration atomically. The update
does not recreate processes, reload models, or alter the current run artifacts.

The next `start` resolves audio commands from the newly accepted lane inputs.
Input updates are rejected during loading, start-up, running, stopping,
unloading, fatal, and unloaded states.

## Frontend Behavior

The UI separates model controls from lane input controls:

- Lane count, offline model selection, and model directory are editable only
  when `unloaded`.
- Lane input selectors are editable when `unloaded` or `loaded`.
- All controls that affect a current run are disabled while loading, starting,
  running, stopping, or unloading.
- When the manager returns to `loaded` after four lanes finish, Start is
  enabled and Unload remains available.
- Stop is enabled only during start-up and running.
- Existing transcript results remain displayed after completion or stop. They
  are cleared only after the next start request succeeds.

The existing path dialog follows the lane-input availability rule, so WAV
selection remains usable in `loaded` while model-directory selection does not.

## Errors and Safety

Invalid input updates return a validation error and leave the accepted inputs
unchanged. Requests made in an invalid state return a conflict error. A lane
fatal error cancels sibling work, moves the manager to `fatal`, and does not
claim that the models remain reusable.

## Verification

Automated tests cover:

- two sequential runs without model unload or reload;
- natural completion and user stop returning to `loaded` only after every lane
  finishes its run-finalization protocol;
- preservation of run results until the subsequent successful start;
- lane input update acceptance in `loaded`, use by the next run, and rejection
  during an active run;
- model controls remaining locked while loaded and lane input controls becoming
  editable after load and after run completion.
