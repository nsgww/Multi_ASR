# Web UI Tutorial Synchronization Design

## Goal

Update `docs/web-ui-tutorial.html` so the standalone, offline tutorial matches
the current Web UI's controls, layout, and key recognition behavior. A user can
open the file locally in Chromium, practise the workflow without a server, and
learn when to use the new loop-audio control.

## Scope

The tutorial remains a browser-only simulation. It does not call `/api/*`,
connect a WebSocket, load models, or capture microphone audio. Its labels,
visible controls, state transitions, lane result presentation, and tutorial
steps follow `src/multi_stream_asr/web/static/index.html`.

## Interface and Interaction

The simulated configuration panel uses the current lane-input layout:

- Every lane exposes input type, local-WAV picker, microphone selector, and a
  `Loop audio` checkbox.
- `Loop audio` is visible for **Default files** and **Local WAV**. It is hidden
  and reset to off for **Microphone**.
- The tutorial explains that loop audio repeats a selected or default file for
  continuous observation or stress testing; it does not apply to live
  microphone input.
- While recognition is active, lane input controls are read-only, matching the
  current UI. The existing model load, unload, start, stop, path-picker, and
  microphone-selection practice flows remain available.

## Simulation Behavior

The tutorial uses the current UI's lane/result model in simplified local form:

- Per-lane results update individual rows instead of clearing and rebuilding
  the complete result list.
- Transcript revisions replace the row for the same sentence; non-transcript
  statuses retain separate rows where the current UI does.
- The display keeps at most 100 sentence groups per lane. Removing the oldest
  sentence removes its related result rows; delayed Qwen errors for an already
  removed sentence are not shown again.
- For loop-enabled file lanes, the sample sequence repeats until Stop is
  selected. Each repeat starts new sentence identifiers so the display and
  event counter continue to behave like a long-running session.
- Stop cancels scheduled simulated events and returns the tutorial to the
  loaded state. Completion returns to loaded only after every lane has
  finished.

## Tutorial Content

The guided steps are refreshed to describe current states and add an explicit
file-loop exercise: choose Default files or Local WAV, enable `Loop audio`,
start recognition, observe repeated results, and choose Stop. The microphone
step explicitly states that loop audio is unavailable for live input.

## Verification

- Open the standalone file in Chromium and confirm the layout, path picker,
  lane controls, model/load transitions, and tutorial navigation work without
  a backend.
- Confirm the loop checkbox visibility rule for all three input types and that
  a loop-enabled file lane repeats until Stop.
- Confirm a microphone lane hides and clears its loop checkbox.
- Run a lightweight HTML/JavaScript syntax check appropriate to the existing
  repository tooling, without introducing a new test suite.
