# WebSocket Backpressure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent browser-side WebSocket/render backlog during long microphone recognition while preserving lossless lifecycle and final-result delivery.

**Architecture:** Keep the microphone queue unbounded. The backend only suppresses stale `partial` events at its broadcast boundary; every non-partial event keeps its existing queue order and delivery behavior. In the browser, receive messages immediately but render once per animation frame: coalesce partials by sentence, retain non-partial order, and update existing result-row DOM nodes in place.

**Tech Stack:** Python 3.13, `asyncio`, FastAPI WebSocket event bus, vanilla browser JavaScript, pytest.

## Global Constraints

- Do not add a size limit or discard policy to microphone event queues.
- Do not coalesce, drop, or reorder `pending`, `final`, `fatal_error`, `qwen_error`, `lane_finished`, or any other non-`partial` event.
- Do not change Qwen scheduling, Zipformer/audio behavior, or stop/drain semantics.
- Use the existing interpreter at `/home/liangff/Workspace/projects/multi-stream-asr/.venv313/bin/python` with `PYTHONPATH=src`; do not create a second environment in the worktree.

---

## Task 1: Suppress stale partials at the event-bus broadcast boundary

**Files:**

- Modify: `src/multi_stream_asr/web/event_bus.py`
- Modify: `tests/test_event_bus.py`

- [ ] **Step 1: Add a focused failing event-bus test.**

  Add a test near the existing partial-coalescing tests that publishes partial revision 1, dequeues it as the candidate for broadcast, then publishes revision 2 for the same `(stream_id, sentence_id)`. Assert that the event bus identifies revision 1 as stale at broadcast time and revision 2 as current. In the same test, publish a `final` event and assert it is not classified as suppressible. This keeps the test deterministic without relying on `set` iteration order of registered WebSockets.

  Run:

  ```bash
  PYTHONPATH=src /home/liangff/Workspace/projects/multi-stream-asr/.venv313/bin/python -m pytest tests/test_event_bus.py -q
  ```

  Expected: the new test fails because the event-bus has no broadcast-boundary stale-partial predicate yet.

- [ ] **Step 2: Implement the smallest event-bus predicate and boundary check.**

  In `FrontendEventBus`, add a private helper that returns true only when an event is `EventStatus.PARTIAL` and its revision is older than the latest stored result for the identical stream/sentence key. Do not use this helper in `_enqueue`, `publish`, or queue retention; existing queue coalescing remains responsible for queued partial replacement.

  In `broadcast_loop`, check this helper immediately before serializing/sending the event and again before each client send if a previous awaited send may have allowed a newer partial to arrive. If the partial has become stale, stop sending that event to remaining clients and continue the loop. Non-partial events must always pass through unchanged.

- [ ] **Step 3: Run focused backend verification.**

  Run:

  ```bash
  PYTHONPATH=src /home/liangff/Workspace/projects/multi-stream-asr/.venv313/bin/python -m pytest tests/test_event_bus.py -q
  ```

  Expected: all event-bus tests pass, including loop-audio retention coverage and the new stale-partial case.

- [ ] **Step 4: Commit the backend change.**

  ```bash
  git add src/multi_stream_asr/web/event_bus.py tests/test_event_bus.py
  git commit -m "fix: coalesce stale websocket partials"
  ```

## Task 2: Batch browser events and render lane rows incrementally

**Files:**

- Modify: `src/multi_stream_asr/web/static/index.html`
- Modify: `tests/test_index_html.py`
- Modify only if an existing static UI assertion needs adjustment: `tests/test_web_server.py`

- [ ] **Step 1: Add static UI regression assertions before changing the page.**

  Extend `tests/test_index_html.py` to assert that the page:

  - uses `requestAnimationFrame` to schedule an event-batch flush;
  - keeps pending partials in a key-addressable collection and keeps non-partials in an arrival-ordered collection;
  - has per-lane row tracking for incremental updates; and
  - does not clear the lane result container in the normal event-rendering path.

  Retain the existing `qwen_error` storage-key assertion, because Qwen errors must remain separate from transcript revisions.

  Run:

  ```bash
  PYTHONPATH=src /home/liangff/Workspace/projects/multi-stream-asr/.venv313/bin/python -m pytest tests/test_index_html.py -q
  ```

  Expected: the new assertions fail against the full-container `innerHTML = ""` implementation.

- [ ] **Step 2: Introduce batch state and a frame scheduler.**

  In the `appState` object, add state for the latest pending partial per lane/sentence key, the ordered pending non-partial list, and whether a frame flush is already scheduled. Add a queueing function used by the WebSocket `message` handler:

  - parse and revision-filter immediately, preserving the current revision semantics;
  - increment the received event counter for accepted messages;
  - replace only a same-key pending `partial` event;
  - append all non-partial events without coalescing; and
  - schedule exactly one `requestAnimationFrame` flush.

  Clear the batch collections during recognition-view reset so a previous session cannot paint into a new session.

- [ ] **Step 3: Make `renderLaneEvent` update one DOM row.**

  Extend each lane record from `ensureLane` with a map of storage key to DOM row. In `renderLaneEvent`, update `lane.events` as today but:

  - reuse and update an existing row for the storage key, or append a new row when absent;
  - update the row class and text using the existing transcript/error formatting rules;
  - retain metrics and pinned-scroll behavior; and
  - on the 100-event display limit, remove the oldest event, its DOM row, and the matching revision entry.

  Do not use `lane.results.innerHTML = ""` in the ordinary event rendering path. Snapshot/restore may continue to call `renderLaneEvent`; a newly created lane begins with an empty row map.

- [ ] **Step 4: Flush safely and preserve completion behavior.**

  Implement the animation-frame flush to copy and clear pending state, discard a queued partial if a newer transcript revision has already been observed, render surviving partials, then render non-partials in original arrival order. Call `renderStatus()` once after a nonempty batch, instead of once per WebSocket message.

  Extract or retain the current `lane_finished` completion logic so it executes after that event has been rendered. Preserve the existing single scheduling of `waitForRecognitionCompletion` and the current timeout behavior.

- [ ] **Step 5: Run focused UI verification.**

  Run:

  ```bash
  PYTHONPATH=src /home/liangff/Workspace/projects/multi-stream-asr/.venv313/bin/python -m pytest tests/test_index_html.py tests/test_web_server.py -q
  ```

  Expected: static page assertions and web-server UI-serving tests pass.

- [ ] **Step 6: Commit the UI change.**

  ```bash
  git add src/multi_stream_asr/web/static/index.html tests/test_index_html.py tests/test_web_server.py
  git commit -m "fix: batch websocket UI rendering"
  ```

## Task 3: Verify the combined behavior and preserve worktree hygiene

**Files:**

- Verify: `src/multi_stream_asr/web/event_bus.py`
- Verify: `src/multi_stream_asr/web/static/index.html`
- Verify: `tests/test_event_bus.py`
- Verify: `tests/test_index_html.py`
- Verify: `tests/test_web_server.py`

- [ ] **Step 1: Run the combined regression suite.**

  ```bash
  PYTHONPATH=src /home/liangff/Workspace/projects/multi-stream-asr/.venv313/bin/python -m pytest tests/test_event_bus.py tests/test_index_html.py tests/test_web_server.py -q
  ```

  Expected: all selected tests pass using the shared project Python 3.13 environment.

- [ ] **Step 2: Check the exact diff and whitespace.**

  Run:

  ```bash
  git diff HEAD~2..HEAD --check
  git status --short
  ```

  Expected: no whitespace errors; only the two intentional implementation commits and the pre-existing committed specification are present on this worktree branch.

- [ ] **Step 3: Review behavior against the approved design.**

  Confirm from the final diff that microphone queues remain uncapped; no non-partial status is coalesced; the backend skip applies exclusively to stale partials; the browser renders in animation-frame batches; normal lane updates avoid full result-container rebuilds; the 100-row display limit still evicts oldest rows; and lane-finished completion handling remains after rendering.
