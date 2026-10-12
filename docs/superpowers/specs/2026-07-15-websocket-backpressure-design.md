# WebSocket Backpressure Design

## Goal

Keep microphone recognition events lossless for `pending`, `final`, `fatal`, and
`lane_finished` while preventing a long-running four-lane microphone session
from accumulating visible Web UI delay. The microphone event queue remains
unbounded; this change does not alter Qwen, Zipformer, audio input, or stop
semantics.

## Problem

`FrontendEventBus` already coalesces queued partial results, but the browser
processes every WebSocket message immediately. For each message it clears and
rebuilds every displayed row in the affected lane. On a repeated microphone
recording, the number of retained rows grows from roughly one playback's worth
of sentences to the 100-row display limit. Four lanes emit updates together,
so browser rendering falls behind even when backend recognition timings stay
stable.

## Chosen Design

### Event semantics

The backend keeps the existing unbounded microphone queue and its reliable
delivery semantics for non-partial events. A newer partial result replaces an
older queued partial for the same `(stream_id, sentence_id)` key. At the
broadcast boundary, a stale partial is skipped when a newer partial for that
key is pending, so the browser receives only the latest useful partial update.

`pending`, `final`, `fatal_error`, `qwen_error`, `lane_finished`, and other
non-partial lifecycle events are never coalesced or dropped. Their order in
the shared event queue remains unchanged.

### Browser batching

The WebSocket message handler parses messages immediately, then puts them in a
pending render batch. One `requestAnimationFrame` callback consumes the batch:

- Partial events are keyed by `(stream_id, sentence_id)` and only the newest
  revision from that frame is rendered.
- Non-partial events retain arrival order and are rendered individually.
- Revision filtering continues to reject obsolete transcript revisions.
- Lane-finished handling runs after the corresponding event is rendered.

This preserves the displayed final result and lifecycle behavior while bounding
partial DOM work to one update per sentence per browser frame.

### Incremental lane rendering

Each lane owns a mapping from its event-storage key to a result-row DOM node.
Rendering an event updates only that row's text and CSS class, or appends one
new row for a new sentence. The lane result container is not cleared during a
normal recognition update.

The existing 100-sentence display limit remains. When a new event exceeds the
limit, the oldest event and its associated DOM row are removed, along with its
revision tracking entry. Scroll-to-bottom behavior remains unchanged for a
user already pinned to the bottom.

## Error Handling and Stop Behavior

This change intentionally does not cancel or reorder Qwen work on Stop. Events
produced while the backend drains existing offline tasks are rendered through
the same batch. If the browser is slow, partial updates are compressed; final
and lifecycle events remain visible.

An indefinitely disconnected browser is removed by the existing WebSocket send
failure path. Because the microphone event stream has no configured queue cap,
this design does not introduce a new drop policy for non-partial events.

## Verification

- Event-bus tests prove queued and broadcast-boundary partials collapse to the
  newest revision, while non-partial events are all delivered in order.
- Web UI tests assert the message handler batches rendering with
  `requestAnimationFrame` and that lane updates do not clear the result
  container.
- Web UI tests cover in-place updates, append behavior, eviction at 100 rows,
  revision handling, and lane-finished completion handling.
- Run the focused event-bus and Web UI tests using the worktree source with the
  existing project Python 3.13 virtual environment.
