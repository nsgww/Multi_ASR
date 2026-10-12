import asyncio

import pytest

from multi_stream_asr.models import ASREvent, EventStatus, ResultType
from multi_stream_asr.web.event_bus import FrontendEventBus


def make_event(
    *,
    stream_id: int = 0,
    sentence_id: int = 0,
    revision: int = 0,
    status: EventStatus = EventStatus.PARTIAL,
    text: str = "",
    timestamp_ms: int | None = None,
) -> ASREvent:
    result_type = (
        ResultType.OFFLINE
        if status in {EventStatus.FINAL, EventStatus.QWEN_ERROR}
        else ResultType.ONLINE
    )
    return ASREvent(
        event_type="asr_result",
        stream_id=stream_id,
        sentence_id=sentence_id,
        revision=revision,
        status=status,
        result_type=result_type,
        is_final=status is EventStatus.FINAL,
        text=text,
        timestamp_ms=revision if timestamp_ms is None else timestamp_ms,
    )


class RecordingWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.messages.append(payload)


class BrokenWebSocket:
    async def send_json(self, payload: dict) -> None:
        raise RuntimeError("socket send failed")


def test_loop_retention_keeps_only_recent_results_per_lane():
    bus = FrontendEventBus()
    bus.set_result_retention_limit(100)

    for sentence_id in range(105):
        bus.publish(make_event(sentence_id=sentence_id, status=EventStatus.FINAL))

    retained = bus.snapshot_results()

    assert len(retained) == 100
    assert {sentence_id for _, sentence_id in retained} == set(range(5, 105))


def test_loop_retention_bounds_queued_events_when_no_websocket_consumer():
    bus = FrontendEventBus()
    bus.set_result_retention_limit(100)

    for sentence_id in range(1000):
        bus.publish(make_event(sentence_id=sentence_id, status=EventStatus.FINAL))

    assert bus._queue.qsize() <= 400


@pytest.mark.asyncio
async def test_newer_revisions_replace_snapshot_and_older_non_partial_events_are_still_delivered():
    bus = FrontendEventBus()

    bus.publish(make_event(status=EventStatus.PENDING, revision=1, text="hello"))
    bus.publish(make_event(status=EventStatus.PENDING, revision=3, text="hello world"))
    bus.publish(make_event(status=EventStatus.PENDING, revision=2, text="stale update"))

    first = await bus.next_event()
    second = await bus.next_event()
    third = await bus.next_event()

    assert first.revision == 1
    assert second.revision == 3
    assert third.revision == 2
    assert third.status is EventStatus.PENDING

    assert bus.snapshot_results() == {
        (0, 0): {
            "revision": 3,
            "text": "hello world",
            "status": EventStatus.PENDING,
            "result_type": ResultType.ONLINE,
            "is_final": False,
        }
    }


@pytest.mark.asyncio
async def test_queued_partial_events_may_merge_to_latest_revision():
    bus = FrontendEventBus()

    bus.publish(make_event(revision=1, text="hel"))
    bus.publish(make_event(revision=2, text="hello"))

    merged = await bus.next_event()

    assert merged.revision == 2
    assert merged.text == "hello"
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(bus.next_event(), timeout=0.05)


@pytest.mark.asyncio
async def test_broadcast_boundary_identifies_stale_partials_but_not_final_events():
    bus = FrontendEventBus()

    bus.publish(make_event(revision=1, text="hel"))
    candidate = await bus.next_event()
    bus.publish(make_event(revision=2, text="hello"))

    assert bus._is_stale_partial(candidate)
    assert not bus._is_stale_partial(make_event(revision=2, text="hello"))
    assert not bus._is_stale_partial(
        make_event(status=EventStatus.FINAL, revision=1, text="final")
    )


@pytest.mark.asyncio
async def test_stale_non_partial_events_are_still_delivered_while_stale_partials_are_ignored():
    bus = FrontendEventBus()

    bus.publish(make_event(revision=1, text="hel"))
    bus.publish(make_event(revision=3, text="hello"))
    bus.publish(make_event(revision=2, text="ignored stale partial"))

    bus.publish(make_event(status=EventStatus.FINAL, revision=2, text="old final"))
    bus.publish(make_event(status=EventStatus.FATAL_ERROR, revision=3, text="fatal"))
    bus.publish(make_event(status=EventStatus.LANE_FINISHED, revision=1, text=""))
    bus.publish(make_event(status=EventStatus.STATS_UPDATED, revision=0, text="stats"))

    delivered = [await bus.next_event() for _ in range(5)]

    assert [(event.status, event.revision, event.text) for event in delivered] == [
        (EventStatus.PARTIAL, 3, "hello"),
        (EventStatus.FINAL, 2, "old final"),
        (EventStatus.FATAL_ERROR, 3, "fatal"),
        (EventStatus.LANE_FINISHED, 1, ""),
        (EventStatus.STATS_UPDATED, 0, "stats"),
    ]
    snapshot = bus.snapshot_results()
    assert set(snapshot) == {
        (0, 0),
        (0, 0, EventStatus.FATAL_ERROR),
        (0, 0, EventStatus.LANE_FINISHED),
        (0, 0, EventStatus.STATS_UPDATED),
    }
    assert snapshot[(0, 0)] == {
        "revision": 3,
        "text": "hello",
        "status": EventStatus.PARTIAL,
        "result_type": ResultType.ONLINE,
        "is_final": False,
    }
    assert snapshot[(0, 0, EventStatus.FATAL_ERROR)]["text"] == "fatal"
    assert snapshot[(0, 0, EventStatus.LANE_FINISHED)]["text"] == ""
    assert snapshot[(0, 0, EventStatus.STATS_UPDATED)]["text"] == "stats"


@pytest.mark.asyncio
async def test_pending_final_fatal_lane_finished_and_stats_updated_are_not_dropped():
    bus = FrontendEventBus()
    events = [
        make_event(status=EventStatus.PENDING, revision=1, text="waiting"),
        make_event(status=EventStatus.FINAL, revision=2, text="done"),
        make_event(status=EventStatus.FATAL_ERROR, revision=3, text="boom"),
        make_event(status=EventStatus.LANE_FINISHED, revision=4, text=""),
        make_event(status=EventStatus.STATS_UPDATED, revision=5, text="stats"),
    ]

    for event in events:
        bus.publish(event)

    delivered = [await bus.next_event() for _ in events]

    assert [(event.status, event.revision) for event in delivered] == [
        (EventStatus.PENDING, 1),
        (EventStatus.FINAL, 2),
        (EventStatus.FATAL_ERROR, 3),
        (EventStatus.LANE_FINISHED, 4),
        (EventStatus.STATS_UPDATED, 5),
    ]


@pytest.mark.asyncio
async def test_pending_updates_internal_state_and_empty_final_replaces_pending_text():
    bus = FrontendEventBus()

    bus.publish(make_event(status=EventStatus.PENDING, revision=1, text="draft"))
    await bus.next_event()

    assert bus.snapshot_results()[(0, 0)]["text"] == "draft"
    assert bus.snapshot_results()[(0, 0)]["status"] is EventStatus.PENDING

    bus.publish(make_event(status=EventStatus.FINAL, revision=2, text=""))
    await bus.next_event()

    assert bus.snapshot_results() == {
        (0, 0): {
            "revision": 2,
            "text": "",
            "status": EventStatus.FINAL,
            "result_type": ResultType.OFFLINE,
            "is_final": True,
        }
    }


@pytest.mark.asyncio
async def test_qwen_error_has_distinct_retained_key_without_replacing_transcript():
    bus = FrontendEventBus()
    bus.set_result_retention_limit(1)

    bus.publish(make_event(status=EventStatus.PENDING, revision=1, text="draft"))
    bus.publish(
        make_event(
            status=EventStatus.QWEN_ERROR,
            revision=2,
            text="Qwen request failed",
        )
    )

    snapshot = bus.snapshot_results()
    assert snapshot[(0, 0)] == {
        "revision": 1,
        "text": "draft",
        "status": EventStatus.PENDING,
        "result_type": ResultType.ONLINE,
        "is_final": False,
    }
    assert snapshot[(0, 0, EventStatus.QWEN_ERROR)] == {
        "revision": 2,
        "text": "Qwen request failed",
        "status": EventStatus.QWEN_ERROR,
        "result_type": ResultType.OFFLINE,
        "is_final": False,
    }

    bus.publish(make_event(status=EventStatus.FINAL, revision=3, text="final"))

    snapshot = bus.snapshot_results()
    assert snapshot[(0, 0)]["status"] is EventStatus.FINAL
    assert snapshot[(0, 0)]["text"] == "final"
    assert snapshot[(0, 0, EventStatus.QWEN_ERROR)]["status"] is EventStatus.QWEN_ERROR


def test_non_transcript_statuses_have_distinct_keys_and_preserve_sentence_zero_pending():
    bus = FrontendEventBus()
    bus.publish(make_event(status=EventStatus.PENDING, revision=1, text="draft"))
    statuses = (
        EventStatus.QWEN_ERROR,
        EventStatus.FATAL_ERROR,
        EventStatus.LANE_FINISHED,
        EventStatus.STATS_UPDATED,
    )

    for revision, status in enumerate(statuses, start=2):
        bus.publish(make_event(status=status, revision=revision, text=status.value))

    snapshot = bus.snapshot_results()
    assert snapshot[(0, 0)]["status"] is EventStatus.PENDING
    assert snapshot[(0, 0)]["text"] == "draft"
    assert {
        key
        for key in snapshot
        if len(key) == 3
    } == {(0, 0, status) for status in statuses}


def test_retention_at_exact_capacity_keeps_qwen_error_with_pending_sentence_group():
    bus = FrontendEventBus()
    bus.set_result_retention_limit(2)
    bus.publish(make_event(sentence_id=0, status=EventStatus.PENDING, revision=1))
    bus.publish(make_event(sentence_id=1, status=EventStatus.PENDING, revision=1))

    bus.publish(make_event(sentence_id=0, status=EventStatus.QWEN_ERROR, revision=2))

    snapshot = bus.snapshot_results()
    assert set(snapshot) == {
        (0, 0),
        (0, 0, EventStatus.QWEN_ERROR),
        (0, 1),
    }


def test_delayed_qwen_error_for_pruned_sentence_is_ignored_without_evicting_newer_group():
    bus = FrontendEventBus()
    bus.set_result_retention_limit(2)
    for sentence_id in range(3):
        bus.publish(
            make_event(
                sentence_id=sentence_id,
                status=EventStatus.FINAL,
                revision=1,
                text=f"final {sentence_id}",
            )
        )

    before = bus.snapshot_results()
    queued_before = bus._queue.qsize()
    assert set(before) == {(0, 1), (0, 2)}

    bus.publish(
        make_event(
            sentence_id=0,
            status=EventStatus.QWEN_ERROR,
            revision=2,
            text="delayed failure",
        )
    )

    assert bus.snapshot_results() == before
    assert bus._queue.qsize() == queued_before


@pytest.mark.asyncio
async def test_broadcast_loop_sends_events_and_removes_only_broken_websockets():
    bus = FrontendEventBus()
    good = RecordingWebSocket()
    bad = BrokenWebSocket()
    bus.register(good)
    bus.register(bad)

    task = asyncio.create_task(bus.broadcast_loop())
    bus.publish(make_event(revision=7, text="broadcast"))

    for _ in range(20):
        if good.messages:
            break
        await asyncio.sleep(0.01)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert good.messages == [
        {
            "event_type": "asr_result",
            "stream_id": 0,
            "sentence_id": 0,
            "revision": 7,
            "status": EventStatus.PARTIAL,
            "result_type": ResultType.ONLINE,
            "is_final": False,
            "text": "broadcast",
            "timestamp_ms": 7,
        }
    ]
    assert bad not in bus._websockets
    assert good in bus._websockets
