from __future__ import annotations

import numpy as np
import pytest

from multi_stream_asr.models import ASREvent, EventStatus, LaneFinalState, LaneMetrics
from multi_stream_asr.pipeline.ipc import (
    LaneIpcMessage,
    asr_event_from_payload,
    asr_event_to_payload,
    lane_metrics_from_payload,
    lane_metrics_to_payload,
    make_message,
)


def test_lane_ipc_message_round_trips_jsonable_payload():
    message = make_message(
        message_type="lifecycle",
        stream_id=2,
        seq=7,
        payload={"state": "lane_loaded", "message": ""},
        time_ms=123456,
    )

    encoded = message.to_dict()
    decoded = LaneIpcMessage.from_dict(encoded)

    assert encoded == {
        "type": "lifecycle",
        "stream_id": 2,
        "seq": 7,
        "time_ms": 123456,
        "payload": {"state": "lane_loaded", "message": ""},
    }
    assert decoded == message


def test_asr_event_payload_round_trips_enums_as_strings():
    event = ASREvent.asr_result(
        stream_id=1,
        sentence_id=3,
        revision=4,
        status=EventStatus.FINAL,
        text="final text",
        timestamp_ms=900,
    )

    payload = asr_event_to_payload(event)
    decoded = asr_event_from_payload(payload)

    assert payload["status"] == "final"
    assert payload["result_type"] == "offline"
    assert decoded == event


def test_lane_metrics_payload_round_trips_final_state():
    metrics = LaneMetrics(
        stream_id=3,
        processed_frames=11,
        dropped_frames=2,
        offline_requests=4,
        final_state=LaneFinalState.FINISHED,
    )

    payload = lane_metrics_to_payload(metrics)
    decoded = lane_metrics_from_payload(payload)

    assert payload["final_state"] == "finished"
    assert decoded == metrics


def test_lane_ipc_message_rejects_non_json_payload_at_every_constructor():
    payload = {"pcm": np.asarray([1.0], dtype=np.float32)}

    with pytest.raises(TypeError):
        make_message(message_type="audio", stream_id=1, seq=1, payload=payload)

    with pytest.raises(TypeError):
        LaneIpcMessage(type="audio", stream_id=1, seq=1, time_ms=1, payload=payload)

    with pytest.raises(TypeError):
        LaneIpcMessage.from_dict(
            {
                "type": "audio",
                "stream_id": 1,
                "seq": 1,
                "time_ms": 1,
                "payload": payload,
            }
        )


def test_lane_ipc_message_rejects_non_finite_float_payload():
    with pytest.raises(ValueError):
        make_message(
            message_type="metrics",
            stream_id=1,
            seq=1,
            payload={"latency": float("nan")},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", object()),
        ("stream_id", 1.5),
        ("stream_id", float("nan")),
        ("seq", True),
        ("time_ms", False),
    ],
)
def test_lane_ipc_message_rejects_invalid_envelope_fields(field, value):
    with pytest.raises(TypeError):
        LaneIpcMessage(
            type=value if field == "type" else "lifecycle",
            stream_id=value if field == "stream_id" else 1,
            seq=value if field == "seq" else 1,
            time_ms=value if field == "time_ms" else 1,
            payload={},
        )

    with pytest.raises(TypeError):
        make_message(
            message_type=value if field == "type" else "lifecycle",
            stream_id=value if field == "stream_id" else 1,
            seq=value if field == "seq" else 1,
            time_ms=value if field == "time_ms" else 1,
            payload={},
        )


def test_lane_ipc_message_from_dict_preserves_scalar_coercions():
    message = LaneIpcMessage.from_dict(
        {
            "type": 123,
            "stream_id": "2",
            "seq": "7",
            "time_ms": "123456",
            "payload": {},
        }
    )

    assert message == LaneIpcMessage(
        type="123",
        stream_id=2,
        seq=7,
        time_ms=123456,
        payload={},
    )
