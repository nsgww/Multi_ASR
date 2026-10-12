from __future__ import annotations

from multi_stream_asr.models import ASREvent, EventStatus, ResultType
from multi_stream_asr.web import messages as msg


def test_transcript_message_for_partial() -> None:
    event = ASREvent.asr_result(
        stream_id=0,
        sentence_id=1,
        revision=2,
        status=EventStatus.PARTIAL,
        text="hello",
        timestamp_ms=100,
    )
    payload = msg.transcript_message(event)
    assert payload["type"] == "transcript"
    assert payload["is_partial"] is True
    assert payload["is_final"] is False
    assert payload["segment"]["text"] == "hello"
    assert payload["lane_id"] == 0


def test_transcript_message_for_final() -> None:
    event = ASREvent.asr_result(
        stream_id=0,
        sentence_id=1,
        revision=2,
        status=EventStatus.FINAL,
        text="hello world",
        timestamp_ms=200,
    )
    payload = msg.transcript_message(event)
    assert payload["is_final"] is True
    assert payload["segment"]["text"] == "hello world"


def test_summary_message_includes_lane_id() -> None:
    payload = msg.summary_message(
        {"summary": "要点", "provider": "llama"},
        lane_id=3,
    )
    assert payload["type"] == "summary"
    assert payload["lane_id"] == 3
    assert payload["provider"] == "llama"


def test_error_message() -> None:
    payload = msg.error_message("something failed", lane_id=2)
    assert payload["type"] == "error"
    assert payload["message"] == "something failed"
    assert payload["lane_id"] == 2
