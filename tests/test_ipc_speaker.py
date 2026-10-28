from __future__ import annotations

from multi_stream_asr.models import ASREvent, EventStatus, SpeakerInfo
from multi_stream_asr.pipeline.ipc import asr_event_from_payload, asr_event_to_payload


def test_asr_event_payload_round_trips_speaker() -> None:
    event = ASREvent.asr_result(
        stream_id=1,
        sentence_id=2,
        revision=0,
        status=EventStatus.FINAL,
        text="你好",
        timestamp_ms=100,
        speaker=SpeakerInfo(
            id="spk_1",
            label="S1",
            name="张三",
            confidence=0.87,
            is_provisional=False,
        ),
    )

    decoded = asr_event_from_payload(asr_event_to_payload(event))

    assert decoded.speaker == event.speaker


def test_asr_event_payload_round_trips_none_speaker() -> None:
    event = ASREvent.asr_result(
        stream_id=0,
        sentence_id=0,
        revision=0,
        status=EventStatus.PARTIAL,
        text="嗯",
        timestamp_ms=50,
    )

    decoded = asr_event_from_payload(asr_event_to_payload(event))

    assert decoded.speaker is None
    assert decoded == event


def test_speaker_info_payload_round_trip() -> None:
    info = SpeakerInfo(id="spk_2", label="S2", confidence=0.5, is_provisional=True)

    decoded = SpeakerInfo.from_payload(info.to_payload())

    assert decoded == info
