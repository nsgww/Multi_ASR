from __future__ import annotations

import asyncio

import numpy as np
import pytest

from multi_stream_asr.buffer.segment_cache import SegmentCache
from multi_stream_asr.models import (
    ASREvent,
    AudioFrame,
    EventStatus,
    SpeakerInfo,
)
from multi_stream_asr.pipeline.pipeline import LanePipeline
from multi_stream_asr.web import messages as msg
from multi_stream_asr.web.event_bus import FrontendEventBus

from conftest import (
    FakeAudioSource,
    FakeOfflineRecognizer,
    FakeOnlineRecognizer,
    make_frame,
    make_online,
)

INFO = SpeakerInfo(
    id="spk_1", label="S1", name="张三", confidence=0.83, is_provisional=False
)


class StubTracker:
    enabled = True

    def __init__(self, info=INFO, mapping=None) -> None:
        self._info = info
        self._mapping = mapping or {}
        self.assigned: list[np.ndarray] = []
        self.resegment_calls = 0

    def assign(self, pcm):  # noqa: ANN001
        self.assigned.append(np.asarray(pcm).copy())
        return self._info

    def resegment(self) -> dict[str, str]:
        self.resegment_calls += 1
        return dict(self._mapping)


class BrokenTracker:
    enabled = True

    def assign(self, pcm):  # noqa: ANN001
        raise RuntimeError("speaker backend exploded")

    def resegment(self) -> dict[str, str]:
        raise RuntimeError("resegment exploded")


def build_source() -> FakeAudioSource:
    return FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[1, 2, 3, 4], timestamp_ms=100),
            AudioFrame.finish(0, 1, 200),
        ]
    )


def build_pipeline(
    *,
    tracker=None,
    on_speaker=None,
    bus: FrontendEventBus | None = None,
    offline: FakeOfflineRecognizer | None = None,
) -> tuple[LanePipeline, FrontendEventBus, FakeOfflineRecognizer]:
    bus = bus or FrontendEventBus()
    offline = offline or FakeOfflineRecognizer(text="offline hello")
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [make_online(text="hello", timestamp_ms=100, is_endpoint=True)]
        ),
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
        speaker_tracker=tracker,
        on_speaker=on_speaker,
    )
    return pipeline, bus, offline


async def drain(bus: FrontendEventBus, limit: int = 20) -> list[ASREvent]:
    events: list[ASREvent] = []
    for _ in range(limit):
        try:
            events.append(await asyncio.wait_for(bus.next_event(), timeout=0.05))
        except asyncio.TimeoutError:
            break
    return events


@pytest.mark.asyncio
async def test_final_event_carries_speaker() -> None:
    tracker = StubTracker()
    pipeline, bus, offline = build_pipeline(tracker=tracker)

    await pipeline.run(build_source())
    events = await drain(bus)

    finals = [e for e in events if e.status is EventStatus.FINAL]
    assert len(finals) == 1
    assert finals[0].speaker == INFO
    # 声纹与离线 ASR 吃的是同一段 pcm
    assert len(tracker.assigned) == 1
    np.testing.assert_array_equal(tracker.assigned[0], offline.requests[0][0])


@pytest.mark.asyncio
async def test_pending_event_has_no_speaker_yet() -> None:
    tracker = StubTracker()
    pipeline, bus, _ = build_pipeline(tracker=tracker)

    await pipeline.run(build_source())
    events = await drain(bus)

    pending = [e for e in events if e.status is EventStatus.PENDING]
    assert len(pending) == 1
    assert pending[0].speaker is None


@pytest.mark.asyncio
async def test_without_tracker_final_has_no_speaker() -> None:
    pipeline, bus, _ = build_pipeline(tracker=None)

    await pipeline.run(build_source())
    events = await drain(bus)

    finals = [e for e in events if e.status is EventStatus.FINAL]
    assert len(finals) == 1
    assert finals[0].speaker is None


@pytest.mark.asyncio
async def test_speaker_remap_is_emitted_at_finish() -> None:
    collected: list[dict] = []
    tracker = StubTracker(mapping={"S3": "S1"})
    pipeline, bus, _ = build_pipeline(
        tracker=tracker, on_speaker=collected.append
    )

    await pipeline.run(build_source())
    await drain(bus)

    assert tracker.resegment_calls == 1
    assert collected == [
        {"type": "speaker_remap", "lane_id": 0, "mapping": {"S3": "S1"}}
    ]


@pytest.mark.asyncio
async def test_empty_remap_is_not_emitted() -> None:
    collected: list[dict] = []
    tracker = StubTracker(mapping={})
    pipeline, bus, _ = build_pipeline(
        tracker=tracker, on_speaker=collected.append
    )

    await pipeline.run(build_source())
    await drain(bus)

    assert collected == []


@pytest.mark.asyncio
async def test_broken_tracker_degrades_without_breaking_transcription() -> None:
    pipeline, bus, _ = build_pipeline(tracker=BrokenTracker())

    await pipeline.run(build_source())
    events = await drain(bus)

    finals = [e for e in events if e.status is EventStatus.FINAL]
    assert len(finals) == 1
    assert finals[0].text == "offline hello"
    assert finals[0].speaker is None
    # 收尾事件仍然正常发出
    assert events[-1].status is EventStatus.LANE_FINISHED


def test_transcript_message_includes_speaker_payload() -> None:
    event = ASREvent.asr_result(
        stream_id=0,
        sentence_id=1,
        revision=2,
        status=EventStatus.FINAL,
        text="你好",
        timestamp_ms=100,
        speaker=INFO,
    )
    payload = msg.transcript_message(event)

    assert payload["segment"]["text"] == "你好"
    assert payload["segment"]["speaker"] == {
        "id": "spk_1",
        "label": "S1",
        "name": "张三",
        "confidence": 0.83,
        "is_provisional": False,
    }


def test_transcript_message_omits_speaker_when_absent() -> None:
    event = ASREvent.asr_result(
        stream_id=0,
        sentence_id=1,
        revision=0,
        status=EventStatus.PARTIAL,
        text="hello",
        timestamp_ms=100,
    )
    payload = msg.transcript_message(event)

    assert "speaker" not in payload["segment"]


def test_speaker_remap_message_shape() -> None:
    payload = msg.speaker_remap_message({"S3": "S1"}, lane_id=2)

    assert payload == {
        "type": "speaker_remap",
        "lane_id": 2,
        "mapping": {"S3": "S1"},
    }
