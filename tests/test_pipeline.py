from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import AsyncIterator

import numpy as np
import pytest

from multi_stream_asr.asr.offline.base import OfflineRecognitionResult
from multi_stream_asr.asr.offline.qwen3asr import QwenRequestError, QwenRequestFailure
from multi_stream_asr.buffer.audio_queue import OnlineFrameQueue
from multi_stream_asr.buffer.segment_cache import SegmentCache
from multi_stream_asr.config import EndpointConfig
from multi_stream_asr.log.metrics import (
    SHARED_TIMING_FIELDS,
    LatencyTracker,
    RtfTracker,
    SharedTimingsTracker,
)
from multi_stream_asr.log.model_output import ModelOutputRecorder
from multi_stream_asr.models import ASREvent, AudioFrame, EventStatus, LaneFinalState, ResultType
from multi_stream_asr.pipeline.pipeline import LanePipeline
from multi_stream_asr.web.event_bus import FrontendEventBus

from conftest import (
    FakeAudioSource,
    FakeOfflineRecognizer,
    FakeOnlineRecognizer,
    make_frame,
    make_online,
)


class BlockingOnlineRecognizer:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.accepted_frames = []
        self.reset_count = 0

    async def accept(self, frame):
        self.accepted_frames.append(frame)
        if len(self.accepted_frames) == 1:
            self.started.set()
            self.release.wait()
        return None

    async def reset(self):
        self.reset_count += 1


class SynchronousSlowOnlineRecognizer:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.accepted_frames = []
        self.reset_count = 0

    async def accept(self, frame):
        self.accepted_frames.append(frame)
        if len(self.accepted_frames) == 1:
            self.started.set()
            self.release.wait()
        return None

    async def reset(self):
        self.reset_count += 1


class GatedBurstAudioSource:
    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = list(frames)
        self.release_burst = threading.Event()
        self.finish_yield_started = threading.Event()
        self.prepare_calls = 0
        self.stop_calls = 0

    async def prepare(self) -> None:
        self.prepare_calls += 1

    async def frames(self) -> AsyncIterator[AudioFrame]:
        yield self._frames[0]
        self.release_burst.wait()
        for frame in self._frames[1:]:
            if frame.is_finish:
                self.finish_yield_started.set()
            yield frame

    async def stop(self) -> None:
        self.stop_calls += 1


class AcknowledgedAudioSource:
    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = list(frames)
        self.frame_accepted = threading.Event()
        self.prepare_calls = 0
        self.stop_calls = 0

    async def prepare(self) -> None:
        self.prepare_calls += 1

    async def frames(self) -> AsyncIterator[AudioFrame]:
        for frame in self._frames:
            self.frame_accepted.clear()
            yield frame
            if not frame.is_finish:
                self.frame_accepted.wait()

    async def stop(self) -> None:
        self.stop_calls += 1
        self.frame_accepted.set()


class AcknowledgingOnlineRecognizer(FakeOnlineRecognizer):
    def __init__(self, outputs, source: AcknowledgedAudioSource) -> None:
        super().__init__(outputs)
        self._source = source

    async def accept(self, frame):
        try:
            return await super().accept(frame)
        finally:
            self._source.frame_accepted.set()


class RawResponseOfflineRecognizer(FakeOfflineRecognizer):
    async def recognize(self, pcm, prompt):
        self.requests.append((pcm.copy(), prompt))
        return OfflineRecognitionResult(
            text="offline final",
            elapsed_ms=400.0,
            audio_seconds=1.0,
            rtf=0.4,
            raw_response={"text": "<asr_text>offline final</asr_text>"},
            timings={
                "queue_ms": 10.0,
                "encode_ms": 100.0,
                "prefill_ms": 20.0,
                "decode_ms": 200.0,
                "total_ms": 330.0,
            },
        )


class CancellationAwareOfflineRecognizer(FakeOfflineRecognizer):
    def __init__(self) -> None:
        super().__init__(release=asyncio.Event())
        self.cancelled = False

    async def recognize(self, pcm, prompt):
        try:
            return await super().recognize(pcm, prompt)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class FailOnceOfflineRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    async def recognize(self, pcm, prompt):
        self.calls += 1
        if self.calls == 1:
            raise QwenRequestError(
                QwenRequestFailure(
                    category="http_status",
                    error_type="HTTPStatusError",
                    message="HTTP 500",
                    method="POST",
                    url="http://qwen/v1/chat/completions",
                    timeout_ms=120000,
                    audio_seconds=1.0,
                    sample_count=16000,
                    status_code=500,
                    response_body="decoder failed",
                )
            )
        return OfflineRecognitionResult(
            text="second final",
            elapsed_ms=100.0,
            audio_seconds=1.0,
            rtf=0.1,
        )


class FailingWhileConsumerBlockedSource:
    def __init__(self, consumer_started: threading.Event) -> None:
        self.consumer_started = consumer_started
        self.failed = threading.Event()
        self.stop_calls = 0
        self.stop_returned = asyncio.Event()

    async def prepare(self) -> None:
        return None

    async def frames(self) -> AsyncIterator[AudioFrame]:
        yield make_frame(frame_index=0, samples=[0.1], timestamp_ms=100)
        self.consumer_started.wait()
        self.failed.set()
        raise RuntimeError("producer failed while consumer blocked")

    async def stop(self) -> None:
        self.stop_calls += 1
        self.stop_returned.set()


class StoppableBlockedSource:
    def __init__(self) -> None:
        self.stopped = threading.Event()
        self.stop_returned = asyncio.Event()

    async def prepare(self) -> None:
        return None

    async def frames(self) -> AsyncIterator[AudioFrame]:
        yield make_frame(frame_index=0, samples=[0.1], timestamp_ms=100)
        while not self.stopped.is_set():
            await asyncio.sleep(0.001)

    async def stop(self) -> None:
        self.stopped.set()
        self.stop_returned.set()


class FinishOnStopSource:
    def __init__(self) -> None:
        self.stop_requested = threading.Event()
        self.stop_calls = 0

    async def prepare(self) -> None:
        return None

    async def frames(self) -> AsyncIterator[AudioFrame]:
        yield make_frame(frame_index=0, samples=[0.1], timestamp_ms=100)
        await asyncio.to_thread(self.stop_requested.wait)
        yield AudioFrame.finish(stream_id=0, frame_index=1, timestamp_ms=200)

    async def stop(self) -> None:
        self.stop_calls += 1
        self.stop_requested.set()


async def next_events(bus: FrontendEventBus, count: int):
    return [await asyncio.wait_for(bus.next_event(), timeout=0.2) for _ in range(count)]


async def wait_for_dropped_frames(pipeline: LanePipeline, expected: int) -> None:
    async with asyncio.timeout(0.2):
        while pipeline.metrics.dropped_frames < expected:
            await asyncio.sleep(0.001)


async def wait_for_backlogged_frames(pipeline: LanePipeline, expected: int) -> None:
    async with asyncio.timeout(0.2):
        while pipeline.metrics.backlogged_frames < expected:
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_natural_finish_flushes_non_empty_online_remainder():
    source = AcknowledgedAudioSource(
        [
            make_frame(frame_index=0, samples=[1, 2], timestamp_ms=100),
            make_frame(frame_index=1, samples=[3, 4], timestamp_ms=200),
            AudioFrame.finish(0, 2, 300),
        ]
    )
    online = AcknowledgingOnlineRecognizer(
        [
            make_online(text="hel", timestamp_ms=100),
            make_online(text="hello", timestamp_ms=200),
        ],
        source,
    )
    offline = FakeOfflineRecognizer(text="offline hello")
    bus = FrontendEventBus()
    recorder = ModelOutputRecorder(emit_stdout=False)
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
        model_output_recorder=recorder,
    )

    await pipeline.run(source)
    events = await next_events(bus, 4)

    assert [
        (event.status, event.text, event.sentence_id, event.revision)
        for event in events
    ] == [
        (EventStatus.PARTIAL, "hello", 0, 2),
        (EventStatus.PENDING, "hello", 0, 3),
        (EventStatus.FINAL, "offline hello", 0, 4),
        (EventStatus.LANE_FINISHED, "", 1, 0),
    ]
    np.testing.assert_array_equal(
        offline.requests[0][0], np.asarray([1, 2, 3, 4], dtype=np.float32)
    )
    assert offline.requests[0][1] == "hello"
    assert [
        (record["model"], record["sentence_id"], record["status"], record["text"])
        for record in recorder.as_dicts()
    ] == [
        ("zipformer", 0, "endpoint", "hello"),
        ("qwen3-asr", 0, "final", "offline hello"),
    ]
    assert pipeline.metrics.offline_requests == 1
    assert pipeline.metrics.final_state is LaneFinalState.FINISHED


@pytest.mark.asyncio
async def test_natural_finish_with_empty_residual_skips_offline_request():
    source = AcknowledgedAudioSource(
        [
            make_frame(frame_index=0, samples=[1, 2], timestamp_ms=100),
            AudioFrame.finish(0, 1, 200),
        ]
    )
    offline = FakeOfflineRecognizer()
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=AcknowledgingOnlineRecognizer(
            [make_online(text="", timestamp_ms=100)],
            source,
        ),
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
    )

    await pipeline.run(source)
    finished = await asyncio.wait_for(bus.next_event(), timeout=0.2)

    assert finished.status is EventStatus.LANE_FINISHED
    assert offline.requests == []
    assert pipeline.metrics.offline_requests == 0


@pytest.mark.asyncio
async def test_manual_stop_discards_unendpointed_remainder():
    source = StoppableBlockedSource()
    online = FakeOnlineRecognizer(
        [make_online(text="unfinished", timestamp_ms=100)]
    )
    offline = FakeOfflineRecognizer()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    run_task = asyncio.create_task(pipeline.run(source))
    async with asyncio.timeout(0.5):
        while not online.accepted_frames:
            await asyncio.sleep(0.001)
    pipeline.request_stop()
    await source.stop()
    await asyncio.wait_for(run_task, timeout=0.5)

    assert offline.requests == []


@pytest.mark.asyncio
async def test_manual_stop_discards_finish_sentinel_delivered_during_stop(
    monkeypatch: pytest.MonkeyPatch,
):
    finish_dequeued = threading.Event()
    release_finish = threading.Event()
    original_wait_get = OnlineFrameQueue.wait_get

    def gated_wait_get(
        queue: OnlineFrameQueue,
        timeout: float | None = None,
    ) -> AudioFrame | None:
        frame = original_wait_get(queue, timeout)
        if frame is not None and frame.is_finish:
            finish_dequeued.set()
            release_finish.wait()
        return frame

    monkeypatch.setattr(OnlineFrameQueue, "wait_get", gated_wait_get)
    source = FinishOnStopSource()
    offline = FakeOfflineRecognizer()
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [make_online(text="unfinished", timestamp_ms=100)]
        ),
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    run_task = asyncio.create_task(pipeline.run(source))
    partial = await asyncio.wait_for(bus.next_event(), timeout=0.5)
    assert partial.status is EventStatus.PARTIAL

    await source.stop()
    try:
        await asyncio.wait_for(asyncio.to_thread(finish_dequeued.wait), timeout=0.5)
        pipeline.request_stop()
    finally:
        release_finish.set()
    await asyncio.wait_for(run_task, timeout=0.5)

    assert offline.requests == []
    assert bus.snapshot_results()[(0, 0)]["status"] is EventStatus.PARTIAL


@pytest.mark.asyncio
async def test_empty_endpoint_retains_rule2_tail_for_next_sentence():
    source = AcknowledgedAudioSource(
        [
            make_frame(
                frame_index=0,
                samples=[0, 1, 2, 3, 4, 5, 6, 7],
                timestamp_ms=800,
            ),
            make_frame(frame_index=1, samples=[8, 9], timestamp_ms=1000),
            AudioFrame.finish(0, 2, 1100),
        ],
    )
    online = AcknowledgingOnlineRecognizer(
        [
            make_online(text="", timestamp_ms=800, is_endpoint=True),
            make_online(text="next", timestamp_ms=1000, is_endpoint=True),
        ],
        source,
    )
    offline = FakeOfflineRecognizer()
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
        endpoint=EndpointConfig(
            rule2_min_trailing_silence=0.5,
            cut_margin_seconds=0.1,
        ),
    )

    await pipeline.run(source)
    events = await next_events(bus, 4)

    assert [
        (event.status, event.text, event.sentence_id, event.revision)
        for event in events
    ] == [
        (EventStatus.PARTIAL, "next", 1, 1),
        (EventStatus.PENDING, "next", 1, 2),
        (EventStatus.FINAL, "offline final", 1, 3),
        (EventStatus.LANE_FINISHED, "", 2, 0),
    ]
    assert all(event.sentence_id != 0 for event in events)
    np.testing.assert_array_equal(
        offline.requests[0][0],
        np.asarray([3, 4, 5, 6, 7, 8, 9], dtype=np.float32),
    )
    assert offline.requests[0][1] == "next"
    assert pipeline.metrics.offline_requests == 1


@pytest.mark.asyncio
async def test_token_timestamps_cut_adjacent_offline_segments_without_overlap():
    source = AcknowledgedAudioSource(
        [
            make_frame(frame_index=0, samples=[0, 1, 2, 3], timestamp_ms=400),
            make_frame(frame_index=1, samples=[4, 5, 6, 7], timestamp_ms=800),
            AudioFrame.finish(0, 2, 900),
        ]
    )
    online = AcknowledgingOnlineRecognizer(
        [
            make_online(
                text="one",
                timestamp_ms=400,
                is_endpoint=True,
                last_token_stream_s=0.2,
            ),
            make_online(
                text="two",
                timestamp_ms=800,
                is_endpoint=True,
                last_token_stream_s=0.6,
            ),
        ],
        source,
    )
    offline = FakeOfflineRecognizer()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
        endpoint=EndpointConfig(cut_margin_seconds=0.1),
    )

    await pipeline.run(source)

    assert online.start_run_count == 1
    np.testing.assert_array_equal(
        offline.requests[0][0], np.asarray([0, 1, 2], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        offline.requests[1][0], np.asarray([3, 4, 5, 6], dtype=np.float32)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("token_time", [None, float("nan"), -1.0])
async def test_invalid_token_time_falls_back_to_complete_cache(
    token_time: float | None,
):
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[1, 2, 3], timestamp_ms=300),
            AudioFrame.finish(0, 1, 400),
        ],
        pause_between_frames=True,
    )
    online = FakeOnlineRecognizer(
        [
            make_online(
                text="one",
                timestamp_ms=300,
                is_endpoint=True,
                last_token_stream_s=token_time,
            )
        ]
    )
    offline = FakeOfflineRecognizer()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
        endpoint=EndpointConfig(),
    )

    await pipeline.run(source)
    np.testing.assert_array_equal(
        offline.requests[0][0], np.asarray([1, 2, 3], dtype=np.float32)
    )


@pytest.mark.asyncio
async def test_overflowed_cut_time_falls_back_to_complete_cache():
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[1, 2, 3], timestamp_ms=300),
            AudioFrame.finish(0, 1, 400),
        ],
        pause_between_frames=True,
    )
    online = FakeOnlineRecognizer(
        [
            make_online(
                text="one",
                timestamp_ms=300,
                is_endpoint=True,
                last_token_stream_s=sys.float_info.max,
            )
        ]
    )
    offline = FakeOfflineRecognizer()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=10, capacity_seconds=15.0),
        endpoint=EndpointConfig(
            rule2_min_trailing_silence=sys.float_info.max,
            cut_margin_seconds=sys.float_info.max / 2,
        ),
    )

    await pipeline.run(source)

    np.testing.assert_array_equal(
        offline.requests[0][0], np.asarray([1, 2, 3], dtype=np.float32)
    )
    assert pipeline.metrics.offline_requests == 1


@pytest.mark.asyncio
async def test_non_empty_endpoint_emits_pending_then_final_for_same_sentence():
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            make_frame(frame_index=1, samples=[0.3, 0.4], timestamp_ms=200),
            make_frame(stream_id=0, frame_index=2, samples=[0.0], timestamp_ms=300),
        ],
        pause_between_frames=True,
    )
    source._frames[-1] = source._frames[-1].__class__.finish(
        stream_id=0, frame_index=2, timestamp_ms=300
    )
    online = FakeOnlineRecognizer(
        [
            make_online(text="hello", timestamp_ms=100),
            make_online(text="hello", timestamp_ms=200, is_endpoint=True),
        ]
    )
    offline = FakeOfflineRecognizer(text="offline hello")
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    await pipeline.run(source)
    events = await next_events(bus, 4)

    assert [(event.status, event.sentence_id, event.text) for event in events] == [
        (EventStatus.PARTIAL, 0, "hello"),
        (EventStatus.PENDING, 0, "hello"),
        (EventStatus.FINAL, 0, "offline hello"),
        (EventStatus.LANE_FINISHED, 1, ""),
    ]
    assert offline.requests[0][1] == "hello"


@pytest.mark.asyncio
async def test_pipeline_records_zipformer_endpoint_and_qwen_final_model_outputs():
    recorder = ModelOutputRecorder(emit_stdout=False)
    shared_timings = SharedTimingsTracker()
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [
                make_online(text="online endpoint", timestamp_ms=100, is_endpoint=True),
            ]
        ),
        offline_recognizer=RawResponseOfflineRecognizer(),
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        model_output_recorder=recorder,
        shared_timings_tracker=shared_timings,
    )
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            AudioFrame.finish(stream_id=0, frame_index=1, timestamp_ms=200),
        ],
        pause_between_frames=True,
    )

    await pipeline.run(source)

    records = recorder.as_dicts()
    assert records[0]["model"] == "zipformer"
    assert records[0]["stream_id"] == 0
    assert records[0]["sentence_id"] == 0
    assert records[0]["status"] == "endpoint"
    assert records[0]["text"] == "online endpoint"
    assert records[1]["model"] == "qwen3-asr"
    assert records[1]["status"] == "final"
    assert records[1]["text"] == "offline final"
    assert records[1]["audio_seconds"] == 1.0
    assert records[1]["elapsed_ms"] == 400.0
    assert records[1]["rtf"] == 0.4
    assert records[1]["raw_response"] == {"text": "<asr_text>offline final</asr_text>"}
    assert shared_timings.as_dict()["total_ms"]["average_ms"] == 330.0


@pytest.mark.asyncio
async def test_latency_tracks_online_accept_time_and_rtf_tracks_offline_time():
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            make_frame(frame_index=1, samples=[0.3, 0.4], timestamp_ms=200),
            make_frame(stream_id=0, frame_index=2, samples=[0.0], timestamp_ms=300),
        ],
        pause_between_frames=True,
    )
    source._frames[-1] = source._frames[-1].__class__.finish(
        stream_id=0, frame_index=2, timestamp_ms=300
    )
    online = FakeOnlineRecognizer(
        [
            make_online(text="hello", timestamp_ms=100),
            make_online(text="hello", timestamp_ms=200, is_endpoint=True),
        ]
    )
    offline = FakeOfflineRecognizer(text="offline hello")
    latency = LatencyTracker()
    rtf = RtfTracker()
    clock_values = iter([1.0, 1.050, 2.0, 2.025])
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        latency_tracker=latency,
        rtf_tracker=rtf,
        monotonic=lambda: next(clock_values),
    )

    await pipeline.run(source)

    assert latency.samples_ms == pytest.approx([50.0, 25.0])
    assert latency.average_ms == pytest.approx(37.5)
    assert rtf.values == pytest.approx([0.4])


@pytest.mark.asyncio
async def test_pipeline_uses_online_frame_queue_and_counts_dropped_frames_when_online_is_slow():
    frames = [
        make_frame(frame_index=index, samples=[float(index)], timestamp_ms=100 * (index + 1))
        for index in range(5)
    ]
    frames.append(frames[-1].__class__.finish(stream_id=0, frame_index=5, timestamp_ms=600))
    source = GatedBurstAudioSource(frames)
    online = BlockingOnlineRecognizer()
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=FakeOfflineRecognizer(),
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    run_task = asyncio.create_task(pipeline.run(source))
    await asyncio.wait_for(asyncio.to_thread(online.started.wait), timeout=0.2)
    try:
        source.release_burst.set()
        await asyncio.wait_for(
            asyncio.to_thread(source.finish_yield_started.wait),
            timeout=0.2,
        )
        await wait_for_backlogged_frames(pipeline, expected=4)
    finally:
        online.release.set()
    await asyncio.wait_for(run_task, timeout=0.2)
    finished = await asyncio.wait_for(bus.next_event(), timeout=0.2)

    assert finished.status is EventStatus.LANE_FINISHED
    assert [frame.frame_index for frame in online.accepted_frames] == [0, 1, 2, 3, 4]
    assert pipeline.metrics.processed_frames == 5
    assert pipeline.metrics.dropped_frames == 0
    assert pipeline.metrics.backlogged_frames == 4


@pytest.mark.asyncio
async def test_audio_producer_continues_when_zipformer_consumer_blocks_synchronously():
    frames = [
        make_frame(frame_index=index, samples=[float(index)], timestamp_ms=100 * (index + 1))
        for index in range(5)
    ]
    frames.append(frames[-1].__class__.finish(stream_id=0, frame_index=5, timestamp_ms=600))
    source = GatedBurstAudioSource(frames)
    online = SynchronousSlowOnlineRecognizer()
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=FakeOfflineRecognizer(),
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    run_task = asyncio.create_task(pipeline.run(source))
    await asyncio.wait_for(asyncio.to_thread(online.started.wait), timeout=0.2)
    try:
        source.release_burst.set()
        await asyncio.wait_for(
            asyncio.to_thread(source.finish_yield_started.wait),
            timeout=0.2,
        )
        await wait_for_backlogged_frames(pipeline, expected=4)
    finally:
        online.release.set()
    await asyncio.wait_for(run_task, timeout=0.2)
    finished = await asyncio.wait_for(bus.next_event(), timeout=0.2)

    assert finished.status is EventStatus.LANE_FINISHED
    assert source.prepare_calls == 1
    assert source.stop_calls == 1
    assert [frame.frame_index for frame in online.accepted_frames] == [0, 1, 2, 3, 4]
    assert pipeline.metrics.processed_frames == 5
    assert pipeline.metrics.dropped_frames == 0
    assert pipeline.metrics.backlogged_frames == 4


@pytest.mark.asyncio
async def test_final_empty_string_replaces_pending_text():
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            make_frame(frame_index=1, samples=[0.3, 0.4], timestamp_ms=200),
            make_frame(stream_id=0, frame_index=2, samples=[0.0], timestamp_ms=300),
        ],
        pause_between_frames=True,
    )
    source._frames[-1] = source._frames[-1].__class__.finish(
        stream_id=0, frame_index=2, timestamp_ms=300
    )
    online = FakeOnlineRecognizer(
        [
            make_online(text="draft", timestamp_ms=100),
            make_online(text="draft", timestamp_ms=200, is_endpoint=True),
        ]
    )
    offline = FakeOfflineRecognizer(text="")
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    await pipeline.run(source)
    events = await next_events(bus, 4)

    assert [(event.status, event.text) for event in events] == [
        (EventStatus.PARTIAL, "draft"),
        (EventStatus.PENDING, "draft"),
        (EventStatus.FINAL, ""),
        (EventStatus.LANE_FINISHED, ""),
    ]
    assert bus.snapshot_results()[(0, 0)]["text"] == ""
    assert bus.snapshot_results()[(0, 0)]["status"] is EventStatus.FINAL


@pytest.mark.asyncio
async def test_finish_resets_online_recognizer_and_waits_for_submitted_offline_work():
    release = asyncio.Event()
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            make_frame(frame_index=1, samples=[0.3, 0.4], timestamp_ms=200),
            make_frame(stream_id=0, frame_index=2, samples=[0.0], timestamp_ms=300),
        ],
        pause_between_frames=True,
    )
    source._frames[-1] = source._frames[-1].__class__.finish(
        stream_id=0, frame_index=2, timestamp_ms=300
    )
    online = FakeOnlineRecognizer(
        [
            make_online(text="hello", timestamp_ms=100),
            make_online(text="hello", timestamp_ms=200, is_endpoint=True),
        ]
    )
    offline = FakeOfflineRecognizer(release=release)
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    run_task = asyncio.create_task(pipeline.run(source))
    partial = await asyncio.wait_for(bus.next_event(), timeout=0.2)
    pending = await asyncio.wait_for(bus.next_event(), timeout=0.2)

    assert partial.status is EventStatus.PARTIAL
    assert pending.status is EventStatus.PENDING
    for _ in range(40):
        if online.reset_count == 1:
            break
        await asyncio.sleep(0.005)
    assert online.reset_count == 1
    assert not run_task.done()

    release.set()
    await asyncio.wait_for(run_task, timeout=0.2)
    final = await asyncio.wait_for(bus.next_event(), timeout=0.2)
    finished = await asyncio.wait_for(bus.next_event(), timeout=0.2)

    assert final.status is EventStatus.FINAL
    assert finished.status is EventStatus.LANE_FINISHED
    assert pipeline.metrics.final_state is LaneFinalState.FINISHED


@pytest.mark.asyncio
async def test_sentence_id_increments_only_after_endpoint_handling():
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[1.0], timestamp_ms=100),
            make_frame(frame_index=1, samples=[2.0], timestamp_ms=200),
            make_frame(frame_index=2, samples=[3.0], timestamp_ms=300),
            make_frame(frame_index=3, samples=[4.0], timestamp_ms=400),
            make_frame(stream_id=0, frame_index=4, samples=[0.0], timestamp_ms=500),
        ],
        pause_between_frames=True,
    )
    source._frames[-1] = source._frames[-1].__class__.finish(
        stream_id=0, frame_index=4, timestamp_ms=500
    )
    online = FakeOnlineRecognizer(
        [
            make_online(text="one", timestamp_ms=100),
            make_online(text="one", timestamp_ms=200, is_endpoint=True),
            make_online(text="two", timestamp_ms=300),
            make_online(text="two", timestamp_ms=400, is_endpoint=True),
        ]
    )
    offline = FakeOfflineRecognizer(text="offline final")
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    await pipeline.run(source)
    events = await next_events(bus, 7)

    event_tuples = [(event.status, event.sentence_id, event.text) for event in events]
    assert event_tuples[:2] == [
        (EventStatus.PARTIAL, 0, "one"),
        (EventStatus.PENDING, 0, "one"),
    ]
    assert (EventStatus.PARTIAL, 1, "two") in event_tuples
    assert (EventStatus.PENDING, 1, "two") in event_tuples
    assert event_tuples.count((EventStatus.FINAL, 0, "offline final")) == 1
    assert event_tuples.count((EventStatus.FINAL, 1, "offline final")) == 1
    assert event_tuples[-1] == (EventStatus.LANE_FINISHED, 2, "")
    assert [prompt for _, prompt in offline.requests] == ["one", "two"]


@pytest.mark.asyncio
async def test_wait_for_offline_tail_preserves_current_final_result():
    release = asyncio.Event()
    offline = FakeOfflineRecognizer(text="drained final", release=release)
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [make_online(text="endpoint", timestamp_ms=100, is_endpoint=True)]
        ),
        offline_recognizer=offline,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )
    source = FakeAudioSource(
        [make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100)],
        pause_between_frames=True,
    )

    await pipeline.run(source)
    drain_task = asyncio.create_task(pipeline.wait_for_offline_tail(timeout_seconds=0.2))
    await asyncio.sleep(0)

    assert not drain_task.done()
    release.set()
    await drain_task
    partial, pending, final = await next_events(bus, 3)
    assert [partial.status, pending.status, final.status] == [
        EventStatus.PARTIAL,
        EventStatus.PENDING,
        EventStatus.FINAL,
    ]
    assert final.text == "drained final"


@pytest.mark.asyncio
async def test_wait_for_offline_tail_cancels_remaining_work_on_timeout():
    offline = CancellationAwareOfflineRecognizer()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [make_online(text="endpoint", timestamp_ms=100, is_endpoint=True)]
        ),
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )
    source = FakeAudioSource(
        [make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100)],
        pause_between_frames=True,
    )

    await pipeline.run(source)

    with pytest.raises(TimeoutError):
        await pipeline.wait_for_offline_tail(timeout_seconds=0.01)

    assert offline.cancelled is True
    assert not pipeline._offline_tasks


@pytest.mark.asyncio
async def test_qwen_failure_preserves_pending_and_allows_later_offline_requests():
    recognizer = FailOnceOfflineRecognizer()
    failures = []
    successes = []
    bus = FrontendEventBus()
    pipeline = LanePipeline(
        stream_id=7,
        online_recognizer=FakeOnlineRecognizer([]),
        offline_recognizer=recognizer,
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        on_qwen_success=successes.append,
        on_qwen_error=failures.append,
    )
    pipeline._publish_event(
        sentence_id=1,
        revision=3,
        status=EventStatus.PENDING,
        text="first pending",
        timestamp_ms=100,
    )
    pipeline._schedule_offline(
        sentence_id=1,
        revision=4,
        prompt="first pending",
        pcm=np.asarray([0.1], dtype=np.float32),
        timestamp_ms=100,
    )
    pipeline._schedule_offline(
        sentence_id=2,
        revision=1,
        prompt="second pending",
        pcm=np.asarray([0.2], dtype=np.float32),
        timestamp_ms=200,
    )

    await pipeline.wait_for_offline_tail(timeout_seconds=1)
    events = await next_events(bus, 2)

    assert recognizer.calls == 2
    assert [failure.sentence_id for failure in failures] == [1]
    assert failures[0].stream_id == 7
    assert failures[0].revision == 4
    assert failures[0].pending_text == "first pending"
    assert failures[0].timestamp_ms == 100
    assert list(failures[0].request) == [
        "category",
        "error_type",
        "message",
        "method",
        "url",
        "timeout_ms",
        "audio_seconds",
        "sample_count",
        "status_code",
        "response_body",
        "response_body_truncated",
    ]
    assert successes == [2]
    assert pipeline._offline_error is None
    assert any(event.status is EventStatus.PENDING and event.sentence_id == 1 for event in events)
    assert not any(event.status is EventStatus.FINAL and event.sentence_id == 1 for event in events)
    assert any(
        event.status is EventStatus.FINAL
        and event.sentence_id == 2
        and event.text == "second final"
        for event in events
    )


def test_qwen_failure_event_uses_offline_nonfinal_serialization():
    event = ASREvent.asr_result(
        stream_id=7,
        sentence_id=1,
        revision=4,
        status=EventStatus.QWEN_ERROR,
        text="",
        timestamp_ms=100,
    )

    assert event.result_type is ResultType.OFFLINE
    assert event.is_final is False


@pytest.mark.asyncio
async def test_cancel_offline_tasks_cancels_work_and_drains_without_timeout():
    offline = CancellationAwareOfflineRecognizer()
    qwen_errors = []
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer([]),
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        on_qwen_error=qwen_errors.append,
    )
    pipeline._schedule_offline(
        sentence_id=1,
        revision=1,
        prompt="blocking pending",
        pcm=np.asarray([0.1], dtype=np.float32),
        timestamp_ms=100,
    )
    task = pipeline._offline_tail
    assert task is not None
    await asyncio.sleep(0)

    pipeline.cancel_offline_tasks()
    await asyncio.wait_for(pipeline.wait_for_offline_tail(timeout_seconds=10), timeout=0.1)

    assert task.cancelled() is True
    assert offline.cancelled is True
    assert not pipeline._offline_tasks
    assert qwen_errors == []


@pytest.mark.asyncio
async def test_fatal_cancellation_rejects_late_endpoint_schedule_from_inflight_accept():
    class BlockingEndpointOnlineRecognizer:
        def __init__(self) -> None:
            self.accept_started = threading.Event()
            self.release_accept = threading.Event()

        async def accept(self, frame: AudioFrame) -> OnlineRecognition:
            self.accept_started.set()
            self.release_accept.wait()
            return make_online(
                text="late endpoint",
                timestamp_ms=frame.timestamp_ms,
                is_endpoint=True,
            )

        async def reset(self) -> None:
            return None

    class BlockingOfflineRecognizer:
        def __init__(self) -> None:
            self.requests: list[tuple[np.ndarray, str]] = []

        async def recognize(self, pcm: np.ndarray, prompt: str) -> OfflineRecognitionResult:
            self.requests.append((pcm.copy(), prompt))
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    source = FinishOnStopSource()
    online = BlockingEndpointOnlineRecognizer()
    offline = BlockingOfflineRecognizer()
    qwen_errors = []
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        on_qwen_error=qwen_errors.append,
    )
    run_task = asyncio.create_task(pipeline.run(source))
    await asyncio.wait_for(asyncio.to_thread(online.accept_started.wait), timeout=0.2)

    pipeline.request_stop()
    pipeline.cancel_offline_tasks()
    await source.stop()
    online.release_accept.set()

    await asyncio.wait_for(run_task, timeout=0.2)
    await asyncio.wait_for(
        pipeline.wait_for_offline_tail(timeout_seconds=10.0),
        timeout=0.1,
    )

    assert offline.requests == []
    assert qwen_errors == []


@pytest.mark.asyncio
async def test_worker_failure_waits_for_blocked_peer_before_run_returns():
    online = BlockingOnlineRecognizer()
    source = FailingWhileConsumerBlockedSource(online.started)
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=FakeOfflineRecognizer(),
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )
    run_task = asyncio.create_task(pipeline.run(source))

    try:
        await asyncio.wait_for(asyncio.to_thread(source.failed.wait), timeout=0.5)
        await asyncio.wait_for(source.stop_returned.wait(), timeout=0.5)
        assert not run_task.done()
    finally:
        online.release.set()

    with pytest.raises(RuntimeError, match="producer failed while consumer blocked"):
        await asyncio.wait_for(run_task, timeout=0.5)
    assert source.stop_calls >= 1


@pytest.mark.asyncio
async def test_caller_cancellation_waits_for_blocked_worker_before_propagating():
    online = BlockingOnlineRecognizer()
    source = StoppableBlockedSource()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=FakeOfflineRecognizer(),
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )
    run_task = asyncio.create_task(pipeline.run(source))
    await asyncio.wait_for(asyncio.to_thread(online.started.wait), timeout=0.5)

    try:
        run_task.cancel()
        await asyncio.wait_for(source.stop_returned.wait(), timeout=0.5)
        assert not run_task.done()
    finally:
        online.release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=0.5)


@pytest.mark.asyncio
async def test_cancelling_offline_tail_wait_cancels_and_awaits_offline_work():
    offline = CancellationAwareOfflineRecognizer()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [make_online(text="endpoint", timestamp_ms=100, is_endpoint=True)]
        ),
        offline_recognizer=offline,
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )
    source = FakeAudioSource(
        [make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100)],
        pause_between_frames=True,
    )
    await pipeline.run(source)

    wait_task = asyncio.create_task(
        pipeline.wait_for_offline_tail(timeout_seconds=10.0)
    )
    await asyncio.sleep(0)
    wait_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await wait_task
    assert offline.cancelled is True
    await pipeline.wait_for_offline_tail(timeout_seconds=0.0)


def test_concurrent_lane_metric_increments_preserve_combined_counters():
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer([]),
        offline_recognizer=FakeOfflineRecognizer(),
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )
    start = threading.Barrier(3)
    errors: list[BaseException] = []

    def increment(field: str) -> None:
        try:
            start.wait()
            for _ in range(1_000):
                pipeline._increment_metrics(**{field: 1})
        except BaseException as exc:
            errors.append(exc)

    workers = [
        threading.Thread(target=increment, args=("processed_frames",)),
        threading.Thread(target=increment, args=("dropped_frames",)),
    ]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join()

    assert errors == []
    assert pipeline.metrics.processed_frames == 1_000
    assert pipeline.metrics.dropped_frames == 1_000


@pytest.mark.asyncio
async def test_source_ready_callback_runs_after_prepare_before_frames():
    actions: list[str] = []

    class RecordingSource:
        async def prepare(self) -> None:
            actions.append("prepare")

        async def frames(self) -> AsyncIterator[AudioFrame]:
            actions.append("frames")
            yield AudioFrame.finish(0, 0, 0)

        async def stop(self) -> None:
            actions.append("stop")

    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer([]),
        offline_recognizer=FakeOfflineRecognizer(),
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    await pipeline.run(
        RecordingSource(),
        on_source_ready=lambda: actions.append("ready"),
    )

    assert actions[:3] == ["prepare", "ready", "frames"]


@pytest.mark.asyncio
async def test_source_ready_callback_is_not_called_when_prepare_fails():
    ready_calls = 0

    class FailingPrepareSource:
        async def prepare(self) -> None:
            raise RuntimeError("input open failed")

        async def frames(self) -> AsyncIterator[AudioFrame]:
            if False:
                yield AudioFrame.finish(0, 0, 0)

        async def stop(self) -> None:
            return None

    def mark_ready() -> None:
        nonlocal ready_calls
        ready_calls += 1

    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer([]),
        offline_recognizer=FakeOfflineRecognizer(),
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
    )

    with pytest.raises(RuntimeError, match="input open failed"):
        await pipeline.run(FailingPrepareSource(), on_source_ready=mark_ready)

    assert ready_calls == 0


class RecordingDetailWriter:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, fields: dict) -> None:
        self.records.append(fields)


@pytest.mark.asyncio
async def test_frame_latency_writer_records_one_entry_per_accept_call():
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            make_frame(frame_index=1, samples=[0.3, 0.4], timestamp_ms=200),
            AudioFrame.finish(stream_id=0, frame_index=2, timestamp_ms=300),
        ],
        pause_between_frames=True,
    )
    online = FakeOnlineRecognizer(
        [
            make_online(text="hello", timestamp_ms=100),
            make_online(text="hello", timestamp_ms=200, is_endpoint=True),
        ]
    )
    frame_latency_writer = RecordingDetailWriter()
    clock_values = iter([1.0, 1.050, 2.0, 2.025])
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=online,
        offline_recognizer=FakeOfflineRecognizer(text="offline hello"),
        event_bus=FrontendEventBus(),
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        frame_latency_writer=frame_latency_writer,
        monotonic=lambda: next(clock_values),
    )

    await pipeline.run(source)

    assert [entry["latency_ms"] for entry in frame_latency_writer.records] == pytest.approx(
        [50.0, 25.0]
    )
    assert all("t_ms" not in entry for entry in frame_latency_writer.records)


@pytest.mark.asyncio
async def test_rtf_writer_records_sentence_id_and_offline_timing():
    bus = FrontendEventBus()
    rtf_writer = RecordingDetailWriter()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [make_online(text="online endpoint", timestamp_ms=100, is_endpoint=True)]
        ),
        offline_recognizer=FakeOfflineRecognizer(text="offline final"),
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        rtf_writer=rtf_writer,
    )
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            AudioFrame.finish(stream_id=0, frame_index=1, timestamp_ms=200),
        ],
        pause_between_frames=True,
    )

    await pipeline.run(source)

    assert rtf_writer.records == [
        {"sentence_id": 0, "audio_seconds": 1.0, "elapsed_ms": 400.0, "rtf": 0.4}
    ]


@pytest.mark.asyncio
async def test_shared_timing_writer_records_sentence_id_and_filtered_fields():
    bus = FrontendEventBus()
    shared_timing_writer = RecordingDetailWriter()
    pipeline = LanePipeline(
        stream_id=0,
        online_recognizer=FakeOnlineRecognizer(
            [make_online(text="online endpoint", timestamp_ms=100, is_endpoint=True)]
        ),
        offline_recognizer=RawResponseOfflineRecognizer(),
        event_bus=bus,
        segment_cache=SegmentCache(sample_rate=16000, capacity_seconds=15.0),
        shared_timing_writer=shared_timing_writer,
    )
    source = FakeAudioSource(
        [
            make_frame(frame_index=0, samples=[0.1, 0.2], timestamp_ms=100),
            AudioFrame.finish(stream_id=0, frame_index=1, timestamp_ms=200),
        ],
        pause_between_frames=True,
    )

    await pipeline.run(source)

    assert shared_timing_writer.records == [
        {
            "sentence_id": 0,
            "queue_ms": 10.0,
            "encode_ms": 100.0,
            "prefill_ms": 20.0,
            "decode_ms": 200.0,
            "total_ms": 330.0,
        }
    ]
    assert set(SHARED_TIMING_FIELDS) == {"queue_ms", "encode_ms", "prefill_ms", "decode_ms", "total_ms"}
