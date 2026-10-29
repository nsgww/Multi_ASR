import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from multi_stream_asr.buffer.audio_queue import OnlineFrameQueue
from multi_stream_asr.buffer.segment_cache import SegmentCache
from multi_stream_asr.config import AppConfig
from multi_stream_asr.log.metrics import (
    EventLoopLagTracker,
    LatencyTracker,
    RtfTracker,
    SharedTimingsTracker,
    _nearest_rank_percentile,
    merge_latency_summaries,
    merge_rtf_summaries,
    merge_shared_timing_summaries,
)


def shared_summary(queue_ms: float, encode_ms: float) -> dict[str, object]:
    fields = {
        "queue_ms": queue_ms,
        "encode_ms": encode_ms,
        "prefill_ms": 20.0,
        "decode_ms": 200.0,
        "total_ms": queue_ms + encode_ms + 220.0,
    }
    return {
        "count": 1,
        **{
            key: {
                "count": 1,
                "total_ms": value,
                "average_ms": value,
                "max_ms": value,
            }
            for key, value in fields.items()
        },
    }


def test_shared_timing_tracker_records_only_known_finite_fields():
    tracker = SharedTimingsTracker()
    tracker.add(
        {
            "queue_ms": 10.0,
            "encode_ms": 100.0,
            "prefill_ms": 20.0,
            "decode_ms": 200.0,
            "total_ms": 330.0,
            "unknown_ms": 1.0,
        }
    )
    tracker.add({"queue_ms": 30.0, "total_ms": 770.0})

    payload = tracker.as_dict()

    assert payload["count"] == 2
    assert payload["queue_ms"] == {
        "count": 2,
        "total_ms": 40.0,
        "average_ms": 20.0,
        "p95_ms": 30.0,
        "max_ms": 30.0,
    }
    assert payload["encode_ms"]["count"] == 1
    assert "unknown_ms" not in payload


def test_merge_shared_timing_summaries_weights_by_field_count():
    merged = merge_shared_timing_summaries(
        [
            {
                "count": 2,
                "queue_ms": {
                    "count": 2,
                    "total_ms": 40.0,
                    "average_ms": 20.0,
                    "max_ms": 30.0,
                },
            },
            {
                "count": 1,
                "queue_ms": {
                    "count": 1,
                    "total_ms": 50.0,
                    "average_ms": 50.0,
                    "max_ms": 50.0,
                },
            },
        ]
    )

    assert merged["count"] == 3
    assert merged["queue_ms"] == {
        "count": 3,
        "total_ms": 90.0,
        "average_ms": 30.0,
        "p95_ms": 0.0,
        "max_ms": 50.0,
    }
from multi_stream_asr.log.recorder import RunRecorder
from multi_stream_asr.models import AudioFrame, LaneFinalState, LaneMetrics


def make_frame(stream_id: int, frame_index: int, value: float, samples: int, *, is_finish: bool = False) -> AudioFrame:
    if is_finish:
        return AudioFrame.finish(stream_id=stream_id, frame_index=frame_index, timestamp_ms=frame_index)
    pcm = np.full(samples, value, dtype=np.float32)
    return AudioFrame(
        stream_id=stream_id,
        frame_index=frame_index,
        pcm=pcm,
        timestamp_ms=frame_index,
    )


def test_online_frame_queue_replaces_old_audio_and_preserves_finish():
    queue = OnlineFrameQueue(maxsize=1)
    first = make_frame(0, 1, 1.0, 3)
    second = make_frame(0, 2, 2.0, 3)
    finish = make_frame(0, 3, 0.0, 0, is_finish=True)
    third = make_frame(0, 4, 3.0, 3)

    assert queue.put(first) is True
    assert queue.put(second) is False
    assert queue.dropped_frames == 1
    assert queue.backlogged_frames == 0
    assert queue.get() == second

    assert queue.put(finish) is True
    assert queue.put(third) is False
    assert queue.dropped_frames == 2
    assert queue.backlogged_frames == 0
    assert queue.get() == finish


def test_online_frame_queue_replaces_queued_finish_with_finish_and_counts_drop():
    queue = OnlineFrameQueue(maxsize=1)
    first_finish = make_frame(0, 1, 0.0, 0, is_finish=True)
    second_finish = make_frame(0, 2, 0.0, 0, is_finish=True)

    assert queue.put(first_finish) is True
    assert queue.put(second_finish) is False
    assert queue.dropped_frames == 1
    assert queue.get() == second_finish


def test_online_frame_queue_keeps_backlogged_frames_until_capacity_is_reached():
    queue = OnlineFrameQueue(maxsize=10)
    frames = [make_frame(0, index, float(index), 3) for index in range(3)]

    assert [queue.put(frame) for frame in frames] == [True, True, True]
    assert queue.dropped_frames == 0
    assert queue.backlogged_frames == 2
    assert [queue.get(), queue.get(), queue.get()] == frames


def test_segment_cache_bounds_to_15_seconds_at_16khz_and_resets():
    cache = SegmentCache(sample_rate=16_000, capacity_seconds=15.0)

    cache.put(make_frame(0, 1, 1.0, 120_000))
    cache.put(make_frame(0, 2, 2.0, 120_000))
    cache.put(make_frame(0, 3, 3.0, 32_000))

    audio = cache.audio()

    assert audio.dtype == np.float32
    assert audio.shape == (240_000,)
    np.testing.assert_allclose(audio[:88_000], 1.0)
    np.testing.assert_allclose(audio[88_000:208_000], 2.0)
    np.testing.assert_allclose(audio[208_000:], 3.0)

    cache.reset()
    assert cache.audio().size == 0


def test_segment_cache_cuts_across_chunks_without_overlap_or_gap():
    cache = SegmentCache(sample_rate=10, capacity_seconds=10.0)
    cache.put(make_frame(0, 0, 1.0, 3))
    cache.put(make_frame(0, 1, 2.0, 4))
    cache.put(make_frame(0, 2, 3.0, 3))

    first = cache.cut(0.5)
    second = cache.cut(0.8)
    tail = cache.audio()

    np.testing.assert_array_equal(first, np.asarray([1, 1, 1, 2, 2], dtype=np.float32))
    np.testing.assert_array_equal(second, np.asarray([2, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(tail, np.asarray([3, 3], dtype=np.float32))
    np.testing.assert_array_equal(
        np.concatenate((first, second, tail)),
        np.asarray([1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=np.float32),
    )
    assert cache.start_time_s == 0.8
    assert cache.end_time_s == 1.0


def test_segment_cache_rounds_and_clamps_cut_time():
    cache = SegmentCache(sample_rate=10, capacity_seconds=10.0)
    cache.put(make_frame(0, 0, 1.0, 5))

    np.testing.assert_array_equal(cache.cut(0.26), np.ones(3, dtype=np.float32))
    assert cache.cut(-100.0).size == 0
    np.testing.assert_array_equal(cache.cut(100.0), np.ones(2, dtype=np.float32))
    assert cache.start_time_s == cache.end_time_s == 0.5


def test_segment_cache_clamps_extreme_finite_time_before_sample_conversion():
    cache = SegmentCache(sample_rate=16_000, capacity_seconds=1.0)
    cache.put(make_frame(0, 0, 1.0, 4))

    assert cache.cut(-sys.float_info.max).size == 0
    np.testing.assert_array_equal(
        cache.cut(sys.float_info.max), np.ones(4, dtype=np.float32)
    )


def test_segment_cache_capacity_trim_advances_absolute_start():
    cache = SegmentCache(sample_rate=10, capacity_seconds=0.5)
    cache.put(make_frame(0, 0, 1.0, 4))
    cache.put(make_frame(0, 1, 2.0, 4))

    assert cache.start_time_s == 0.3
    assert cache.end_time_s == 0.8
    np.testing.assert_array_equal(
        cache.audio(), np.asarray([1, 2, 2, 2, 2], dtype=np.float32)
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_segment_cache_rejects_non_finite_cut_time(value: float):
    cache = SegmentCache(sample_rate=10, capacity_seconds=1.0)
    with pytest.raises(ValueError, match="stream_time_s"):
        cache.cut(value)


def test_segment_cache_reset_preserves_monotonic_clock():
    cache = SegmentCache(sample_rate=10, capacity_seconds=1.0)
    cache.put(make_frame(0, 0, 1.0, 4))
    cache.reset()
    assert cache.start_time_s == cache.end_time_s == 0.4
    cache.put(make_frame(0, 1, 2.0, 2))
    assert cache.start_time_s == 0.4
    assert cache.end_time_s == 0.6


def test_latency_and_rtf_trackers_compute_averages():
    latency = LatencyTracker()
    rtf = RtfTracker()

    latency.add(100.0)
    latency.add(300.0)
    rtf.add(audio_seconds=1.0, elapsed_ms=400.0)
    rtf.add(audio_seconds=0.0, elapsed_ms=123.0)

    assert latency.count == 2
    assert latency.total_ms == 400.0
    assert latency.average_ms == 200.0
    assert rtf.count == 2
    assert rtf.values == [0.4, 0.0]
    assert rtf.average == 0.2


def test_metric_trackers_keep_exact_full_run_totals_with_bounded_recent_samples():
    latency = LatencyTracker()
    rtf = RtfTracker()
    shared = SharedTimingsTracker()

    for _ in range(5000):
        latency.add(2.0)
        rtf.add(audio_seconds=1.0, elapsed_ms=500.0)
        shared.add({"total_ms": 3.0})

    assert latency.count == 5000
    assert latency.total_ms == 10000.0
    assert latency.average_ms == 2.0
    assert latency.max_ms == 2.0
    assert len(latency.samples_ms) <= 1024
    assert rtf.count == 5000
    assert rtf.total == 2500.0
    assert rtf.average == 0.5
    assert rtf.max == 0.5
    assert len(rtf.values) <= 1024
    assert shared.as_dict()["count"] == 5000
    assert shared.as_dict()["total_ms"]["total_ms"] == 15000.0
    assert len(shared.samples["total_ms"]) <= 1024


def test_tracker_as_dict_uses_one_snapshot_during_concurrent_add():
    class InterleavingLatencyTracker(LatencyTracker):
        def __init__(self) -> None:
            super().__init__(samples_ms=[10.0])
            self.count_read = threading.Event()
            self.added = threading.Event()

        @property
        def count(self) -> int:
            count = super().count
            self.count_read.set()
            self.added.wait()
            return count

    class InterleavingRtfTracker(RtfTracker):
        def __init__(self) -> None:
            super().__init__(values=[0.5])
            self.count_read = threading.Event()
            self.added = threading.Event()

        @property
        def count(self) -> int:
            count = super().count
            self.count_read.set()
            self.added.wait()
            return count

    def latency_summary() -> dict[str, float | int]:
        tracker = InterleavingLatencyTracker()

        def add() -> None:
            tracker.count_read.wait()
            tracker.add(20.0)
            tracker.added.set()

        writer = threading.Thread(target=add)
        writer.start()
        try:
            return tracker.as_dict()
        finally:
            tracker.count_read.set()
            tracker.added.set()
            writer.join(timeout=1.0)

    def rtf_summary() -> dict[str, float | int]:
        tracker = InterleavingRtfTracker()

        def add() -> None:
            tracker.count_read.wait()
            tracker.add(audio_seconds=1.0, elapsed_ms=1500.0)
            tracker.added.set()

        writer = threading.Thread(target=add)
        writer.start()
        try:
            return tracker.as_dict()
        finally:
            tracker.count_read.set()
            tracker.added.set()
            writer.join(timeout=1.0)

    latency = latency_summary()
    rtf = rtf_summary()

    assert latency["average_ms"] == latency["total_ms"] / latency["count"]
    assert rtf["average"] == rtf["total"] / rtf["count"]


def test_latency_tracker_add_contends_with_single_snapshot():
    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    writer_attempted = threading.Event()
    writer_completed = threading.Event()
    summary: dict[str, float | int] = {}

    class GatedLatencyTracker(LatencyTracker):
        def _snapshot(self):
            with self._lock:
                snapshot_entered.set()
                release_snapshot.wait()
                return (
                    self._count,
                    self._total_ms,
                    self._max_ms,
                    _nearest_rank_percentile(self.samples_ms),
                )

    class SignalingFloat:
        def __float__(self) -> float:
            writer_attempted.set()
            return 20.0

    tracker = GatedLatencyTracker(samples_ms=[10.0])

    def read_summary() -> None:
        summary.update(tracker.as_dict())

    def add_sample() -> None:
        tracker.add(SignalingFloat())
        writer_completed.set()

    reader = threading.Thread(target=read_summary)
    writer = threading.Thread(target=add_sample)
    reader.start()
    assert snapshot_entered.wait(timeout=0.5)
    writer.start()
    assert writer_attempted.wait(timeout=0.5)
    assert not writer_completed.is_set()

    release_snapshot.set()
    reader.join(timeout=0.5)
    writer.join(timeout=0.5)

    assert summary == {
        "count": 1,
        "total_ms": 10.0,
        "average_ms": 10.0,
        "p95_ms": 10.0,
        "max_ms": 10.0,
    }
    assert tracker.as_dict() == {
        "count": 2,
        "total_ms": 30.0,
        "average_ms": 15.0,
        "p95_ms": 20.0,
        "max_ms": 20.0,
    }


def test_summary_mergers_recompute_weighted_averages():
    assert merge_latency_summaries(
        [
            {"count": 2, "total_ms": 100.0, "average_ms": 50.0, "max_ms": 60.0},
            {"count": 1, "total_ms": 90.0, "average_ms": 90.0, "max_ms": 90.0},
        ]
    ) == {
        "count": 3,
        "total_ms": 190.0,
        "average_ms": 190.0 / 3.0,
        "p95_ms": 0.0,
        "max_ms": 90.0,
    }
    assert merge_rtf_summaries(
        [
            {"count": 2, "total": 0.8, "average": 0.4, "max": 0.5},
            {"count": 1, "total": 0.6, "average": 0.6, "max": 0.6},
        ]
    ) == {
        "count": 3,
        "total": 1.4,
        "average": 1.4 / 3.0,
        "p95": 0.0,
        "max": 0.6,
    }


def test_run_recorder_writes_config_metrics_and_results(tmp_path: Path):
    output_dir = tmp_path / "run"
    recorder = RunRecorder()
    config = AppConfig(lanes=4)
    lane_metrics = [
        LaneMetrics(stream_id=0, processed_frames=7, dropped_frames=1, offline_requests=2, final_state=LaneFinalState.FINISHED),
        LaneMetrics(stream_id=1, processed_frames=5, dropped_frames=0, offline_requests=1, final_state=LaneFinalState.STOPPED),
    ]
    latency = LatencyTracker()
    latency.add(50.0)
    latency.add(150.0)
    rtf = RtfTracker()
    rtf.add(audio_seconds=2.0, elapsed_ms=800.0)
    event_loop_lag = EventLoopLagTracker()
    event_loop_lag.add(1.0)
    event_loop_lag.add(5.0)
    results = {
        (0, 0): {"stream_id": 0, "sentence_id": 0, "status": "final", "text": "hello"},
        (0, 1): {"stream_id": 0, "sentence_id": 1, "status": "final", "text": "world"},
        (1, 0): {"stream_id": 1, "sentence_id": 0, "status": "pending", "text": "draft"},
    }

    saved_dir = recorder.save(
        output_dir,
        config=config,
        lane_metrics=lane_metrics,
        latency_tracker=latency,
        rtf_tracker=rtf,
        event_loop_lag_tracker=event_loop_lag,
        results=results,
    )

    assert saved_dir == output_dir
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert not (output_dir / "results.json").exists()
    assert (output_dir / "results" / "lane0.txt").exists()
    assert (output_dir / "results" / "lane1.txt").exists()
    assert (output_dir / "results" / "lane2.txt").exists()
    assert (output_dir / "results" / "lane3.txt").exists()

    config_json = json.loads((output_dir / "config.json").read_text())
    metrics_json = json.loads((output_dir / "metrics.json").read_text())

    assert config_json["lanes"] == 4
    assert config_json["segment_cache_seconds"] == 15.0
    assert metrics_json["metrics_schema_version"] == 5
    assert "latency" not in metrics_json
    assert "rtf" not in metrics_json
    assert metrics_json["online_frame_latency"]["average_ms"] == 100.0
    assert metrics_json["online_frame_latency"]["max_ms"] == 150.0
    assert metrics_json["offline_rtf"]["average"] == 0.4
    assert metrics_json["offline_rtf"]["max"] == 0.4
    assert "values" not in metrics_json["offline_rtf"]
    assert metrics_json["event_loop_lag"]["average_ms"] == 3.0
    assert metrics_json["event_loop_lag"]["max_ms"] == 5.0
    assert metrics_json["lanes"][0]["final_state"] == LaneFinalState.FINISHED.value
    assert (output_dir / "results" / "lane0.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert (output_dir / "results" / "lane1.txt").read_text(encoding="utf-8") == ""
    assert (output_dir / "results" / "lane2.txt").read_text(encoding="utf-8") == ""
    assert (output_dir / "results" / "lane3.txt").read_text(encoding="utf-8") == ""


def test_run_recorder_writes_per_lane_compact_model_outputs_jsonl(tmp_path: Path):
    recorder = RunRecorder()
    config = AppConfig(lanes=2, output_dir=tmp_path)

    run_dir = recorder.save(
        tmp_path / "run",
        config=config,
        model_outputs=[
            {
                "model": "qwen3-asr",
                "stream_id": 0,
                "sentence_id": 1,
                "status": "final",
                "text": "recognized",
                "text_preview": "recognized",
                "audio_seconds": 1.0,
                "elapsed_ms": 400.0,
                "rtf": 0.4,
                "raw_response": {"text": "recognized"},
            },
            {
                "model": "zipformer",
                "stream_id": 1,
                "sentence_id": 0,
                "status": "endpoint",
                "text": "draft",
                "text_preview": "draft",
                "audio_seconds": None,
                "elapsed_ms": None,
                "rtf": None,
                "raw_response": None,
            }
        ],
    )

    assert not (run_dir / "model_outputs.jsonl").exists()
    lane0_lines = (run_dir / "model_outputs" / "lane0.jsonl").read_text(encoding="utf-8").splitlines()
    lane1_lines = (run_dir / "model_outputs" / "lane1.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lane0_lines[0]) == {
        "audio_seconds": 1.0,
        "model": "qwen3-asr",
        "rtf": 0.4,
        "sentence_id": 1,
        "stream_id": 0,
        "text": "recognized",
    }
    assert json.loads(lane1_lines[0]) == {
        "audio_seconds": None,
        "model": "zipformer",
        "rtf": None,
        "sentence_id": 0,
        "stream_id": 1,
        "text": "draft",
    }


def test_run_recorder_save_process_run_writes_schema_v5_without_rewriting_lane_files(tmp_path: Path):
    recorder = RunRecorder()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    run_dir = tmp_path / "run"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "model_outputs").mkdir(parents=True)
    (run_dir / "results" / "lane0.txt").write_text("lane-owned\n", encoding="utf-8")
    (run_dir / "model_outputs" / "lane0.jsonl").write_text('{"lane":0}\n', encoding="utf-8")

    saved = recorder.save_process_run(
        run_dir,
        config=config,
        lane_summaries=[
            {
                "lane_metrics": {
                    "stream_id": 0,
                    "processed_frames": 10,
                    "dropped_frames": 1,
                    "offline_requests": 2,
                    "final_state": "finished",
                },
                "online_frame_latency": {"count": 2, "total_ms": 100.0, "average_ms": 50.0, "max_ms": 60.0},
                "offline_rtf": {"count": 1, "total": 0.4, "average": 0.4, "max": 0.4},
                "offline_shared_timings": shared_summary(10.0, 100.0),
                "event_loop_lag": {"count": 3, "total_ms": 6.0, "average_ms": 2.0, "max_ms": 4.0},
                "results_path": "results/lane0.txt",
                "model_outputs_path": "model_outputs/lane0.jsonl",
            },
            {
                "lane_metrics": {
                    "stream_id": 1,
                    "processed_frames": 20,
                    "dropped_frames": 0,
                    "offline_requests": 3,
                    "final_state": "finished",
                },
                "online_frame_latency": {"count": 2, "total_ms": 120.0, "average_ms": 60.0, "max_ms": 70.0},
                "offline_rtf": {"count": 1, "total": 0.5, "average": 0.5, "max": 0.5},
                "offline_shared_timings": shared_summary(30.0, 300.0),
                "event_loop_lag": {"count": 2, "total_ms": 2.0, "average_ms": 1.0, "max_ms": 1.5},
                "results_path": "results/lane1.txt",
                "model_outputs_path": "model_outputs/lane1.jsonl",
            },
        ],
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved == run_dir
    assert metrics["metrics_schema_version"] == 5
    assert metrics["online_frame_latency"] == {
        "count": 4,
        "total_ms": 220.0,
        "average_ms": 55.0,
        "p95_ms": 0.0,
        "max_ms": 70.0,
    }
    assert metrics["offline_rtf"] == {
        "count": 2,
        "total": 0.9,
        "average": 0.45,
        "p95": 0.0,
        "max": 0.5,
    }
    assert metrics["offline_shared_timings"]["count"] == 2
    assert metrics["offline_shared_timings"]["queue_ms"] == {
        "count": 2,
        "total_ms": 40.0,
        "average_ms": 20.0,
        "p95_ms": 0.0,
        "max_ms": 30.0,
    }
    assert metrics["event_loop_lag"]["max_ms"] == 4.0
    assert metrics["lanes"][0] == {
        "stream_id": 0,
        "processed_frames": 10,
        "dropped_frames": 1,
        "offline_requests": 2,
        "final_state": "finished",
        "online_frame_latency": {"count": 2, "total_ms": 100.0, "average_ms": 50.0, "max_ms": 60.0},
        "offline_rtf": {"count": 1, "total": 0.4, "average": 0.4, "max": 0.4},
        "offline_shared_timings": shared_summary(10.0, 100.0),
        "event_loop_lag": {"count": 3, "total_ms": 6.0, "average_ms": 2.0, "max_ms": 4.0},
        "results_path": "results/lane0.txt",
        "model_outputs_path": "model_outputs/lane0.jsonl",
    }
    assert "lane_metrics" not in metrics["lanes"][0]
    assert metrics["lanes"][1]["stream_id"] == 1
    assert "lane_metrics" not in metrics["lanes"][1]
    assert (run_dir / "results" / "lane0.txt").read_text(encoding="utf-8") == "lane-owned\n"
    assert (run_dir / "model_outputs" / "lane0.jsonl").read_text(encoding="utf-8") == '{"lane":0}\n'
