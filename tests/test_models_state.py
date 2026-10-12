import numpy as np
import pytest

from multi_stream_asr.models import (
    ASREvent,
    AudioFrame,
    EventStatus,
    LaneFinalState,
    LaneMetrics,
    OnlineRecognition,
    ResultType,
)
from multi_stream_asr.pipeline.state import PipelineState, assert_transition


def test_audio_frame_finish_has_empty_pcm():
    frame = AudioFrame.finish(stream_id=1, frame_index=42, timestamp_ms=1000)

    assert frame.stream_id == 1
    assert frame.frame_index == 42
    assert frame.is_finish is True
    assert frame.pcm.dtype == np.float32
    assert frame.pcm.size == 0


def test_online_recognition_defaults_token_time_to_none():
    result = OnlineRecognition(
        stream_id=0,
        sentence_id=0,
        revision=0,
        text="你好",
        timestamp_ms=100,
    )
    assert result.last_token_stream_s is None


def test_asr_event_final_flag_matches_status():
    pending = ASREvent.asr_result(
        stream_id=0,
        sentence_id=3,
        revision=2,
        status=EventStatus.PENDING,
        text="hello",
        timestamp_ms=123,
    )
    final = ASREvent.asr_result(
        stream_id=0,
        sentence_id=3,
        revision=3,
        status=EventStatus.FINAL,
        text="hello world",
        timestamp_ms=124,
    )

    assert pending.result_type == ResultType.ONLINE
    assert pending.is_final is False
    assert final.result_type == ResultType.OFFLINE
    assert final.is_final is True


def test_lane_metrics_defaults():
    metrics = LaneMetrics(stream_id=2)

    assert metrics.processed_frames == 0
    assert metrics.dropped_frames == 0
    assert metrics.offline_requests == 0
    assert metrics.final_state == LaneFinalState.STOPPED


def test_state_allows_normal_lifecycle():
    assert_transition(PipelineState.UNLOADED, PipelineState.LOADING)
    assert_transition(PipelineState.LOADING, PipelineState.LOADED)
    assert_transition(PipelineState.LOADED, PipelineState.STARTING)
    assert_transition(PipelineState.STARTING, PipelineState.RUNNING)
    assert_transition(PipelineState.RUNNING, PipelineState.STOPPING)
    assert_transition(PipelineState.STOPPING, PipelineState.LOADED)
    assert_transition(PipelineState.LOADED, PipelineState.STARTING)
    assert_transition(PipelineState.LOADED, PipelineState.UNLOADING)
    assert_transition(PipelineState.UNLOADING, PipelineState.UNLOADED)


def test_state_rejects_illegal_transition():
    with pytest.raises(ValueError, match="illegal pipeline state transition"):
        assert_transition(PipelineState.UNLOADED, PipelineState.RUNNING)
