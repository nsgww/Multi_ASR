from __future__ import annotations

import numpy as np
import pytest

from multi_stream_asr.config import SpeakerConfig
from multi_stream_asr.speaker.registry import VoiceprintRegistry
from multi_stream_asr.speaker.tracker import SpeakerTracker

SR = 16000


def vec(*values: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).astype(np.float32)


def pcm(seconds: float = 1.5) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


class StubEmbedder:
    """按顺序吐出预设向量；可用 None 模拟提特征失败。"""

    def __init__(self, vectors, *, available: bool = True) -> None:
        self._vectors = list(vectors)
        self.available = available
        self.calls: list[int] = []

    def embed(self, samples):  # noqa: ANN001
        self.calls.append(int(np.asarray(samples).size))
        return self._vectors.pop(0) if self._vectors else None


class RaisingEmbedder:
    available = True

    def embed(self, samples):  # noqa: ANN001
        raise RuntimeError("boom")


def make_tracker(embedder, **overrides) -> SpeakerTracker:
    config = SpeakerConfig(enabled=True, **overrides)
    return SpeakerTracker(embedder=embedder, config=config, stream_id=0)


def test_tracker_without_embedder_is_disabled() -> None:
    tracker = SpeakerTracker(embedder=None, config=SpeakerConfig(enabled=True))
    assert tracker.enabled is False
    assert tracker.assign(pcm()) is None


def test_tracker_disabled_when_model_unavailable() -> None:
    tracker = make_tracker(StubEmbedder([vec(1, 0, 0)], available=False))
    assert tracker.enabled is False


def test_too_short_segment_returns_none_at_first() -> None:
    tracker = make_tracker(StubEmbedder([vec(1, 0, 0)]), min_duration_seconds=1.2)
    assert tracker.assign(pcm(0.3)) is None


def test_too_short_segment_inherits_previous_speaker() -> None:
    embedder = StubEmbedder([vec(1, 0, 0)])
    tracker = make_tracker(embedder, min_duration_seconds=1.2)

    first = tracker.assign(pcm(2.0))
    assert first is not None and first.label == "S1"

    inherited = tracker.assign(pcm(0.3))
    assert inherited is not None
    assert inherited.label == "S1"
    assert inherited.is_provisional is True
    # 短句不应消耗 embedding
    assert len(embedder.calls) == 1


def test_two_similar_sentences_share_one_speaker() -> None:
    tracker = make_tracker(StubEmbedder([vec(1, 0, 0), vec(1, 0.05, 0)]))

    first = tracker.assign(pcm())
    second = tracker.assign(pcm())

    assert first is not None and second is not None
    assert first.label == second.label == "S1"
    assert tracker.speaker_count == 1


def test_different_voices_get_different_labels() -> None:
    tracker = make_tracker(StubEmbedder([vec(1, 0, 0), vec(0, 1, 0)]))

    first = tracker.assign(pcm())
    second = tracker.assign(pcm())

    assert first is not None and second is not None
    assert first.label == "S1"
    assert second.label == "S2"
    assert tracker.speaker_count == 2


def test_new_speaker_is_marked_provisional() -> None:
    tracker = make_tracker(StubEmbedder([vec(1, 0, 0)]))
    info = tracker.assign(pcm())
    assert info is not None
    assert info.is_provisional is True
    assert info.confidence == 1.0


def test_embed_failure_inherits_and_does_not_raise() -> None:
    tracker = make_tracker(StubEmbedder([vec(1, 0, 0), None]))

    first = tracker.assign(pcm())
    assert first is not None and first.label == "S1"

    second = tracker.assign(pcm())
    assert second is not None
    assert second.label == "S1"
    assert second.is_provisional is True


def test_embed_exception_inherits_and_does_not_raise() -> None:
    tracker = make_tracker(RaisingEmbedder())
    # 第一次没有可继承的说话人
    assert tracker.assign(pcm()) is None
    assert tracker.summary()["failed"] == 1


def test_registry_hit_sets_name_and_clears_provisional() -> None:
    registry = VoiceprintRegistry()
    registry.enroll("张三", vec(1, 0, 0))
    tracker = SpeakerTracker(
        embedder=StubEmbedder([vec(1, 0.01, 0)]),
        config=SpeakerConfig(enabled=True),
        registry=registry,
    )

    info = tracker.assign(pcm())
    assert info is not None
    assert info.name == "张三"
    assert info.is_provisional is False
    assert info.display_name == "张三"


def test_registry_miss_keeps_label_as_display_name() -> None:
    registry = VoiceprintRegistry()
    registry.enroll("张三", vec(0, 1, 0))
    tracker = SpeakerTracker(
        embedder=StubEmbedder([vec(1, 0, 0)]),
        config=SpeakerConfig(enabled=True),
        registry=registry,
    )

    info = tracker.assign(pcm())
    assert info is not None
    assert info.name is None
    assert info.display_name == "S1"


def test_resegment_merges_over_split_cluster() -> None:
    # 在线阶段因为阈值高把同一个人的第三句单独开簇，二遍重聚类应合回 S1
    embedder = StubEmbedder(
        [vec(1, 0, 0), vec(0, 1, 0), vec(1, -0.9, 0)]
    )
    tracker = make_tracker(
        embedder,
        match_threshold=0.85,
        new_speaker_threshold=0.80,
        merge_threshold=0.70,
        sticky_bonus=0.0,
        min_duration_seconds=0.001,
    )

    labels = [tracker.assign(pcm()).label for _ in range(3)]
    assert labels == ["S1", "S2", "S3"]

    mapping = tracker.resegment()
    assert mapping == {"S3": "S1"}


def test_resegment_returns_empty_when_disabled() -> None:
    tracker = make_tracker(
        StubEmbedder([vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)]),
        enable_second_pass=False,
        min_duration_seconds=0.001,
    )
    for _ in range(3):
        tracker.assign(pcm())
    assert tracker.resegment() == {}


def test_resegment_needs_at_least_three_sentences() -> None:
    tracker = make_tracker(
        StubEmbedder([vec(1, 0, 0), vec(0, 1, 0)]), min_duration_seconds=0.001
    )
    tracker.assign(pcm())
    tracker.assign(pcm())
    assert tracker.resegment() == {}


def test_summary_counts() -> None:
    tracker = make_tracker(StubEmbedder([vec(1, 0, 0), vec(0, 1, 0)]))
    tracker.assign(pcm())
    tracker.assign(pcm())
    summary = tracker.summary()

    assert summary["stream_id"] == 0
    assert summary["sentences"] == 2
    assert summary["speakers"] == 2
    assert summary["enabled"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"match_threshold": 0.3, "new_speaker_threshold": 0.6},
        {"identify_margin": 1.5},
        {"min_duration_seconds": 0.0},
        {"max_speakers": 0},
    ],
)
def test_config_validation(overrides: dict) -> None:
    with pytest.raises(ValueError):
        SpeakerConfig(enabled=True, **overrides)
