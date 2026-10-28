from __future__ import annotations

import numpy as np
import pytest

from multi_stream_asr.speaker.cluster import (
    OnlineSpeakerClusterer,
    agglomerative_cluster,
)


def unit(*values: float) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def test_first_embedding_creates_first_cluster() -> None:
    clusterer = OnlineSpeakerClusterer()
    assignment = clusterer.assign(unit(1, 0, 0))

    assert assignment.index == 0
    assert assignment.created is True
    assert assignment.similarity == 1.0
    assert clusterer.cluster_count == 1


def test_similar_embedding_joins_existing_cluster() -> None:
    clusterer = OnlineSpeakerClusterer(match_threshold=0.55)
    clusterer.assign(unit(1, 0, 0))
    assignment = clusterer.assign(unit(1, 0.05, 0))

    assert assignment.index == 0
    assert assignment.created is False
    assert assignment.similarity > 0.9
    assert clusterer.cluster_count == 1


def test_dissimilar_embedding_creates_new_cluster() -> None:
    clusterer = OnlineSpeakerClusterer(new_threshold=0.40)
    clusterer.assign(unit(1, 0, 0))
    assignment = clusterer.assign(unit(0, 1, 0))

    assert assignment.index == 1
    assert assignment.created is True
    assert clusterer.cluster_count == 2


def test_grey_zone_goes_to_best_cluster_without_creating() -> None:
    # 相似度落在 new_threshold 与 match_threshold 之间：不新开簇
    clusterer = OnlineSpeakerClusterer(match_threshold=0.55, new_threshold=0.40)
    clusterer.assign(unit(1, 0, 0))
    # cos ≈ 0.45
    grey = unit(1, 2, 0)
    assignment = clusterer.assign(grey)

    assert assignment.index == 0
    assert assignment.created is False
    assert 0.40 <= assignment.similarity <= 0.55
    assert clusterer.cluster_count == 1


def test_sticky_bonus_prevents_label_flapping() -> None:
    # 两个簇相似度接近时，上一句说话人的加成应选中它
    clusterer = OnlineSpeakerClusterer(
        match_threshold=0.50, new_threshold=0.30, sticky_bonus=0.10
    )
    clusterer.assign(unit(1, 0, 0))  # S1
    clusterer.assign(unit(0, 1, 0))  # S2（当前 last=1）
    ambiguous = unit(1, 1, 0)        # 对两个簇相似度都约 0.707
    assignment = clusterer.assign(ambiguous)

    assert assignment.index == 1


def test_max_speakers_caps_cluster_creation() -> None:
    clusterer = OnlineSpeakerClusterer(max_speakers=2, new_threshold=0.30)
    clusterer.assign(unit(1, 0, 0))
    clusterer.assign(unit(0, 1, 0))
    assignment = clusterer.assign(unit(0, 0, 1))

    assert clusterer.cluster_count == 2
    assert assignment.created is False


def test_inherit_last_before_any_assignment_returns_none() -> None:
    clusterer = OnlineSpeakerClusterer()
    assert clusterer.inherit_last() is None


def test_inherit_last_returns_previous_speaker() -> None:
    clusterer = OnlineSpeakerClusterer()
    clusterer.assign(unit(1, 0, 0))
    clusterer.assign(unit(0, 1, 0))
    inherited = clusterer.inherit_last()

    assert inherited is not None
    assert inherited.index == 1
    assert inherited.created is False


def test_merge_similar_reports_groups() -> None:
    clusterer = OnlineSpeakerClusterer(new_threshold=0.20, merge_threshold=0.70)
    clusterer.assign(unit(1, 0, 0))
    clusterer.assign(unit(1, 0.1, 0))    # 并入 S1
    clusterer.assign(unit(0, 0, 1))      # S2

    mapping = clusterer.merge_similar(threshold=0.70)
    assert mapping[0] == mapping[0]
    assert mapping[0] != mapping[1]


def test_agglomerative_cluster_groups_two_speakers() -> None:
    a1, a2 = unit(1, 0, 0), unit(1, 0.08, 0)
    b1, b2 = unit(0, 1, 0), unit(0, 1, 0.08)
    labels = agglomerative_cluster([a1, a2, b1, b2], threshold=0.70)

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_agglomerative_cluster_keeps_three_speakers_apart() -> None:
    vecs = [unit(1, 0, 0), unit(1, 0.05, 0), unit(0, 1, 0), unit(0, 0, 1)]
    labels = agglomerative_cluster(vecs, threshold=0.70)

    assert len({labels[0], labels[2], labels[3]}) == 3


def test_agglomerative_cluster_labels_start_from_zero_by_first_occurrence() -> None:
    vecs = [unit(0, 1, 0), unit(0, 1, 0.02), unit(1, 0, 0)]
    labels = agglomerative_cluster(vecs, threshold=0.70)

    assert labels == [0, 0, 1]


@pytest.mark.parametrize("count", [0, 1])
def test_agglomerative_cluster_handles_degenerate_input(count: int) -> None:
    vecs = [unit(1, 0, 0) for _ in range(count)]
    labels = agglomerative_cluster(vecs, threshold=0.5)
    assert labels == list(range(count))


def test_threshold_validation_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        OnlineSpeakerClusterer(match_threshold=0.3, new_threshold=0.6)
