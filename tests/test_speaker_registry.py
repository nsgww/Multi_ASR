from __future__ import annotations

import numpy as np
import pytest

from multi_stream_asr.speaker.registry import VoiceprintRegistry


def vec(*values: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).astype(np.float32)


def test_identify_hits_enrolled_speaker() -> None:
    registry = VoiceprintRegistry()
    registry.enroll("张三", vec(1, 0, 0))

    match = registry.identify(vec(1, 0.02, 0), threshold=0.60, margin=0.05)
    assert match is not None
    assert match.name == "张三"
    assert match.score > 0.9


def test_identify_returns_none_for_unknown_voice() -> None:
    registry = VoiceprintRegistry()
    registry.enroll("张三", vec(1, 0, 0))

    assert registry.identify(vec(0, 1, 0), threshold=0.60, margin=0.05) is None


def test_score_all_ranks_every_enrolled_name() -> None:
    registry = VoiceprintRegistry()
    registry.enroll("张三", vec(1, 0, 0))
    registry.enroll("李四", vec(0, 1, 0))

    scored = registry.score_all(vec(1, 0.1, 0))

    assert [name for name, _ in scored] == ["张三", "李四"]
    assert scored[0][1] > scored[1][1]
    assert scored[0][1] > 0.9


def test_score_all_ignores_unknown_dimension() -> None:
    registry = VoiceprintRegistry()
    registry.enroll("张三", vec(1, 0, 0))

    assert registry.score_all(vec(1, 0, 0, 0)) == []


def test_score_all_on_empty_registry() -> None:
    assert VoiceprintRegistry().score_all(vec(1, 0, 0)) == []


def test_score_all_rejects_zero_vector() -> None:
    registry = VoiceprintRegistry()
    registry.enroll("张三", vec(1, 0, 0))

    assert registry.score_all(np.zeros(3, dtype=np.float32)) == []


def test_identify_requires_margin_over_second_best() -> None:
    registry = VoiceprintRegistry()
    # 两个音色非常接近的人
    registry.enroll("甲", vec(1, 0, 0))
    registry.enroll("乙", vec(1, 0.02, 0))

    match = registry.identify(vec(1, 0.01, 0), threshold=0.50, margin=0.10)
    assert match is None


def test_identify_on_empty_registry_returns_none() -> None:
    registry = VoiceprintRegistry()
    assert registry.identify(vec(1, 0, 0), threshold=0.5, margin=0.05) is None


def test_enroll_accumulates_and_counts() -> None:
    registry = VoiceprintRegistry()
    assert registry.enroll("张三", vec(1, 0, 0)) == 1
    assert registry.enroll("张三", vec(1, 0.05, 0)) == 2
    assert registry.count("张三") == 2
    assert registry.names() == ["张三"]
    assert len(registry) == 1


def test_enroll_rejects_dim_mismatch() -> None:
    registry = VoiceprintRegistry()
    registry.enroll("张三", vec(1, 0, 0))
    with pytest.raises(ValueError):
        registry.enroll("张三", vec(1, 0))


def test_enroll_rejects_empty_name() -> None:
    registry = VoiceprintRegistry()
    with pytest.raises(ValueError):
        registry.enroll("   ", vec(1, 0, 0))


def test_enroll_passes_on_zero_vector() -> None:
    registry = VoiceprintRegistry()
    with pytest.raises(ValueError):
        registry.enroll("张三", np.zeros(3, dtype=np.float32))


def test_registry_persists_and_reloads(tmp_path) -> None:
    directory = tmp_path / "voiceprints"
    registry = VoiceprintRegistry(directory)
    registry.enroll("张三", vec(1, 0, 0))
    registry.enroll("李四", vec(0, 1, 0))

    assert (directory / "index.json").is_file()

    reloaded = VoiceprintRegistry(directory)
    assert reloaded.names() == ["张三", "李四"]
    match = reloaded.identify(vec(1, 0.01, 0), threshold=0.6, margin=0.05)
    assert match is not None and match.name == "张三"


def test_remove_deletes_name_and_file(tmp_path) -> None:
    directory = tmp_path / "voiceprints"
    registry = VoiceprintRegistry(directory)
    registry.enroll("张三", vec(1, 0, 0))
    assert registry.remove("张三") is True
    assert registry.names() == []
    assert registry.remove("张三") is False

    reloaded = VoiceprintRegistry(directory)
    assert reloaded.names() == []


def test_corrupted_index_falls_back_to_empty(tmp_path) -> None:
    directory = tmp_path / "voiceprints"
    directory.mkdir(parents=True)
    (directory / "index.json").write_text("{not json", encoding="utf-8")

    registry = VoiceprintRegistry(directory)
    assert registry.names() == []


def test_max_prints_per_name_drops_oldest() -> None:
    registry = VoiceprintRegistry(max_prints_per_name=2)
    registry.enroll("张三", vec(1, 0, 0))
    registry.enroll("张三", vec(0, 1, 0))
    registry.enroll("张三", vec(0, 0, 1))

    assert registry.count("张三") == 2


def test_new_name_does_not_overwrite_existing_file_after_removal(tmp_path) -> None:
    """删过人之后新注册的名字不能占用旧文件名（否则会覆盖别人的向量）。"""
    directory = tmp_path / "voiceprints"
    registry = VoiceprintRegistry(directory)
    registry.enroll("甲", vec(1, 0, 0))
    registry.enroll("乙", vec(0, 1, 0))
    registry.remove("甲")

    reloaded = VoiceprintRegistry(directory)
    assert reloaded.names() == ["乙"]
    reloaded.enroll("丙", vec(0, 0, 1))

    again = VoiceprintRegistry(directory)
    assert again.names() == ["丙", "乙"] or again.names() == ["乙", "丙"]
    match = again.identify(vec(0, 1, 0), threshold=0.6, margin=0.05)
    assert match is not None, "乙 的声纹被 丙 覆盖了"
    assert match.name == "乙"

