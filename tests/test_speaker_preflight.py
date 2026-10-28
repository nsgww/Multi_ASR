from __future__ import annotations

from pathlib import Path

import pytest

from multi_stream_asr.speaker.preflight import (
    clear_preflight_cache,
    verify_speaker_model,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_preflight_cache()
    yield
    clear_preflight_cache()


def test_missing_model_is_reported_without_spawning(tmp_path: Path) -> None:
    result = verify_speaker_model(tmp_path / "nope.onnx")

    assert result.ok is False
    assert "model not found" in result.detail


def test_invalid_onnx_is_reported_as_failure(tmp_path: Path) -> None:
    """坏模型必须返回 ok=False，而不是让调用方进程崩溃。

    sherpa-onnx 在元数据缺失时会 SIGSEGV；预检把它关在子进程里，所以这里能
    安全地拿到非 0 退出码。
    """
    bogus = tmp_path / "bogus.onnx"
    bogus.write_bytes(b"not an onnx file")

    result = verify_speaker_model(bogus)

    assert result.ok is False
    assert "exit=" in result.detail


def test_result_is_cached_per_file(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.onnx"
    bogus.write_bytes(b"not an onnx file")

    first = verify_speaker_model(bogus)
    second = verify_speaker_model(bogus)

    assert first is second, "同一文件应命中缓存，避免反复起子进程"


def test_cache_is_invalidated_when_file_changes(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.onnx"
    bogus.write_bytes(b"v1")
    first = verify_speaker_model(bogus)

    bogus.write_bytes(b"v2-longer-content")
    second = verify_speaker_model(bogus)

    assert first is not second, "文件变了要重新预检"


def test_empty_file_is_reported_as_failure(tmp_path: Path) -> None:
    empty = tmp_path / "empty.onnx"
    empty.write_bytes(b"")

    assert verify_speaker_model(empty).ok is False
