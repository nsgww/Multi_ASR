from __future__ import annotations

from pathlib import Path

from multi_stream_asr.asr.kws.keywords_tool import find_invalid_keyword_tokens


def _tokens_file(tmp_path: Path) -> Path:
    path = tmp_path / "tokens.txt"
    path.write_text("n 0\nǐ 0\nh 0\nǎo 0\nx 0\niǎo 0\nl 0\nián 0\n", encoding="utf-8")
    return path


def test_valid_lines_pass(tmp_path: Path) -> None:
    lines = ["n ǐ h ǎo x iǎo l ián @你好小联"]
    assert find_invalid_keyword_tokens(lines, _tokens_file(tmp_path)) == {}


def test_unknown_tokens_reported(tmp_path: Path) -> None:
    lines = ["n i3 h ao3 @你好", "# 注释行", ""]
    invalid = find_invalid_keyword_tokens(lines, _tokens_file(tmp_path))
    assert invalid == {"n i3 h ao3 @你好": ["i3", "ao3"]}


def test_missing_tokens_file_skips_validation(tmp_path: Path) -> None:
    assert find_invalid_keyword_tokens(["a b @x"], tmp_path / "nope.txt") == {}
