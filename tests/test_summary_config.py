from __future__ import annotations

from pathlib import Path

import pytest

from multi_stream_asr.asr.summary.config import (
    SummaryConfig,
    load_summary_config,
    split_transcript_chunks,
)


def test_default_summary_config() -> None:
    cfg = SummaryConfig()
    assert cfg.provider == "llama"
    assert cfg.enabled is True
    assert "语音内容摘要助手" in cfg.build_prompt("test")
    assert "完整保留全部条目" in cfg.build_prompt("test")


def test_load_summary_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "summary.yaml"
    path.write_text(
        """
summary:
  provider: llama
  enabled: true
  model: "Qwen3.5-2B-Q4_0"
  port: 8070
  temperature: 0.3
  merge_prompt_template: |
    MERGE:{text}
""",
        encoding="utf-8",
    )
    cfg = load_summary_config(path)
    assert cfg.provider == "llama"
    assert cfg.model == "Qwen3.5-2B-Q4_0"
    assert cfg.port == 8070
    assert cfg.temperature == 0.3
    assert cfg.build_merge_prompt(["a", "b"]).startswith("MERGE:")


def test_summary_config_rejects_invalid_provider() -> None:
    with pytest.raises(ValueError):
        SummaryConfig(provider="unknown")


def test_summary_prompt_style_injection() -> None:
    cfg = SummaryConfig()
    prompt = cfg.build_prompt("hello", style="concise")
    assert "简洁" in prompt
    detailed = cfg.build_prompt("hello", style="detailed")
    assert "完整保留" in detailed
    partial = cfg.build_prompt("hello", partial=True)
    assert "本段" in partial
    assert "250 字" in partial


def test_split_transcript_chunks_preserves_full_text() -> None:
    lines = [f"line-{i}-" + ("x" * 80) for i in range(40)]
    text = "\n".join(lines)
    chunks = split_transcript_chunks(text, max_chars=500)
    assert len(chunks) >= 2
    assert "".join(c.replace("\n", "") for c in chunks) == text.replace("\n", "")
    assert all(len(c) <= 500 for c in chunks)


def test_split_transcript_chunks_short_text() -> None:
    assert split_transcript_chunks("hello world", 100) == ["hello world"]
