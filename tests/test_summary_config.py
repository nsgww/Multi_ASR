from __future__ import annotations

from pathlib import Path

import pytest

from multi_stream_asr.asr.summary.config import SummaryConfig, load_summary_config


def test_default_summary_config() -> None:
    cfg = SummaryConfig()
    assert cfg.provider == "llama"
    assert cfg.enabled is True
    assert "语音内容摘要助手" in cfg.build_prompt("test")


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
""",
        encoding="utf-8",
    )
    cfg = load_summary_config(path)
    assert cfg.provider == "llama"
    assert cfg.model == "Qwen3.5-2B-Q4_0"
    assert cfg.port == 8070
    assert cfg.temperature == 0.3


def test_summary_config_rejects_invalid_provider() -> None:
    with pytest.raises(ValueError):
        SummaryConfig(provider="unknown")


def test_summary_prompt_style_injection() -> None:
    cfg = SummaryConfig()
    prompt = cfg.build_prompt("hello", style="concise")
    assert "简洁" in prompt
