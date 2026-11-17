from __future__ import annotations

from typing import Any

import pytest

from multi_stream_asr.asr.summary.config import SummaryConfig
from multi_stream_asr.asr.summary.summary_client import SummaryClient


@pytest.mark.asyncio
async def test_generate_map_reduce_merges_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SummaryConfig(
        provider="openai",
        base_url="http://example.invalid",
        model="test-model",
        max_input_chars=40,
        max_tokens=128,
        chunk_max_tokens=64,
        temperature=0.1,
        fallback_on_error=False,
    )
    client = SummaryClient(cfg)

    calls: list[str] = []

    async def fake_post(
        user_content: str,
        *,
        temperature: float,
        with_system_prompt: bool = True,
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        _ = temperature, with_system_prompt, max_tokens
        calls.append(user_content)
        if "分段摘要" in user_content:
            return (
                "- **主题**：测试\n- **要点**：\n  1. A\n  2. B\n  3. C",
                "",
            )
        # 段摘要：从转写块里抓一段标识
        marker = "CHUNK"
        for token in ("AAA", "BBB", "CCC"):
            if token in user_content:
                marker = token
                break
        return f"- 段摘要 {marker}", ""

    monkeypatch.setattr(client, "_post_chat", fake_post)

    # 三块：每行足够长，强制按行分块
    text = "\n".join(
        [
            "AAA " + ("甲" * 36),
            "BBB " + ("乙" * 36),
            "CCC " + ("丙" * 36),
        ]
    )
    result = await client.generate(text)
    await client.close()

    assert result.chunk_count >= 2
    assert result.summary
    assert "主题" in result.summary or "要点" in result.summary
    assert any("分段摘要" in c for c in calls)
    assert len(calls) >= 3


@pytest.mark.asyncio
async def test_regenerate_keeps_configured_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SummaryConfig(
        provider="openai",
        base_url="http://example.invalid",
        model="test-model",
        temperature=0.2,
        max_input_chars=2000,
    )
    client = SummaryClient(cfg)
    seen: dict[str, Any] = {}

    async def fake_post(
        user_content: str,
        *,
        temperature: float,
        with_system_prompt: bool = True,
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        _ = user_content, with_system_prompt, max_tokens
        seen["temperature"] = temperature
        return "- ok", ""

    monkeypatch.setattr(client, "_post_chat", fake_post)
    result = await client.generate(
        "This is a long enough transcript for regenerate temperature test.",
        regenerate=True,
    )
    await client.close()
    assert result.summary == "- ok"
    assert seen["temperature"] == 0.2
