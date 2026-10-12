from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any

import httpx

from multi_stream_asr.asr.summary.config import SummaryConfig

_logger = logging.getLogger(__name__)

DEFAULT_MIN_TEMPERATURE = 0.25
DEFAULT_MAX_TEMPERATURE = 0.5

_THINK_RE = re.compile(
    r"<think>.*?</think>|<think>.*?</think>",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    provider: str
    text: str = ""
    reason: str = ""
    error: str = ""
    regenerated: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "text": self.text or self.summary,
            "provider": self.provider,
            "reason": self.reason,
            "error": self.error,
            "regenerated": self.regenerated,
        }


class SummaryClient:
    """HTTP client for the summary service (OpenAI-compatible chat completions)."""

    def __init__(self, config: SummaryConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url_or_default,
            timeout=config.timeout,
        )

    async def generate(
        self,
        text: str,
        *,
        style: str | None = None,
        temperature: float | None = None,
        random_temperature: bool = False,
        min_temperature: float = DEFAULT_MIN_TEMPERATURE,
        max_temperature: float = DEFAULT_MAX_TEMPERATURE,
        regenerate: bool = False,
    ) -> SummaryResult:
        if not text or len(text.strip()) < self._config.min_text_chars:
            return SummaryResult(
                summary="",
                provider="skipped",
                reason=f"transcript too short (< {self._config.min_text_chars} chars)",
            )

        if self._config.provider in ("skipped", "disabled") or not self._config.enabled:
            return SummaryResult(
                summary="",
                provider="disabled",
                reason="summary is disabled",
            )

        if self._config.provider == "local":
            return SummaryResult(
                summary=self._local_summary(text),
                provider="local",
                text=self._local_summary(text),
                regenerated=regenerate,
            )

        resolved_temperature = self._resolve_temperature(
            temperature=temperature,
            random_temperature=random_temperature,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            regenerate=regenerate,
        )

        prompt = self._config.build_prompt(text, style=style)
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": resolved_temperature,
            "max_tokens": self._config.max_tokens,
            # Qwen3 / llama.cpp: without this, thinking may consume all tokens
            # and leave message.content empty.
            "chat_template_kwargs": {
                "enable_thinking": bool(self._config.enable_thinking),
            },
        }

        headers: dict[str, str] = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            raw = response.json()
        except httpx.HTTPError as exc:
            return self._on_error(text, f"{type(exc).__name__}: {exc}", regenerate)
        except Exception as exc:
            return self._on_error(text, f"{type(exc).__name__}: {exc}", regenerate)

        content = self._extract_content(raw)
        if not content:
            finish_reason = ""
            choices = raw.get("choices") if isinstance(raw, dict) else None
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                finish_reason = str(choices[0].get("finish_reason") or "")
            _logger.warning(
                "summary LLM returned empty content (finish_reason=%s keys=%s)",
                finish_reason,
                list(raw.keys()) if isinstance(raw, dict) else type(raw),
            )
            return self._on_error(
                text,
                "LLM returned empty summary"
                + (f" (finish_reason={finish_reason})" if finish_reason else ""),
                regenerate,
            )

        return SummaryResult(
            summary=content,
            provider=self._config.provider,
            text=content,
            regenerated=regenerate,
        )

    def _on_error(self, text: str, error: str, regenerate: bool) -> SummaryResult:
        if self._config.fallback_on_error:
            fallback = self._local_summary(text)
            return SummaryResult(
                summary=fallback,
                provider=self._config.provider,
                text=fallback,
                error=error,
                reason="fallback_local",
                regenerated=regenerate,
            )
        return SummaryResult(
            summary="",
            provider=self._config.provider,
            error=error,
            regenerated=regenerate,
        )

    def _resolve_temperature(
        self,
        *,
        temperature: float | None,
        random_temperature: bool,
        min_temperature: float,
        max_temperature: float,
        regenerate: bool,
    ) -> float:
        if temperature is not None:
            return temperature
        if regenerate or random_temperature:
            return random.uniform(min_temperature, max_temperature)
        return self._config.temperature

    @classmethod
    def _extract_content(cls, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first, dict) else {}
            if not isinstance(message, dict):
                message = {}
            content = cls._normalize_content(message.get("content"))
            if content:
                return content
            # Some builds put visible text only in reasoning when thinking is on.
            reasoning = cls._normalize_content(
                message.get("reasoning_content") or message.get("reasoning")
            )
            if reasoning:
                return reasoning
            # Rare: text completion style
            text = cls._normalize_content(first.get("text") if isinstance(first, dict) else "")
            if text:
                return text
        return cls._normalize_content(payload.get("content"))

    @classmethod
    def _normalize_content(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
            content = "".join(parts)
        if not isinstance(content, str):
            content = str(content)
        content = _THINK_RE.sub("", content).strip()
        return content

    @staticmethod
    def _local_summary(text: str) -> str:
        """Simple extractive fallback when LLM returns empty / fails."""
        normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
        if not normalized:
            return "暂无可总结内容。"
        parts = re.split(r"(?<=[。！？!?；;])\s*|\n+", normalized)
        sentences = [p.strip() for p in parts if p and p.strip()]
        if not sentences:
            clip = normalized[:200]
            return f"- {clip}"
        bullets = sentences[:5]
        return "\n".join(f"- {item}" for item in bullets)

    async def close(self) -> None:
        await self._client.aclose()
