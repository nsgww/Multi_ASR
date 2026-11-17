from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from multi_stream_asr.asr.summary.config import SummaryConfig, split_transcript_chunks

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
    chunk_count: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "text": self.text or self.summary,
            "provider": self.provider,
            "reason": self.reason,
            "error": self.error,
            "regenerated": self.regenerated,
            "chunk_count": self.chunk_count,
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
            local = self._local_summary(text)
            return SummaryResult(
                summary=local,
                provider="local",
                text=local,
                regenerated=regenerate,
            )

        resolved_temperature = self._resolve_temperature(
            temperature=temperature,
            random_temperature=random_temperature,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            regenerate=regenerate,
        )

        chunks = split_transcript_chunks(text, self._config.max_input_chars)
        if not chunks:
            return SummaryResult(
                summary="",
                provider="skipped",
                reason="empty transcript after normalize",
            )

        if len(chunks) == 1:
            content, error = await self._summarize_chunk(
                chunks[0],
                style=style,
                temperature=resolved_temperature,
                partial=False,
            )
            if error:
                return self._on_error(text, error, regenerate, chunk_count=1)
            return SummaryResult(
                summary=content,
                provider=self._config.provider,
                text=content,
                regenerated=regenerate,
                chunk_count=1,
            )

        _logger.info(
            "summary map-reduce: %s chars -> %s chunks (max_input_chars=%s chunk_max_tokens=%s max_tokens=%s)",
            len(text),
            len(chunks),
            self._config.max_input_chars,
            self._config.chunk_max_tokens,
            self._config.max_tokens,
        )
        partials: list[str] = []
        errors: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            t0 = time.perf_counter()
            content, error = await self._summarize_chunk(
                chunk,
                style=style,
                temperature=resolved_temperature,
                partial=True,
            )
            _logger.info(
                "summary chunk %s/%s done in %.1fs (chars=%s error=%s)",
                idx,
                len(chunks),
                time.perf_counter() - t0,
                len(chunk),
                bool(error),
            )
            if error:
                _logger.warning("summary chunk %s/%s failed: %s", idx, len(chunks), error)
                errors.append(f"chunk{idx}: {error}")
                # 单块失败时用抽取式兜底，尽量不丢该段信息
                partials.append(self._local_summary(chunk))
                continue
            partials.append(content)

        if not any(p.strip() for p in partials):
            joined_error = "; ".join(errors) if errors else "all chunks empty"
            return self._on_error(
                text, joined_error, regenerate, chunk_count=len(chunks)
            )

        t_merge = time.perf_counter()
        merged, merge_error = await self._merge_partials(
            partials, style=style, temperature=resolved_temperature
        )
        _logger.info(
            "summary merge done in %.1fs (partials=%s error=%s)",
            time.perf_counter() - t_merge,
            len(partials),
            bool(merge_error),
        )
        if merge_error or not merged.strip():
            # 合并失败则拼接分段摘要，仍比截断全文有用
            fallback = self._join_partials(partials)
            return SummaryResult(
                summary=fallback,
                provider=self._config.provider,
                text=fallback,
                error=merge_error or "merge returned empty",
                reason="chunk_join_fallback",
                regenerated=regenerate,
                chunk_count=len(chunks),
            )

        return SummaryResult(
            summary=merged,
            provider=self._config.provider,
            text=merged,
            error="; ".join(errors) if errors else "",
            reason="map_reduce" if not errors else "map_reduce_partial_fallback",
            regenerated=regenerate,
            chunk_count=len(chunks),
        )

    def _chunk_token_budget(self) -> int:
        configured = int(self._config.chunk_max_tokens or 0)
        if configured > 0:
            return configured
        return self._config.max_tokens

    async def _summarize_chunk(
        self,
        chunk: str,
        *,
        style: str | None,
        temperature: float,
        partial: bool,
    ) -> tuple[str, str]:
        prompt = self._config.build_prompt(chunk, style=style, partial=partial)
        max_tokens = (
            self._chunk_token_budget() if partial else self._config.max_tokens
        )
        return await self._post_chat(
            prompt, temperature=temperature, max_tokens=max_tokens
        )

    async def _merge_partials(
        self,
        partials: list[str],
        *,
        style: str | None,
        temperature: float,
    ) -> tuple[str, str]:
        # 分段摘要本身可能仍超长：再按块合并
        working = [p.strip() for p in partials if p and p.strip()]
        if not working:
            return "", "no partial summaries"
        if len(working) == 1:
            return working[0], ""

        safety = 0
        while len(working) > 1 and safety < 8:
            safety += 1
            packed = self._config.build_merge_prompt(working, style=style)
            # build_merge_prompt 内部会截断；若截断则改为分层两两/分组合并
            numbered = "\n\n".join(
                f"【分段摘要 {idx}】\n{part}"
                for idx, part in enumerate(working, start=1)
            )
            if len(numbered) <= self._config.max_input_chars:
                return await self._post_chat(
                    packed,
                    temperature=temperature,
                    max_tokens=self._config.max_tokens,
                )

            group_size = max(2, (len(working) + 1) // 2)
            next_round: list[str] = []
            errors: list[str] = []
            for start in range(0, len(working), group_size):
                group = working[start : start + group_size]
                if len(group) == 1:
                    next_round.append(group[0])
                    continue
                prompt = self._config.build_merge_prompt(group, style=style)
                content, error = await self._post_chat(
                    prompt,
                    temperature=temperature,
                    max_tokens=self._config.max_tokens,
                )
                if error or not content.strip():
                    errors.append(error or "empty merge group")
                    next_round.append(self._join_partials(group))
                else:
                    next_round.append(content)
            if errors:
                _logger.warning("summary hierarchical merge warnings: %s", "; ".join(errors))
            if next_round == working:
                break
            working = next_round

        if len(working) == 1:
            return working[0], ""
        # 最后再试一次全量合并；仍失败则拼接
        prompt = self._config.build_merge_prompt(working, style=style)
        content, error = await self._post_chat(
            prompt,
            temperature=temperature,
            max_tokens=self._config.max_tokens,
        )
        if error or not content.strip():
            return self._join_partials(working), error or "final merge empty"
        return content, ""

    async def _post_chat(
        self,
        user_content: str,
        *,
        temperature: float,
        with_system_prompt: bool = True,
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        """单次 chat 请求，返回 (content, error)。"""
        messages: list[dict[str, Any]] = []
        if with_system_prompt and self._config.system_prompt:
            messages.append(
                {"role": "system", "content": self._config.system_prompt}
            )
        messages.append({"role": "user", "content": user_content})
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": (
                self._config.max_tokens if max_tokens is None else max_tokens
            ),
            # gemma 等易退化模型：惩罚重复，正常模型不配置则不发送
            **(
                {"repeat_penalty": self._config.repeat_penalty}
                if self._config.repeat_penalty is not None
                else {}
            ),
            **(
                {"presence_penalty": self._config.presence_penalty}
                if self._config.presence_penalty is not None
                else {}
            ),
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
        except Exception as exc:
            return "", f"{type(exc).__name__}: {exc}"

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
            return "", "LLM returned empty summary" + (
                f" (finish_reason={finish_reason})" if finish_reason else ""
            )
        return content, ""

    def _on_error(
        self,
        text: str,
        error: str,
        regenerate: bool,
        *,
        chunk_count: int = 1,
    ) -> SummaryResult:
        if self._config.fallback_on_error:
            fallback = self._local_summary(text)
            return SummaryResult(
                summary=fallback,
                provider=self._config.provider,
                text=fallback,
                error=error,
                reason="fallback_local",
                regenerated=regenerate,
                chunk_count=chunk_count,
            )
        return SummaryResult(
            summary="",
            provider=self._config.provider,
            error=error,
            regenerated=regenerate,
            chunk_count=chunk_count,
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
        # 仅显式 random_temperature 时抖动；单纯 regenerate 保持配置温度更稳
        if random_temperature:
            lo = min(min_temperature, max_temperature)
            hi = max(min_temperature, max_temperature)
            return random.uniform(lo, hi)
        _ = regenerate  # 保留参数兼容旧调用方
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
    def _join_partials(partials: list[str]) -> str:
        blocks = [p.strip() for p in partials if p and p.strip()]
        if not blocks:
            return "暂无可总结内容。"
        if len(blocks) == 1:
            return blocks[0]
        return "\n\n".join(
            f"### 分段 {idx}\n{block}" for idx, block in enumerate(blocks, start=1)
        )

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
        # 长文本取前/中/后，避免只留开头
        if len(sentences) <= 8:
            picked = sentences[:8]
        else:
            head = sentences[:3]
            mid_start = max((len(sentences) - 2) // 2, 3)
            mid = sentences[mid_start : mid_start + 2]
            tail = sentences[-3:]
            picked = []
            for item in head + mid + tail:
                if item not in picked:
                    picked.append(item)
        return "\n".join(f"- {item}" for item in picked)

    async def close(self) -> None:
        await self._client.aclose()
