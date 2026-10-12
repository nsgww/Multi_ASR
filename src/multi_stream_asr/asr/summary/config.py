from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SUMMARY_PROMPT_TEMPLATE = """\
/no_think
你是语音内容摘要助手。请根据转写文本生成中文核心内容总结。

目标：
用少量要点让读者快速理解音频主要内容与关键信息。

要求：
1. 只基于原文，不得补充或推测
2. 必须归纳总结，不逐句复述
3. 抓取核心信息：在说什么、发生了什么、讨论了什么
4. 保留关键事实：人名、事项、结论、决定、问题、时间
5. 去除口语、重复、无意义填充
6. 最多3-5条要点（视内容而定）

输出格式：
- 要点1
- 要点2
- 要点3
- 要点4（如有必要）
- 要点5（如有必要）

转写内容：
{text}
"""


@dataclass(frozen=True)
class SummaryConfig:
    enabled: bool = True
    provider: str = "llama"
    offline_enabled: bool = True
    realtime_enabled: bool = True
    auto_start: bool = True
    base_url: str = ""
    api_key: str = ""
    model: str = "Qwen3.5-2B-Q4_0"
    model_path: Path = field(
        default_factory=lambda: Path(
            "/var/lib/llm-sdk/models/llm/Qwen3.5-2B-Q4_0.gguf"
        )
    )
    llama_server_bin: str = "llama-server"
    host: str = "127.0.0.1"
    port: int = 8070
    ctx_size: int = 8102
    num_threads: int = 8
    n_gpu_layers: int = 0
    temperature: float = 0.2
    timeout: int = 600
    max_input_chars: int = 2000
    max_tokens: int = 1024
    min_text_chars: int = 20
    realtime_min_chars: int = 80
    fallback_on_error: bool = True
    enable_thinking: bool = False
    transcript_dir: str = "outputs/transcripts"
    language: str = "zh"
    prompt: str = ""
    prompt_template: str = DEFAULT_SUMMARY_PROMPT_TEMPLATE

    def __post_init__(self) -> None:
        if self.provider not in {"local", "llama", "openai", "ollama"}:
            raise ValueError(f"unsupported summary provider: {self.provider}")

    @property
    def base_url_or_default(self) -> str:
        if self.base_url:
            return self.base_url
        return f"http://{self.host}:{self.port}"

    def build_prompt(self, text: str, style: str | None = None) -> str:
        template = self.prompt_template or DEFAULT_SUMMARY_PROMPT_TEMPLATE
        if style:
            style_instruction = self._style_instruction(style)
            if style_instruction:
                # Insert style instruction before the requirements or at the top
                template = self._inject_style(template, style_instruction)
        return template.format(text=text[: self.max_input_chars])

    @staticmethod
    def _style_instruction(style: str) -> str:
        mapping = {
            "concise": "请尽量简洁，控制在3条要点以内。",
            "detailed": "请给出更详细的总结，可以包含更多细节。",
            "action_items": "请重点提炼待办事项和责任人。",
            "meeting_minutes": "请按会议纪要格式输出，包含议题、结论、待办。",
        }
        return mapping.get(style, "")

    @staticmethod
    def _inject_style(template: str, instruction: str) -> str:
        marker = "要求："
        if marker in template:
            return template.replace(marker, f"额外要求：\n{instruction}\n\n{marker}", 1)
        return f"{instruction}\n\n{template}"


def load_summary_config(path: str | Path | None = None) -> SummaryConfig:
    if path is None:
        path = Path("summary.yaml")
    config_path = Path(path)
    if not config_path.is_file():
        return SummaryConfig()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "summary" not in raw:
        return SummaryConfig()
    summary = raw["summary"]
    if not isinstance(summary, dict):
        return SummaryConfig()

    kwargs: dict[str, Any] = {}
    for key in (
        "enabled",
        "provider",
        "offline_enabled",
        "realtime_enabled",
        "auto_start",
        "base_url",
        "api_key",
        "model",
        "llama_server_bin",
        "host",
        "port",
        "ctx_size",
        "num_threads",
        "n_gpu_layers",
        "temperature",
        "timeout",
        "max_input_chars",
        "max_tokens",
        "min_text_chars",
        "realtime_min_chars",
        "fallback_on_error",
        "enable_thinking",
        "transcript_dir",
        "language",
        "prompt",
        "prompt_template",
    ):
        if key in summary:
            kwargs[key] = summary[key]
    if "model_path" in summary:
        kwargs["model_path"] = Path(summary["model_path"])

    return SummaryConfig(**kwargs)
