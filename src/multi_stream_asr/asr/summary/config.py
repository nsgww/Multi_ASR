from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SUMMARY_PROMPT_TEMPLATE = """\
/no_think
你是语音内容摘要助手。请根据转写文本生成中文核心内容总结。

目标：
让读者快速掌握音频在讲什么、关键观点、完整要点与结论。

要求：
1. 只基于原文，不得补充或推测
2. 必须归纳总结，不逐句复述，去除口语、重复、无意义填充
3. 抓取核心信息：主题、关键观点、发生了什么、讨论了什么
4. 保留关键事实：人名、品牌、事项、结论、决定、问题、时间
5. 若原文有编号方法/步骤/清单，必须完整保留全部条目，每条只用一句短话
6. 全文控制在 500-800 字，优先短句列表，不要长段落
7. 商业合作/广告植入如有，单独用一条简要说明，不要当成会议待办

输出格式：
- **主题**：一句话
- **核心观点**：1-3句
- **要点**：
  1. ...
  2. ...
  （按原文完整列出）
- **结论**：如有
- **其他**：商业植入/补充信息（如有）

转写内容：
{text}
"""

DEFAULT_CHUNK_PROMPT_TEMPLATE = """\
/no_think
你是语音内容摘要助手。这是长音频的其中一段转写，请只总结本段。

要求：
1. 只基于本段原文，不得补充或推测
2. 用短句要点列表，去掉口语和重复
3. 若本段出现编号方法/步骤，保留编号与要点（一句一条）
4. 本段摘要控制在 250 字以内，不要写主题/结论套话

输出格式：
- ...
- ...

本段转写：
{text}
"""

DEFAULT_MERGE_PROMPT_TEMPLATE = """\
/no_think
你是语音内容摘要助手。下面是同一段音频按时间顺序切分后得到的分段摘要，请合并为一份完整中文总结。

要求：
1. 只基于分段摘要，不得补充或推测
2. 去重合并，保持时间/逻辑顺序
3. 编号方法、步骤、清单必须完整保留，不得因合并而丢掉后半段
4. 去除口语、重复与互相矛盾的冗余表述
5. 商业合作/广告植入如有，单独一条简要说明
6. 全文控制在 500-800 字
7. 输出结构：主题、核心观点、要点、结论、其他（如有）

分段摘要：
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
    ctx_size: int = 8192
    num_threads: int = 6
    n_gpu_layers: int = 0
    temperature: float = 0.2
    # 采样惩罚（gemma 等易退化模型需要；None 表示不发送该参数）
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    # 单次 LLM 请求超时（秒）。边端 CPU 解码慢，但应远小于“假死”
    timeout: int = 240
    # 分块粒度：越大块数越少、总轮次越少（边端更重要）
    max_input_chars: int = 10000
    # 终稿/单段全文摘要的最大生成长度；边端 ~15-20 tok/s，768≈40s
    max_tokens: int = 768
    # 分段摘要更短，加快 map 阶段；<=0 表示复用 max_tokens
    chunk_max_tokens: int = 384
    min_text_chars: int = 20
    realtime_min_chars: int = 80
    fallback_on_error: bool = True
    enable_thinking: bool = False
    transcript_dir: str = "outputs/transcripts"
    language: str = "zh"
    prompt: str = ""
    prompt_template: str = DEFAULT_SUMMARY_PROMPT_TEMPLATE
    chunk_prompt_template: str = DEFAULT_CHUNK_PROMPT_TEMPLATE
    merge_prompt_template: str = DEFAULT_MERGE_PROMPT_TEMPLATE
    # 模型自带的 system prompt（如会议助理微调模型的 system_prompt.txt）。
    # 设置后请求改为 system+user 两段消息，user 内容为转写原文。
    system_prompt: str = ""
    # 从该文件读取 system prompt（优先级低于直接配置的 system_prompt 文本）
    system_prompt_path: str = ""

    def __post_init__(self) -> None:
        if self.provider not in {"local", "llama", "openai", "ollama"}:
            raise ValueError(f"unsupported summary provider: {self.provider}")

    @property
    def base_url_or_default(self) -> str:
        if self.base_url:
            return self.base_url
        return f"http://{self.host}:{self.port}"

    def build_prompt(
        self,
        text: str,
        style: str | None = None,
        *,
        partial: bool = False,
    ) -> str:
        # 有 system prompt 时（会议助理类微调模型），user 消息只放转写原文
        clipped = text[: self.max_input_chars]
        if self.system_prompt:
            return clipped
        if partial:
            template = self.chunk_prompt_template or DEFAULT_CHUNK_PROMPT_TEMPLATE
        else:
            template = self.prompt_template or DEFAULT_SUMMARY_PROMPT_TEMPLATE
        if style and not partial:
            style_instruction = self._style_instruction(style)
            if style_instruction:
                template = self._inject_style(template, style_instruction)
        return template.format(text=clipped)

    def build_merge_prompt(
        self, partials: list[str], style: str | None = None
    ) -> str:
        numbered = "\n\n".join(
            f"【分段摘要 {idx}】\n{part.strip()}"
            for idx, part in enumerate(partials, start=1)
            if part and part.strip()
        )
        clipped = numbered[: self.max_input_chars]
        if self.system_prompt:
            return (
                "请将以下分段摘要合并为一份完整总结，保留全部要点且去重：\n\n"
                + clipped
            )
        template = self.merge_prompt_template or DEFAULT_MERGE_PROMPT_TEMPLATE
        if style:
            style_instruction = self._style_instruction(style)
            if style_instruction:
                template = self._inject_style(template, style_instruction)
        return template.format(text=clipped)

    @staticmethod
    def _style_instruction(style: str) -> str:
        mapping = {
            "concise": "请尽量简洁，控制在3条要点以内。",
            "detailed": "请给出更详细的总结，方法/步骤类内容必须完整保留全部条目。",
            "action_items": "请重点提炼待办事项和责任人。",
            "meeting_minutes": "请按会议纪要格式输出，包含议题、结论、待办；若原文不是会议，仍按内容结构完整归纳，不要虚构待办。",
        }
        return mapping.get(style, "")

    @staticmethod
    def _inject_style(template: str, instruction: str) -> str:
        marker = "要求："
        if marker in template:
            return template.replace(marker, f"额外要求：\n{instruction}\n\n{marker}", 1)
        return f"{instruction}\n\n{template}"


def split_transcript_chunks(text: str, max_chars: int) -> list[str]:
    """按行优先切分转写，保证每块不超过 max_chars，全文不丢弃。"""
    normalized = text.strip()
    if not normalized:
        return []
    if max_chars <= 0 or len(normalized) <= max_chars:
        return [normalized]

    lines = normalized.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n".join(current).strip())
            current = []
            current_len = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        # 空行作为软分隔，保留可读性
        piece = line if line else ""
        add_len = len(piece) + (1 if current else 0)
        if piece and len(piece) > max_chars:
            flush()
            for start in range(0, len(piece), max_chars):
                chunks.append(piece[start : start + max_chars])
            continue
        if current and current_len + add_len > max_chars:
            flush()
            add_len = len(piece)
        if not piece and not current:
            continue
        current.append(piece)
        current_len += add_len

    flush()
    return [c for c in chunks if c]


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
        "repeat_penalty",
        "presence_penalty",
        "timeout",
        "max_input_chars",
        "max_tokens",
        "chunk_max_tokens",
        "min_text_chars",
        "realtime_min_chars",
        "fallback_on_error",
        "enable_thinking",
        "transcript_dir",
        "language",
        "prompt",
        "prompt_template",
        "chunk_prompt_template",
        "merge_prompt_template",
        "system_prompt",
        "system_prompt_path",
    ):
        if key in summary:
            kwargs[key] = summary[key]
    if "model_path" in summary:
        kwargs["model_path"] = Path(summary["model_path"])

    cfg = SummaryConfig(**kwargs)
    if not cfg.system_prompt and cfg.system_prompt_path:
        path = Path(cfg.system_prompt_path)
        if path.is_file():
            cfg = SummaryConfig(
                **{**cfg.__dict__, "system_prompt": path.read_text(encoding="utf-8")}
            )
    return cfg
