"""transcript 累积与渲染（``/ws/stream``、``/ws/multicast``、``/ws/local_mic`` 共用）。

之前三个 handler 各自复制了一份"收 final 文本 + 拼字符串"的逻辑，加说话人时
就会各改一遍、还容易漏。这里收成一份实现：

* ``add_transcript``：只在 ``is_final`` 且文本非空时记一句，连说话人一起记。
* ``add_speaker_remap``：收会话末二遍重聚类给出的 ``旧标签 -> 新标签``。
* ``render``：一次性套用 remap 后渲染成带 ``[S1]`` 前缀和表头的文本。

remap **必须原子套用**——``S1->S2`` 与 ``S2->S1`` 互换时逐条链式替换会互相覆盖，
所以这里先算出每句的最终标签再统一输出。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def speaker_sort_key(label: str) -> tuple[int, str]:
    """``S1 < S2 < S10``，避免按字符串排序出现 S10 排在 S2 前面。"""
    digits = label[1:]
    if label[:1] == "S" and digits.isdigit():
        return (int(digits), label)
    return (10**6, label)


def render_transcript(
    lines: list[tuple[str | None, str | None, str]],
    remap: Mapping[str, str] | None = None,
) -> str:
    """渲染带说话人的 transcript。

    Args:
        lines: ``(说话人标签, 真实姓名, 文本)``。标签为 ``None`` 表示该句没判出
            说话人，按纯文本输出。
        remap: 会话末二遍重聚类给出的 ``旧标签 -> 新标签``。

    Returns:
        有说话人时带表头（``# S1 → 张三（12 句）``）和 ``[S1]`` 前缀；全部没有
        说话人时退化成朴素的分行文本，与引入声纹前一致。
    """
    mapping = remap or {}
    body: list[str] = []
    counts: dict[str, int] = {}
    names: dict[str, str] = {}

    for label, name, text in lines:
        resolved = mapping.get(label, label) if label else None
        if resolved is None:
            body.append(text)
            continue
        counts[resolved] = counts.get(resolved, 0) + 1
        if name:
            names.setdefault(resolved, name)
        body.append(f"[{resolved}] {text}")

    if not counts:
        return "\n".join(body).strip()

    header = ["# 说话人"]
    for label in sorted(counts, key=speaker_sort_key):
        suffix = f" → {names[label]}" if label in names else ""
        header.append(f"# {label}{suffix}（{counts[label]} 句）")
    return "\n".join([*header, "", *body]).strip()


class TranscriptBuffer:
    """一句一句攒 transcript，落盘时统一渲染。"""

    def __init__(self) -> None:
        self._lines: list[tuple[str | None, str | None, str]] = []
        self._remap: dict[str, str] = {}

    def __bool__(self) -> bool:
        return bool(self._lines)

    def __len__(self) -> int:
        return len(self._lines)

    @property
    def lines(self) -> tuple[tuple[str | None, str | None, str], ...]:
        return tuple(self._lines)

    @property
    def speaker_remap(self) -> dict[str, str]:
        return dict(self._remap)

    def reset(self) -> None:
        """新一轮转写开始时清空。"""
        self._lines.clear()
        self._remap.clear()

    def add_transcript(self, message: Mapping[str, Any]) -> bool:
        """记录一条 transcript 消息；非 final / 空文本返回 ``False``。

        调用方必须已经把敏感词过滤后的文本写回 ``segment.text``——这里不做过滤，
        避免过滤逻辑出现两份。
        """
        if not message.get("is_final"):
            return False
        segment = message.get("segment")
        if not isinstance(segment, Mapping):
            return False
        text = segment.get("text", "")
        if not text:
            return False

        speaker = segment.get("speaker")
        label: str | None = None
        name: str | None = None
        if isinstance(speaker, Mapping):
            raw_label = speaker.get("label")
            raw_name = speaker.get("name")
            label = str(raw_label) if raw_label else None
            name = str(raw_name) if raw_name else None

        self._lines.append((label, name, str(text)))
        return True

    def add_speaker_remap(self, message: Mapping[str, Any]) -> bool:
        """记录二遍重聚类的重标映射；空映射返回 ``False``。"""
        mapping = message.get("mapping")
        if not isinstance(mapping, Mapping) or not mapping:
            return False
        self._remap = {str(key): str(value) for key, value in mapping.items()}
        return True

    def render(self) -> str:
        return render_transcript(self._lines, self._remap)


def consume_event(buffer: TranscriptBuffer, message: Mapping[str, Any]) -> None:
    """把会话事件喂给 buffer：``transcript`` 记句子，``speaker_remap`` 记映射。

    其它类型一律忽略，调用方无需自己分支。
    """
    msg_type = message.get("type")
    if msg_type == "transcript":
        buffer.add_transcript(message)
    elif msg_type == "speaker_remap":
        buffer.add_speaker_remap(message)
