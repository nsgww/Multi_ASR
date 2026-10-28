from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EventStatus(str, Enum):
    PARTIAL = "partial"
    PENDING = "pending"
    FINAL = "final"
    QWEN_ERROR = "qwen_error"
    FATAL_ERROR = "fatal_error"
    LANE_FINISHED = "lane_finished"
    STATS_UPDATED = "stats_updated"


class ResultType(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


class LaneFinalState(str, Enum):
    FINISHED = "finished"
    STOPPED = "stopped"
    FATAL = "fatal"


@dataclass(frozen=True)
class AudioFrame:
    stream_id: int
    frame_index: int
    pcm: np.ndarray
    timestamp_ms: int
    is_finish: bool = False

    @classmethod
    def finish(cls, stream_id: int, frame_index: int, timestamp_ms: int) -> "AudioFrame":
        return cls(
            stream_id=stream_id,
            frame_index=frame_index,
            pcm=np.asarray([], dtype=np.float32),
            timestamp_ms=timestamp_ms,
            is_finish=True,
        )


@dataclass(frozen=True)
class OnlineRecognition:
    stream_id: int
    sentence_id: int
    revision: int
    text: str
    timestamp_ms: int
    is_endpoint: bool = False
    last_token_stream_s: float | None = None


def speaker_label(index: int) -> str:
    """说话人簇下标 -> 展示标签（0 -> ``S1``）。"""
    return f"S{index + 1}"


def speaker_id(index: int) -> str:
    """说话人簇下标 -> 机器标识（0 -> ``spk_1``）。"""
    return f"spk_{index + 1}"


@dataclass(frozen=True)
class SpeakerInfo:
    """一个 VAD 句段的说话人归属。

    Attributes:
        id: 会话内稳定的机器标识，如 ``spk_1``。
        label: 展示用短标签，如 ``S1``。
        name: 声纹库命中时的真实姓名；未命中为 ``None``。
        confidence: 与该说话人簇质心的余弦相似度（0~1）。刚刚新建的簇记 1.0，
            但 ``is_provisional`` 为 ``True`` 表示"尚无足够证据"。
        is_provisional: ``True`` 表示还没经过二遍全局重聚类校准。
    """

    id: str
    label: str
    name: str | None = None
    confidence: float = 0.0
    is_provisional: bool = True

    @property
    def display_name(self) -> str:
        """给人看的名字：优先真实姓名，否则用 ``S1`` 这类标签。"""
        return self.name or self.label

    def to_payload(self) -> dict[str, object]:
        """序列化成前端 / 落盘用的字典。"""
        return {
            "id": self.id,
            "label": self.label,
            "name": self.name,
            "confidence": round(float(self.confidence), 4),
            "is_provisional": bool(self.is_provisional),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "SpeakerInfo":
        """从 ``to_payload`` 的字典还原（跨进程 IPC 用）。"""
        name = payload.get("name")
        return cls(
            id=str(payload["id"]),
            label=str(payload["label"]),
            name=None if name is None else str(name),
            confidence=float(payload.get("confidence", 0.0)),
            is_provisional=bool(payload.get("is_provisional", True)),
        )

    def relabel(self, index: int) -> "SpeakerInfo":
        """二遍重聚类后换标签，同时清掉 provisional 标记。"""
        return SpeakerInfo(
            id=speaker_id(index),
            label=speaker_label(index),
            name=self.name,
            confidence=self.confidence,
            is_provisional=False,
        )


@dataclass(frozen=True)
class ASREvent:
    event_type: str
    stream_id: int
    sentence_id: int
    revision: int
    status: EventStatus
    result_type: ResultType
    is_final: bool
    text: str
    timestamp_ms: int
    speaker: SpeakerInfo | None = None

    @classmethod
    def asr_result(
        cls,
        *,
        stream_id: int,
        sentence_id: int,
        revision: int,
        status: EventStatus,
        text: str,
        timestamp_ms: int,
        speaker: SpeakerInfo | None = None,
    ) -> "ASREvent":
        is_final = status is EventStatus.FINAL
        result_type = (
            ResultType.OFFLINE
            if status in {EventStatus.FINAL, EventStatus.QWEN_ERROR}
            else ResultType.ONLINE
        )
        return cls(
            event_type="asr_result",
            stream_id=stream_id,
            sentence_id=sentence_id,
            revision=revision,
            status=status,
            result_type=result_type,
            is_final=is_final,
            text=text,
            timestamp_ms=timestamp_ms,
            speaker=speaker,
        )


@dataclass(frozen=True)
class LaneMetrics:
    stream_id: int
    processed_frames: int = 0
    dropped_frames: int = 0
    backlogged_frames: int = 0
    offline_requests: int = 0
    final_state: LaneFinalState = LaneFinalState.STOPPED


@dataclass(frozen=True)
class QwenSentenceFailure:
    stream_id: int
    sentence_id: int
    revision: int
    pending_text: str
    timestamp_ms: int
    request: dict[str, object]
