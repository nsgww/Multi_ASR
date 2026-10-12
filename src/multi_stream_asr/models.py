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
