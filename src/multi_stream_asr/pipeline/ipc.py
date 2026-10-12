from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from multi_stream_asr.models import ASREvent, EventStatus, LaneFinalState, LaneMetrics, ResultType


LaneMessageType = str


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class LaneIpcMessage:
    type: LaneMessageType
    stream_id: int
    seq: int
    time_ms: int
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        self._validate_json_envelope()

    def _validate_json_envelope(self) -> None:
        if not isinstance(self.type, str):
            raise TypeError("type must be a string")
        for name, value in (
            ("stream_id", self.stream_id),
            ("seq", self.seq),
            ("time_ms", self.time_ms),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a JSON object")
        json.dumps(
            {
                "type": self.type,
                "stream_id": self.stream_id,
                "seq": self.seq,
                "time_ms": self.time_ms,
                "payload": self.payload,
            },
            allow_nan=False,
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "type": self.type,
            "stream_id": self.stream_id,
            "seq": self.seq,
            "time_ms": self.time_ms,
            "payload": self.payload,
        }
        self._validate_json_envelope()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LaneIpcMessage":
        return cls(
            type=str(value["type"]),
            stream_id=int(value["stream_id"]),
            seq=int(value["seq"]),
            time_ms=int(value["time_ms"]),
            payload=dict(value.get("payload", {})),
        )


def make_message(
    *,
    message_type: LaneMessageType,
    stream_id: int,
    seq: int,
    payload: dict[str, Any],
    time_ms: int | None = None,
) -> LaneIpcMessage:
    return LaneIpcMessage(
        type=message_type,
        stream_id=stream_id,
        seq=seq,
        time_ms=now_ms() if time_ms is None else time_ms,
        payload=payload,
    )


def asr_event_to_payload(event: ASREvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "stream_id": event.stream_id,
        "sentence_id": event.sentence_id,
        "revision": event.revision,
        "status": event.status.value,
        "result_type": event.result_type.value,
        "is_final": event.is_final,
        "text": event.text,
        "timestamp_ms": event.timestamp_ms,
    }


def asr_event_from_payload(payload: dict[str, Any]) -> ASREvent:
    return ASREvent(
        event_type=str(payload["event_type"]),
        stream_id=int(payload["stream_id"]),
        sentence_id=int(payload["sentence_id"]),
        revision=int(payload["revision"]),
        status=EventStatus(str(payload["status"])),
        result_type=ResultType(str(payload["result_type"])),
        is_final=bool(payload["is_final"]),
        text=str(payload.get("text", "")),
        timestamp_ms=int(payload["timestamp_ms"]),
    )


def lane_metrics_to_payload(metrics: LaneMetrics) -> dict[str, Any]:
    return {
        "stream_id": metrics.stream_id,
        "processed_frames": metrics.processed_frames,
        "dropped_frames": metrics.dropped_frames,
        "backlogged_frames": metrics.backlogged_frames,
        "offline_requests": metrics.offline_requests,
        "final_state": metrics.final_state.value,
    }


def lane_metrics_from_payload(payload: dict[str, Any]) -> LaneMetrics:
    return LaneMetrics(
        stream_id=int(payload["stream_id"]),
        processed_frames=int(payload.get("processed_frames", 0)),
        dropped_frames=int(payload.get("dropped_frames", 0)),
        backlogged_frames=int(payload.get("backlogged_frames", 0)),
        offline_requests=int(payload.get("offline_requests", 0)),
        final_state=LaneFinalState(str(payload.get("final_state", LaneFinalState.STOPPED.value))),
    )
