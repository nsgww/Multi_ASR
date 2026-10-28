from __future__ import annotations

from dataclasses import asdict
from typing import Any

from multi_stream_asr.models import ASREvent, EventStatus


def status_message(status: str, *, lane_id: int | None = None, message: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "status", "status": status}
    if lane_id is not None:
        payload["lane_id"] = lane_id
    if message:
        payload["message"] = message
    return payload


def config_ok_message(lane_id: int) -> dict[str, Any]:
    return {"type": "config_ok", "lane_id": lane_id}


def transcript_message(event: ASREvent) -> dict[str, Any]:
    segment: dict[str, Any] = {"text": event.text}
    # 声纹关闭时不带 speaker 字段，保持与改动前一致的报文形状
    if event.speaker is not None:
        segment["speaker"] = event.speaker.to_payload()
    return {
        "type": "transcript",
        "lane_id": event.stream_id,
        "sentence_id": event.sentence_id,
        "revision": event.revision,
        "is_partial": event.status is EventStatus.PARTIAL,
        "is_final": event.is_final,
        "segment": segment,
        "timestamp_ms": event.timestamp_ms,
    }


def speaker_remap_message(mapping: dict[str, str], *, lane_id: int) -> dict[str, Any]:
    """会话末二遍重聚类结果：``旧标签 -> 新标签``。

    前端必须**原子地**套用整张表，不能链式逐条替换（``S1->S2`` 与
    ``S2->S1`` 互换时会互相覆盖）。
    """
    return {
        "type": "speaker_remap",
        "lane_id": lane_id,
        "mapping": dict(mapping),
    }


def saved_message(lane_id: int, path: str) -> dict[str, Any]:
    return {"type": "saved", "lane_id": lane_id, "path": path}


def finished_message(lane_id: int) -> dict[str, Any]:
    return {"type": "finished", "lane_id": lane_id}


def summary_message(summary: dict[str, Any], *, lane_id: int | None = None) -> dict[str, Any]:
    payload = dict(summary)
    payload["type"] = "summary"
    if lane_id is not None:
        payload["lane_id"] = lane_id
    return payload


def error_message(message: str, *, lane_id: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "error", "message": message}
    if lane_id is not None:
        payload["lane_id"] = lane_id
    return payload


def multicast_ok_message(*, group: str, port: int, lane_id: int) -> dict[str, Any]:
    return {
        "type": "multicast_ok",
        "group": group,
        "port": port,
        "lane_id": lane_id,
    }


def kws_message(
    *,
    keyword: str,
) -> dict[str, Any]:
    """Build KWS payload containing only the detected keyword."""
    return {
        "type": "kws",
        "keyword": keyword,
    }


def asr_event_to_frontend(event: ASREvent) -> dict[str, Any] | None:
    if event.status in {EventStatus.PARTIAL, EventStatus.FINAL, EventStatus.PENDING}:
        return transcript_message(event)
    if event.status is EventStatus.QWEN_ERROR:
        return error_message(event.text, lane_id=event.stream_id)
    if event.status is EventStatus.FATAL_ERROR:
        return error_message(event.text, lane_id=event.stream_id)
    return None
