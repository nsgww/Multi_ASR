from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from multi_stream_asr.models import ASREvent, EventStatus


_TranscriptKey = tuple[int, int]
_StatusKey = tuple[int, int, EventStatus]
_ResultKey = _TranscriptKey | _StatusKey
_TRANSCRIPT_STATUSES = frozenset(
    {EventStatus.PARTIAL, EventStatus.PENDING, EventStatus.FINAL}
)


class FrontendEventBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[ASREvent | _TranscriptKey] = asyncio.Queue()
        self._results: dict[_ResultKey, dict[str, Any]] = {}
        self._pending_partial_events: dict[_TranscriptKey, ASREvent] = {}
        self._pending_partial_keys: set[_TranscriptKey] = set()
        self._websockets: set[Any] = set()
        self._result_retention_limit: int | None = None
        self._queue_retention_limit: int | None = None
        self._pruned_sentence_through: dict[int, int] = {}

    def set_result_retention_limit(self, limit: int | None) -> None:
        if limit is not None and limit < 1:
            raise ValueError("result retention limit must be positive or null")
        self._result_retention_limit = limit
        self._queue_retention_limit = limit * 4 if limit is not None else None
        if limit is None:
            return
        for stream_id in {key[0] for key in self._results}:
            self._prune_stream_results(stream_id)

    def _prune_stream_results(self, stream_id: int) -> None:
        limit = self._result_retention_limit
        if limit is None:
            return
        keys = [key for key in self._results if key[0] == stream_id]
        sentence_ids = list(dict.fromkeys(key[1] for key in keys))
        retained_sentence_ids = set(sentence_ids[-limit:])
        pruned_sentence_ids: set[int] = set()
        for key in keys:
            if key[1] in retained_sentence_ids:
                continue
            self._results.pop(key, None)
            pruned_sentence_ids.add(key[1])
            transcript_key = (key[0], key[1])
            self._pending_partial_events.pop(transcript_key, None)
            self._pending_partial_keys.discard(transcript_key)
        if pruned_sentence_ids:
            self._pruned_sentence_through[stream_id] = max(
                self._pruned_sentence_through.get(stream_id, -1),
                *pruned_sentence_ids,
            )

    def _enqueue(self, item: ASREvent | _TranscriptKey) -> None:
        limit = self._queue_retention_limit
        if limit is not None and self._queue.qsize() >= limit:
            try:
                discarded = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                discarded = None
            if isinstance(discarded, tuple):
                self._pending_partial_events.pop(discarded, None)
                self._pending_partial_keys.discard(discarded)
        self._queue.put_nowait(item)

    def publish(self, event: ASREvent) -> None:
        transcript_key = (event.stream_id, event.sentence_id)
        if (
            event.status is EventStatus.QWEN_ERROR
            and transcript_key not in self._results
            and event.sentence_id
            <= self._pruned_sentence_through.get(event.stream_id, -1)
        ):
            return
        key: _ResultKey = (
            (*transcript_key, event.status)
            if event.status not in _TRANSCRIPT_STATUSES
            else transcript_key
        )
        current = self._results.get(key)
        if event.status is EventStatus.PARTIAL:
            if current is not None and event.revision <= current["revision"]:
                return
            self._results[key] = {
                "revision": event.revision,
                "text": event.text,
                "status": event.status,
                "result_type": event.result_type,
                "is_final": event.is_final,
            }
            self._prune_stream_results(event.stream_id)
            self._pending_partial_events[transcript_key] = event
            if transcript_key not in self._pending_partial_keys:
                self._pending_partial_keys.add(transcript_key)
                self._enqueue(transcript_key)
            return

        if current is None or event.revision > current["revision"]:
            self._results[key] = {
                "revision": event.revision,
                "text": event.text,
                "status": event.status,
                "result_type": event.result_type,
                "is_final": event.is_final,
            }
            self._prune_stream_results(event.stream_id)

        self._enqueue(event)

    async def next_event(self) -> ASREvent:
        while True:
            item = await self._queue.get()
            if isinstance(item, tuple):
                key = item
                self._pending_partial_keys.discard(key)
                event = self._pending_partial_events.pop(key, None)
                if event is None:
                    continue
                return event
            return item

    def _is_stale_partial(self, event: ASREvent) -> bool:
        if event.status is not EventStatus.PARTIAL:
            return False
        current = self._results.get((event.stream_id, event.sentence_id))
        return current is not None and event.revision < current["revision"]

    def register(self, websocket: Any) -> None:
        self._websockets.add(websocket)

    def unregister(self, websocket: Any) -> None:
        self._websockets.discard(websocket)

    async def broadcast_loop(self) -> None:
        while True:
            event = await self.next_event()
            if self._is_stale_partial(event):
                continue
            payload = asdict(event)
            # 声纹关闭时不把 speaker:null 发出去，保持与改动前一致的报文形状
            if payload.get("speaker") is None:
                payload.pop("speaker", None)
            broken: list[Any] = []
            for websocket in tuple(self._websockets):
                if self._is_stale_partial(event):
                    break
                try:
                    await websocket.send_json(payload)
                except Exception:
                    broken.append(websocket)
            for websocket in broken:
                self.unregister(websocket)

    def snapshot_results(self) -> dict[_ResultKey, dict[str, Any]]:
        return {
            key: value.copy()
            for key, value in self._results.items()
        }

    def restore_results(self, results: dict[_ResultKey, dict[str, Any]]) -> None:
        self.clear_results()
        self._results = {
            key: value.copy()
            for key, value in results.items()
        }
        if self._result_retention_limit is not None:
            for stream_id in {key[0] for key in self._results}:
                self._prune_stream_results(stream_id)

    def clear_results(self) -> None:
        self._results.clear()
        self._pruned_sentence_through.clear()
        self._pending_partial_events.clear()
        self._pending_partial_keys.clear()
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


frontend_event_bus = FrontendEventBus()
