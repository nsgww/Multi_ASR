from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from multi_stream_asr.asr.summary.config import SummaryConfig
from multi_stream_asr.asr.summary.summary_client import SummaryClient
from multi_stream_asr.asr.text_filter import SensitiveWordFilter
from multi_stream_asr.session.process_pool import ProcessPoolSessionManager, Session
from multi_stream_asr.web import messages as msg
from multi_stream_asr.web.transcript_buffer import TranscriptBuffer, consume_event
from multi_stream_asr.web.wakeup_gate import WakeupGate, WakeupSettings
from multi_stream_asr.web.ws_utils import is_client_disconnect

_logger = logging.getLogger(__name__)


class StreamHandler:
    """Handles a single /ws/stream WebSocket connection."""

    def __init__(
        self,
        websocket: WebSocket,
        session_manager: ProcessPoolSessionManager,
        *,
        summary_config: SummaryConfig,
        output_dir: Path,
        sample_rate: int = 16000,
        wakeup_enabled: bool = True,
        sensitive_filter: SensitiveWordFilter | None = None,
        wakeup_settings: WakeupSettings | None = None,
    ) -> None:
        self._websocket = websocket
        self._session_manager = session_manager
        self._summary_config = summary_config
        self._output_dir = Path(output_dir)
        self._sample_rate = sample_rate
        self._sensitive_filter = sensitive_filter or SensitiveWordFilter([])
        self._gate = WakeupGate(enabled=wakeup_enabled, settings=wakeup_settings)
        self._session: Session | None = None
        self._config_received = False
        self._started = False
        self._enable_summary = False
        self._paused = False
        self._finished_event = asyncio.Event()
        self._outgoing_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._receive_task: asyncio.Task[None] | None = None
        self._send_task: asyncio.Task[None] | None = None
        self._closed = False
        self._transcript = TranscriptBuffer()
        self._transcript_path: Path | None = None

    async def run(self) -> None:
        await self._websocket.accept()
        self._send_task = asyncio.create_task(self._send_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())
        # Session/lane is allocated on start, not on connect.
        await self._send(msg.status_message("ready", message="send config then start"))

        try:
            await asyncio.gather(
                self._send_task,
                self._receive_task,
                return_exceptions=True,
            )
        finally:
            await self._cleanup()

    async def _receive_loop(self) -> None:
        client_gone = False
        try:
            while True:
                data = await self._websocket.receive()
                if data.get("type") == "websocket.disconnect":
                    client_gone = True
                    _logger.info(
                        "stream client disconnected (lane=%s)",
                        self._session.lane_id if self._session else "?",
                    )
                    break
                if "bytes" in data:
                    await self._handle_binary(data["bytes"])
                elif "text" in data:
                    await self._handle_text(data["text"])
        except WebSocketDisconnect:
            client_gone = True
            _logger.info(
                "stream client disconnected (lane=%s)",
                self._session.lane_id if self._session else "?",
            )
        except Exception as exc:
            if is_client_disconnect(exc):
                client_gone = True
                _logger.info(
                    "stream client disconnected (lane=%s)",
                    self._session.lane_id if self._session else "?",
                )
            else:
                _logger.exception("stream receive error")
                await self._send(msg.error_message(f"receive error: {exc}"))
        finally:
            # Mic audio stops with the client; finalize then release in cleanup.
            if self._started and not self._finished_event.is_set():
                await self._finish_pipeline(notify=not client_gone)

    async def _send_loop(self) -> None:
        try:
            while True:
                message = await self._outgoing_queue.get()
                if message is None:
                    break
                await self._websocket.send_json(message)
        except Exception as exc:
            if is_client_disconnect(exc):
                _logger.info(
                    "stream send stopped; client disconnected (lane=%s)",
                    self._session.lane_id if self._session else "?",
                )
            else:
                _logger.exception("stream send error")

    async def _handle_text(self, text: str) -> None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            await self._send(msg.error_message(f"invalid JSON: {exc}"))
            return
        if not isinstance(payload, dict):
            await self._send(msg.error_message("message must be a JSON object"))
            return

        msg_type = payload.get("type")
        if msg_type == "config":
            await self._handle_config(payload)
        elif msg_type == "start":
            await self._handle_start()
        elif msg_type == "pause":
            await self._handle_pause()
        elif msg_type == "resume":
            await self._handle_resume()
        elif msg_type in ("finish", "done"):
            self._enable_summary = bool(payload.get("enable_summary", False))
            await self._finish_pipeline()
        elif msg_type == "new_transcription":
            await self._handle_new_transcription()
        else:
            await self._send(msg.error_message(f"unknown message type: {msg_type}"))

    async def _handle_binary(self, data: bytes) -> None:
        if self._session is None or not self._started or self._paused:
            return
        self._session_manager.push_audio(self._session, data)
        self._audio_chunk_count = getattr(self, "_audio_chunk_count", 0) + 1
        if self._audio_chunk_count == 1 or self._audio_chunk_count % 20 == 0:
            _logger.info(
                "stream lane=%d received %d audio chunks (last %d bytes)",
                self._session.lane_id,
                self._audio_chunk_count,
                len(data),
            )

    async def _handle_pause(self) -> None:
        if not self._started or self._paused:
            await self._send(msg.error_message("transcription not running or already paused"))
            return
        self._paused = True
        await self._send(
            msg.status_message("paused", lane_id=self._session.lane_id)
        )

    async def _handle_resume(self) -> None:
        if not self._started or not self._paused:
            await self._send(msg.error_message("transcription not paused"))
            return
        self._paused = False
        await self._send(
            msg.status_message("resumed", lane_id=self._session.lane_id)
        )

    async def _handle_config(self, payload: dict[str, Any]) -> None:
        if self._config_received:
            await self._send(msg.error_message("config already received"))
            return
        sample_rate = payload.get("sample_rate", self._sample_rate)
        channels = payload.get("channels", 1)
        width = payload.get("width", 2)
        if sample_rate != self._sample_rate:
            await self._send(
                msg.error_message(
                    f"config sample_rate={sample_rate} does not match "
                    f"connection sample_rate={self._sample_rate}"
                )
            )
            return
        if channels != 1 or width != 2:
            await self._send(
                msg.error_message(
                    "unsupported audio format; required: mono PCM16"
                )
            )
            return
        self._config_received = True
        lane = self._session.lane_id if self._session is not None else -1
        await self._send(msg.config_ok_message(lane))

    async def _handle_start(self) -> None:
        if self._started:
            await self._send(msg.error_message("transcription already started"))
            return
        # config is optional: connection query params already declare sample_rate,
        # and channels=1/width=2 is the only supported format.
        if self._session is None:
            try:
                self._session = await self._session_manager.create_session(
                    on_event=self._on_session_event,
                    input_sample_rate=self._sample_rate,
                )
            except Exception as exc:
                await self._send(msg.error_message(f"failed to create session: {exc}"))
                return
        self._started = True
        self._finished_event.clear()
        self._gate.reset()
        await self._session_manager.start_session(self._session)
        await self._send(msg.status_message("started", lane_id=self._session.lane_id))

    async def _finish_pipeline(self, *, notify: bool = True) -> None:
        if self._session is None or not self._started or self._finished_event.is_set():
            return
        self._session_manager.push_audio(self._session, None)
        session = self._session
        try:
            await asyncio.wait_for(
                session.finished_event.wait(),
                timeout=self._session_manager._lifecycle_timeout_seconds,
            )
        except asyncio.TimeoutError:
            if notify:
                await self._send(msg.error_message("transcription finish timed out"))
        self._finished_event.set()
        lane_id = session.lane_id
        # Save transcript and optionally generate summary before releasing lane.
        transcript_path = await self._save_transcript()
        if transcript_path is not None:
            await self._send(msg.saved_message(lane_id, str(transcript_path)))
            summary = await self._maybe_generate_summary()
            if summary is not None:
                await self._send(msg.summary_message(summary, lane_id=lane_id))
        # Release the lane immediately; the WebSocket stays open.
        with contextlib.suppress(Exception):
            await self._session_manager.stop_session(session)
        self._session = None
        self._started = False
        self._paused = False
        if notify:
            await self._send(
                msg.status_message(
                    "ready_for_new_transcription",
                    lane_id=lane_id,
                )
            )

    async def _handle_new_transcription(self) -> None:
        # Release any existing session first.
        if self._session is not None and not self._finished_event.is_set():
            await self._send(
                msg.status_message("finalizing", lane_id=self._session.lane_id)
            )
            self._session_manager.push_audio(self._session, None)
            try:
                await asyncio.wait_for(
                    self._session.finished_event.wait(),
                    timeout=self._session_manager._lifecycle_timeout_seconds,
                )
            except asyncio.TimeoutError:
                await self._send(msg.error_message("finalize timed out"))
                return
        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session_manager.stop_session(self._session)
            self._session = None

        # Reset transcript buffer for the new round.
        self._transcript.reset()
        self._transcript_path = None

        # Create a fresh session directly for the next transcription round.
        try:
            self._session = await self._session_manager.create_session(
                on_event=self._on_session_event,
                input_sample_rate=self._sample_rate,
            )
        except Exception as exc:
            await self._send(msg.error_message(f"failed to create session: {exc}"))
            return

        self._started = False
        self._enable_summary = False
        self._config_received = False
        self._paused = False
        self._finished_event.clear()
        await self._send(
            msg.status_message("ready", lane_id=self._session.lane_id)
        )

    def _on_session_event(self, message: dict[str, Any]) -> None:
        for m in self._gate.process(message):
            if m.get("type") == "transcript":
                seg = m.get("segment")
                if isinstance(seg, dict) and seg.get("text"):
                    m = dict(m, segment=dict(seg, text=self._sensitive_filter.filter(seg["text"])))
            # final 文本与 speaker_remap 都喂给同一个缓冲，落盘时统一渲染
            consume_event(self._transcript, m)
            asyncio.create_task(self._outgoing_queue.put(m))

    async def _send(self, message: dict[str, Any]) -> None:
        await self._outgoing_queue.put(message)

    def _transcript_filename(self) -> str:
        ts = int(time.time() * 1000)
        lane = self._session.lane_id if self._session is not None else 0
        return f"spacemit-asr-lane{lane}-{ts}.txt"

    async def _save_transcript(self) -> Path | None:
        if not self._transcript:
            return None
        full_text = self._transcript.render()
        if not full_text:
            return None
        transcript_dir = self._output_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        path = transcript_dir / self._transcript_filename()
        await asyncio.to_thread(path.write_text, full_text, encoding="utf-8")
        self._transcript_path = path
        return path

    async def _maybe_generate_summary(self) -> dict[str, Any] | None:
        if not self._enable_summary:
            return None
        if self._transcript_path is None:
            return None
        if not self._summary_config.realtime_enabled:
            return {
                "provider": "disabled",
                "reason": "realtime summary disabled",
                "summary": "",
                "text": "",
            }
        text = await asyncio.to_thread(self._transcript_path.read_text, encoding="utf-8")
        if not text.strip():
            return {
                "provider": "skipped",
                "reason": "empty transcript",
                "summary": "",
                "text": "",
            }
        client = SummaryClient(config=self._summary_config)
        try:
            result = await client.generate(text)
            return result.to_payload()
        except Exception as exc:
            _logger.exception("realtime summary generation failed")
            return {
                "provider": "error",
                "reason": str(exc),
                "summary": "",
                "text": "",
                "error": str(exc),
            }
        finally:
            await client.close()

    async def _cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._receive_task is not None and not self._receive_task.done():
            self._receive_task.cancel()
        if self._send_task is not None and not self._send_task.done():
            await self._outgoing_queue.put(None)
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        if self._session is not None:
            if self._started and not self._finished_event.is_set():
                with contextlib.suppress(Exception):
                    self._session_manager.push_audio(self._session, None)
            with contextlib.suppress(Exception):
                await self._session_manager.stop_session(self._session)
            self._session = None
        try:
            await self._websocket.close()
        except Exception:
            pass
