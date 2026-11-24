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
from multi_stream_asr.audio.local_mic import LocalMicSource, find_first_input_device
from multi_stream_asr.session.process_pool import ProcessPoolSessionManager, Session
from multi_stream_asr.web import messages as msg
from multi_stream_asr.web.keepalive import (
    close_websocket_soon,
    heartbeat_interval_seconds,
    outgoing_queue_maxsize,
    relay_outgoing,
    send_timeout_seconds,
)
from multi_stream_asr.web.transcript_buffer import TranscriptBuffer, consume_event
from multi_stream_asr.web.wakeup_gate import WakeupGate, WakeupSettings
from multi_stream_asr.web.ws_utils import is_client_disconnect

_logger = logging.getLogger(__name__)


class LocalMicHandler:
    """WebSocket handler that uses the board local microphone as ASR input.

    Control flow mirrors ``/ws/stream`` but the audio comes from a local
    pyaudio device instead of the WebSocket binary frames.
    """

    def __init__(
        self,
        websocket: WebSocket,
        session_manager: ProcessPoolSessionManager,
        *,
        summary_config: SummaryConfig,
        output_dir: Path,
        device: int | str | None = None,
        sample_rate: int = 16000,
        chunk: int = 512,
        wakeup_enabled: bool = True,
        sensitive_filter: SensitiveWordFilter | None = None,
        wakeup_settings: WakeupSettings | None = None,
    ) -> None:
        self._websocket = websocket
        self._session_manager = session_manager
        self._summary_config = summary_config
        self._output_dir = Path(output_dir)
        self._device = device
        self._sample_rate = sample_rate
        self._chunk = chunk
        self._sensitive_filter = sensitive_filter or SensitiveWordFilter([])
        self._gate = WakeupGate(enabled=wakeup_enabled, settings=wakeup_settings)
        self._session: Session | None = None
        self._mic: LocalMicSource | None = None
        self._forward_task: asyncio.Task[None] | None = None
        self._send_task: asyncio.Task[None] | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._outgoing_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._heartbeat_seconds = heartbeat_interval_seconds()
        self._send_timeout = send_timeout_seconds()
        self._outgoing_maxsize = outgoing_queue_maxsize()
        self._slow_client_dropped = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._started = False
        self._paused = False
        self._finished_event = asyncio.Event()
        self._enable_summary = False
        self._transcript = TranscriptBuffer()
        self._transcript_path: Path | None = None

    async def run(self) -> None:
        await self._websocket.accept()
        self._loop = asyncio.get_running_loop()
        self._send_task = asyncio.create_task(self._send_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())

        try:
            await self._resolve_device()
        except Exception as exc:
            await self._send(msg.error_message(f"failed to resolve microphone: {exc}"))
            await self._cleanup()
            return

        try:
            # 记录开始转写的墙钟时间，落盘文件名用
            self._started_at = time.time()
            self._session = await self._session_manager.create_session(
                on_event=self._on_session_event,
                input_sample_rate=self._sample_rate,
            )
        except Exception as exc:
            await self._send(msg.error_message(f"failed to create session: {exc}"))
            await self._cleanup()
            return

        await self._send(
            msg.status_message(
                "ready",
                lane_id=self._session.lane_id,
                message=(
                    f"local_mic ready: device={self._device} "
                    f"sample_rate={self._sample_rate} chunk={self._chunk}; "
                    "send {\"type\":\"start\"} to begin"
                ),
            )
        )

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
                    break
                if "text" in data:
                    await self._handle_text(data["text"])
        except WebSocketDisconnect:
            client_gone = True
        except Exception as exc:
            if is_client_disconnect(exc):
                client_gone = True
            else:
                _logger.exception("local_mic receive error")
                await self._send(msg.error_message(f"receive error: {exc}"))
        finally:
            if self._session is not None and self._started and not self._finished_event.is_set():
                await self._finish_pipeline(notify=not client_gone)

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
        elif msg_type == "regenerate_summary":
            await self._handle_regenerate_summary()
        else:
            await self._send(msg.error_message(f"unknown message type: {msg_type}"))

    async def _resolve_device(self) -> None:
        """未指定 device 时自动扫描并选用第一个可用输入设备。"""
        if self._device is not None:
            return
        device = await asyncio.to_thread(find_first_input_device)
        if device is None:
            raise RuntimeError("no input audio device found")
        self._device = device

    async def _handle_config(self, payload: dict[str, Any]) -> None:
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
            await self._send(msg.error_message("unsupported audio format; required: mono PCM16"))
            return
        lane = self._session.lane_id if self._session is not None else -1
        await self._send(msg.config_ok_message(lane))

    async def _handle_start(self) -> None:
        if self._started:
            await self._send(msg.error_message("transcription already started"))
            return
        await self._start_round()

    async def _handle_pause(self) -> None:
        if not self._started or self._paused:
            await self._send(msg.error_message("transcription not running or already paused"))
            return
        self._paused = True
        await self._send(msg.status_message("paused", lane_id=self._session.lane_id))

    async def _handle_resume(self) -> None:
        if not self._started or not self._paused:
            await self._send(msg.error_message("transcription not paused"))
            return
        self._paused = False
        await self._send(msg.status_message("resumed", lane_id=self._session.lane_id))

    async def _start_round(self) -> None:
        if self._mic is not None or self._started:
            await self._send(msg.error_message("local microphone already started"))
            return
        if self._session is None:
            await self._send(msg.error_message("no session"))
            return

        await self._session_manager.start_session(self._session)
        self._gate.reset()

        self._mic = LocalMicSource(
            device=self._device,
            rate=self._sample_rate,
            chunk=self._chunk,
        )
        self._mic.start()
        self._forward_task = asyncio.create_task(self._forward_loop())
        self._started = True
        self._paused = False
        self._finished_event.clear()

        await self._send(
            msg.status_message(
                "started",
                lane_id=self._session.lane_id,
                message="local microphone recording started",
            )
        )

    async def _forward_loop(self) -> None:
        try:
            while True:
                item = await self._mic.queue.get()
                if item is None:
                    break
                if not self._paused:
                    self._session_manager.push_audio(self._session, item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.exception("local_mic forward error")
            await self._send(msg.error_message(f"local microphone error: {exc}"))

    async def _finish_pipeline(self, *, notify: bool = True) -> None:
        if self._session is None or not self._started or self._finished_event.is_set():
            return
        if self._mic is not None:
            await self._mic.stop()
            self._mic = None
        if self._forward_task is not None and not self._forward_task.done():
            self._forward_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._forward_task
        self._forward_task = None

        self._session_manager.push_audio(self._session, None)
        try:
            await asyncio.wait_for(
                self._session.finished_event.wait(),
                timeout=self._session_manager._lifecycle_timeout_seconds,
            )
        except asyncio.TimeoutError:
            if notify:
                await self._send(msg.error_message("finish timed out"))

        self._finished_event.set()
        transcript_path = await self._save_transcript()
        if transcript_path is not None:
            await self._send(msg.saved_message(self._session.lane_id, str(transcript_path)))
            summary = await self._maybe_generate_summary()
            if summary is not None:
                await self._send(msg.summary_message(summary, lane_id=self._session.lane_id))

        await self._session_manager.stop_session(self._session)
        lane_id = self._session.lane_id
        self._session = None
        self._started = False
        self._paused = False
        if notify:
            await self._send(
                msg.status_message(
                    "ready_for_new_transcription",
                    lane_id=lane_id,
                    message="local microphone stopped; send new_transcription to start again",
                )
            )

    async def _handle_new_transcription(self) -> None:
        if self._session is not None and self._started and not self._finished_event.is_set():
            await self._send(msg.status_message("finalizing", lane_id=self._session.lane_id))
            await self._finish_pipeline(notify=False)
        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session_manager.stop_session(self._session)
            self._session = None

        self._transcript.reset()
        self._transcript_path = None
        self._enable_summary = False
        self._started = False
        self._paused = False
        self._finished_event.clear()

        try:
            self._session = await self._session_manager.create_session(
                on_event=self._on_session_event,
                input_sample_rate=self._sample_rate,
            )
        except Exception as exc:
            await self._send(msg.error_message(f"failed to create session: {exc}"))
            return

        await self._send(
            msg.status_message(
                "ready",
                lane_id=self._session.lane_id,
                message="send {\"type\":\"start\"} to begin",
            )
        )

    async def _handle_regenerate_summary(self) -> None:
        if self._transcript_path is None:
            await self._send(msg.error_message("no transcript available to summarize"))
            return
        lane = self._session.lane_id if self._session is not None else 0
        summary = await self._maybe_generate_summary(force=True)
        if summary is not None:
            await self._send(msg.summary_message(summary, lane_id=lane))

    def _on_session_event(self, message: dict[str, Any]) -> None:
        for m in self._gate.process(message):
            if m.get("type") == "transcript":
                seg = m.get("segment")
                if isinstance(seg, dict) and seg.get("text"):
                    m = dict(
                        m,
                        segment=dict(seg, text=self._sensitive_filter.filter(seg["text"])),
                    )
            # final 文本与 speaker_remap 都喂给同一个缓冲，落盘时统一渲染
            consume_event(self._transcript, m)
            asyncio.create_task(self._send(m))

    async def _send_loop(self) -> None:
        try:
            await relay_outgoing(
                self._outgoing_queue,
                self._websocket.send_json,
                heartbeat_seconds=self._heartbeat_seconds,
                send_timeout_seconds=self._send_timeout,
            )
        except Exception as exc:
            if is_client_disconnect(exc):
                _logger.info("local_mic send stopped; client disconnected")
            else:
                _logger.exception("local_mic send error; dropping client")
                self._drop_slow_client()

    def _enqueue(self, message: dict[str, Any]) -> None:
        """非阻塞入队；积压超限说明客户端长期不消费，主动断开。"""
        if self._slow_client_dropped:
            return
        if self._outgoing_queue.qsize() >= self._outgoing_maxsize:
            _logger.error(
                "local_mic outgoing backlog %d >= %d, dropping slow client",
                self._outgoing_queue.qsize(),
                self._outgoing_maxsize,
            )
            self._drop_slow_client()
            return
        self._outgoing_queue.put_nowait(message)

    def _drop_slow_client(self) -> None:
        if self._slow_client_dropped:
            return
        self._slow_client_dropped = True
        if self._send_task is not None and not self._send_task.done():
            self._send_task.cancel()
        if self._loop is None:
            return
        close_websocket_soon(self._loop, self._websocket)

    async def _send(self, message: dict[str, Any]) -> None:
        self._enqueue(message)

    def _transcript_filename(self, lane_id: int) -> str:
        start_ts = getattr(self, "_started_at", None) or time.time()
        stamp = time.strftime("%Y%m%d-%H%M", time.localtime(start_ts))
        return f"local_mic-lane{lane_id}-{stamp}.txt"

    async def _save_transcript(self) -> Path | None:
        if not self._transcript:
            return None
        full_text = self._transcript.render()
        if not full_text:
            return None
        transcript_dir = self._output_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        lane = self._session.lane_id if self._session is not None else 0
        path = transcript_dir / self._transcript_filename(lane)
        await asyncio.to_thread(path.write_text, full_text, encoding="utf-8")
        self._transcript_path = path
        return path

    async def _maybe_generate_summary(self, force: bool = False) -> dict[str, Any] | None:
        if not force and not self._enable_summary:
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
            _logger.exception("summary generation failed")
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
        if self._mic is not None:
            await self._mic.stop()
            self._mic = None
        if self._forward_task is not None and not self._forward_task.done():
            self._forward_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._forward_task
        if self._receive_task is not None and not self._receive_task.done():
            self._receive_task.cancel()
        if self._send_task is not None and not self._send_task.done():
            await self._outgoing_queue.put(None)
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        if self._session is not None:
            with contextlib.suppress(Exception):
                self._session_manager.push_audio(self._session, None)
            with contextlib.suppress(Exception):
                await self._session_manager.stop_session(self._session)
            self._session = None
        self._started = False
        self._paused = False
        try:
            await self._websocket.close()
        except Exception:
            pass
