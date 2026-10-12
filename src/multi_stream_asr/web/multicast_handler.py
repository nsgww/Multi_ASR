from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from multi_stream_asr.asr.summary.config import SummaryConfig
from multi_stream_asr.asr.summary.summary_client import SummaryClient
from multi_stream_asr.asr.text_filter import SensitiveWordFilter
from multi_stream_asr.audio.multicast import (
    DEFAULT_MULTICAST_GROUP,
    DEFAULT_MULTICAST_PORT,
)
from multi_stream_asr.session.manager import SessionManager
from multi_stream_asr.web import messages as msg
from multi_stream_asr.web.multicast_runtime import MulticastJob, MulticastRuntimeRegistry
from multi_stream_asr.web.wakeup_gate import WakeupGate, WakeupSettings
from multi_stream_asr.web.ws_utils import is_client_disconnect

_logger = logging.getLogger(__name__)


class MulticastHandler:
    """WebSocket control/view for a multicast ASR job.

    The UDP→ASR job is owned by ``MulticastRuntimeRegistry`` and survives
    WebSocket disconnect. Only an explicit ``finish`` (or server shutdown)
    stops and releases the lane.
    """

    def __init__(
        self,
        websocket: WebSocket,
        session_manager: SessionManager,
        runtime: MulticastRuntimeRegistry,
        *,
        summary_config: SummaryConfig,
        output_dir: Path,
        group: str = DEFAULT_MULTICAST_GROUP,
        port: int = DEFAULT_MULTICAST_PORT,
        local_ip: str = "0.0.0.0",
        packet_bytes: int = 1280,
        use_full_packet: bool = False,
        heartbeat_interval: float = 5.0,
        speexdsp_lib_path: str | None = None,
        wakeup_enabled: bool = True,
        sensitive_filter: SensitiveWordFilter | None = None,
        wakeup_settings: WakeupSettings | None = None,
    ) -> None:
        self._websocket = websocket
        self._session_manager = session_manager
        self._runtime = runtime
        self._summary_config = summary_config
        self._output_dir = Path(output_dir)
        self._sensitive_filter = sensitive_filter or SensitiveWordFilter([])
        self._gate = WakeupGate(enabled=wakeup_enabled, settings=wakeup_settings)
        self._group = group
        self._port = port
        self._local_ip = local_ip
        self._packet_bytes = packet_bytes
        self._use_full_packet = use_full_packet
        self._heartbeat_interval = heartbeat_interval
        self._speexdsp_lib_path = speexdsp_lib_path
        self._job: MulticastJob | None = None
        self._attached = False
        self._explicit_finish = False
        self._enable_summary = False
        self._outgoing_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._receive_task: asyncio.Task[None] | None = None
        self._send_task: asyncio.Task[None] | None = None
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._transcript_texts: list[str] = []
        self._transcript_path: Path | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self) -> None:
        await self._websocket.accept()
        self._loop = asyncio.get_running_loop()
        self._send_task = asyncio.create_task(self._send_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())

        existing = self._runtime.attach(self._group, self._port, self._on_job_event)
        if existing is not None:
            self._job = existing
            self._attached = True
            if existing.wakeup_active:
                # 挂回到已唤醒的 job: 门控直接置为激活
                self._gate.activate()
            await self._send(
                msg.status_message(
                    "ready",
                    lane_id=existing.session.lane_id,
                    message=(
                        f"reattached to running multicast job "
                        f"{self._group}:{self._port}"
                    ),
                )
            )
            await self._send(
                msg.status_message(
                    "started",
                    lane_id=existing.session.lane_id,
                    message="multicast job already running; receiving events",
                )
            )
        else:
            await self._send(msg.status_message("ready", message="send start to listen"))

        try:
            await asyncio.gather(
                self._send_task,
                self._receive_task,
                return_exceptions=True,
            )
        finally:
            await self._cleanup()

    async def _receive_loop(self) -> None:
        try:
            while True:
                text = await self._websocket.receive_text()
                await self._handle_text(text)
        except WebSocketDisconnect:
            _logger.info(
                "multicast client disconnected (job keeps running) %s:%d",
                self._group,
                self._port,
            )
        except Exception as exc:
            if is_client_disconnect(exc):
                _logger.info(
                    "multicast client disconnected (job keeps running) %s:%d",
                    self._group,
                    self._port,
                )
            else:
                _logger.exception("multicast receive error")
                await self._send(msg.error_message(f"receive error: {exc}"))

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
                    "multicast send stopped; client disconnected %s:%d",
                    self._group,
                    self._port,
                )
            else:
                _logger.exception("multicast send error")

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
            lane = self._job.session.lane_id if self._job else None
            await self._send(msg.config_ok_message(lane if lane is not None else -1))
        elif msg_type == "start":
            await self._handle_start()
        elif msg_type == "set_multicast_group":
            await self._handle_set_group(payload)
        elif msg_type == "pause":
            await self._handle_pause()
        elif msg_type == "resume":
            await self._handle_resume()
        elif msg_type in ("finish", "done"):
            self._enable_summary = bool(payload.get("enable_summary", False))
            await self._handle_finish()
        else:
            await self._send(msg.error_message(f"unknown message type: {msg_type}"))

    async def _handle_start(self) -> None:
        if self._job is not None and not self._job.finished:
            await self._send(
                msg.status_message(
                    "started",
                    lane_id=self._job.session.lane_id,
                    message="multicast job already running",
                )
            )
            return
        try:
            job = await self._runtime.start_job(
                group=self._group,
                port=self._port,
                local_ip=self._local_ip,
                packet_bytes=self._packet_bytes,
                use_full_packet=self._use_full_packet,
                heartbeat_interval=self._heartbeat_interval,
                speexdsp_lib_path=self._speexdsp_lib_path,
                listener=self._on_job_event,
            )
        except Exception as exc:
            _logger.exception("multicast start failed")
            await self._send(msg.error_message(f"multicast start failed: {exc}"))
            return

        self._job = job
        self._attached = True
        self._enable_summary = False
        self._transcript_texts = []
        self._transcript_path = None
        self._gate.reset()
        job.wakeup_active = False
        joined = ",".join(job.audio_source._joined_ifs) or "?"
        await self._send(
            msg.status_message(
                "started",
                lane_id=job.session.lane_id,
                message=(
                    f"listening {self._group}:{self._port} on ifaces=[{joined}]; "
                    "WS disconnect will NOT stop this job — send finish to release"
                ),
            )
        )

    async def _handle_set_group(self, payload: dict[str, Any]) -> None:
        new_group = str(payload.get("group", self._group)).strip()
        if not new_group:
            await self._send(msg.error_message("invalid multicast group"))
            return
        if self._job is not None and not self._job.finished:
            await self._send(
                msg.error_message(
                    "cannot change group while job running; finish first"
                )
            )
            return
        if self._attached:
            self._runtime.detach(self._group, self._port, self._on_job_event)
            self._attached = False
            self._job = None
        self._group = new_group
        await self._send(
            msg.multicast_ok_message(
                group=self._group,
                port=self._port,
                lane_id=-1,
            )
        )

    async def _handle_pause(self) -> None:
        if self._job is None:
            await self._send(msg.error_message("multicast not started"))
            return
        self._job.audio_source.pause()
        await self._send(
            msg.status_message("paused", lane_id=self._job.session.lane_id)
        )

    async def _handle_resume(self) -> None:
        if self._job is None:
            await self._send(msg.error_message("multicast not started"))
            return
        self._job.audio_source.resume()
        await self._send(
            msg.status_message("resumed", lane_id=self._job.session.lane_id)
        )

    async def _handle_finish(self) -> None:
        self._explicit_finish = True
        if self._attached:
            self._runtime.detach(self._group, self._port, self._on_job_event)
            self._attached = False
        job = await self._runtime.stop_job(self._group, self._port)
        lane_id = job.session.lane_id if job is not None else None
        self._job = None
        # Save transcript and optionally generate summary before returning ready.
        transcript_path = await self._save_transcript(lane_id) if lane_id is not None else None
        if transcript_path is not None and lane_id is not None:
            await self._send(msg.saved_message(lane_id, str(transcript_path)))
            summary = await self._maybe_generate_summary()
            if summary is not None:
                await self._send(msg.summary_message(summary, lane_id=lane_id))
        if job is None:
            await self._send(
                msg.status_message(
                    "ready_for_new_transcription",
                    message="no active job; send start to begin",
                )
            )
        else:
            await self._send(
                msg.status_message(
                    "ready_for_new_transcription",
                    lane_id=job.session.lane_id,
                    message="multicast job stopped and lane released; send start to begin",
                )
            )

    def _on_job_event(self, message: dict[str, Any]) -> None:
        # ASR/KWS callbacks may arrive from worker threads.
        if self._loop is None:
            return
        for m in self._gate.process(message):
            if m.get("type") == "transcript":
                seg = m.get("segment")
                if isinstance(seg, dict) and seg.get("text"):
                    m = dict(m, segment=dict(seg, text=self._sensitive_filter.filter(seg["text"])))
                if m.get("is_final"):
                    text = m["segment"].get("text", "")
                    if text:
                        self._transcript_texts.append(text)
            if self._gate.active and self._job is not None and not self._job.wakeup_active:
                self._job.wakeup_active = True
            self._loop.call_soon_threadsafe(self._enqueue, m)

    def _enqueue(self, message: dict[str, Any]) -> None:
        try:
            self._outgoing_queue.put_nowait(message)
        except asyncio.QueueFull:
            pass

    async def _send(self, message: dict[str, Any]) -> None:
        await self._outgoing_queue.put(message)

    def _transcript_filename(self, lane_id: int) -> str:
        ts = int(time.time() * 1000)
        return f"spacemit-asr-lane{lane_id}-{ts}.txt"

    async def _save_transcript(self, lane_id: int) -> Path | None:
        if not self._transcript_texts:
            return None
        full_text = "\n".join(self._transcript_texts).strip()
        if not full_text:
            return None
        transcript_dir = self._output_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        path = transcript_dir / self._transcript_filename(lane_id)
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
            _logger.exception("multicast summary generation failed")
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
        # Detach only — do not stop the multicast job unless finish was sent.
        if self._attached and not self._explicit_finish:
            self._runtime.detach(self._group, self._port, self._on_job_event)
            self._attached = False
            _logger.info(
                "multicast WS cleanup detached; job still running %s:%d",
                self._group,
                self._port,
            )
        if self._receive_task is not None and not self._receive_task.done():
            self._receive_task.cancel()
        if self._send_task is not None and not self._send_task.done():
            await self._outgoing_queue.put(None)
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        try:
            await self._websocket.close()
        except Exception:
            pass
