from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from multi_stream_asr.audio.multicast import MulticastAudioSource
from multi_stream_asr.session.manager import Session, SessionManager

_logger = logging.getLogger(__name__)

EventSink = Callable[[dict[str, Any]], None]


@dataclass
class MulticastJob:
    """One running multicast→ASR lane, independent of any WebSocket client."""

    key: tuple[str, int]
    session: Session
    audio_source: MulticastAudioSource
    audio_task: asyncio.Task[None]
    stats_task: asyncio.Task[None]
    listeners: set[EventSink] = field(default_factory=set)
    finished: bool = False
    wakeup_active: bool = False

    def publish(self, message: dict[str, Any]) -> None:
        dead: list[EventSink] = []
        for sink in list(self.listeners):
            try:
                sink(message)
            except Exception:
                dead.append(sink)
        for sink in dead:
            self.listeners.discard(sink)


class MulticastRuntimeRegistry:
    """Keeps multicast jobs alive across WebSocket disconnect/reconnect."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager
        self._jobs: dict[tuple[str, int], MulticastJob] = {}
        self._lock = asyncio.Lock()

    def get(self, group: str, port: int) -> MulticastJob | None:
        return self._jobs.get((group, port))

    async def start_job(
        self,
        *,
        group: str,
        port: int,
        local_ip: str,
        packet_bytes: int,
        use_full_packet: bool,
        heartbeat_interval: float,
        speexdsp_lib_path: str | None,
        listener: EventSink,
    ) -> MulticastJob:
        key = (group, port)
        async with self._lock:
            existing = self._jobs.get(key)
            if existing is not None and not existing.finished:
                existing.listeners.add(listener)
                _logger.info(
                    "multicast job reattached %s:%d lane=%d listeners=%d",
                    group,
                    port,
                    existing.session.lane_id,
                    len(existing.listeners),
                )
                return existing

            session = await self._session_manager.create_session(
                on_event=lambda message, k=key: self._fanout(k, message)
            )
            audio_source = MulticastAudioSource(
                stream_id=session.lane_id,
                group=group,
                port=port,
                local_ip=local_ip,
                packet_bytes=packet_bytes,
                use_full_packet=use_full_packet,
                heartbeat_interval=heartbeat_interval,
                speexdsp_lib_path=speexdsp_lib_path,
            )
            try:
                await audio_source.prepare()
                await self._session_manager.start_session(session)
            except Exception:
                with contextlib.suppress(Exception):
                    await audio_source.stop()
                with contextlib.suppress(Exception):
                    await self._session_manager.stop_session(session)
                raise

            job = MulticastJob(
                key=key,
                session=session,
                audio_source=audio_source,
                audio_task=asyncio.create_task(
                    self._audio_loop(key), name=f"mcast-audio-{group}-{port}"
                ),
                stats_task=asyncio.create_task(
                    self._stats_loop(key, group, port),
                    name=f"mcast-stats-{group}-{port}",
                ),
                listeners={listener},
            )
            self._jobs[key] = job
            _logger.info(
                "multicast job started %s:%d lane=%d",
                group,
                port,
                session.lane_id,
            )
            return job

    def attach(self, group: str, port: int, listener: EventSink) -> MulticastJob | None:
        job = self._jobs.get((group, port))
        if job is None or job.finished:
            return None
        job.listeners.add(listener)
        return job

    def detach(self, group: str, port: int, listener: EventSink) -> None:
        job = self._jobs.get((group, port))
        if job is None:
            return
        job.listeners.discard(listener)
        _logger.info(
            "multicast client detached %s:%d lane=%d listeners=%d (job keeps running)",
            group,
            port,
            job.session.lane_id,
            len(job.listeners),
        )

    async def stop_job(self, group: str, port: int) -> MulticastJob | None:
        key = (group, port)
        async with self._lock:
            job = self._jobs.pop(key, None)
            if job is None:
                return None
            job.finished = True
            job.listeners.clear()
            if not job.stats_task.done():
                job.stats_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await job.stats_task
            if not job.audio_task.done():
                job.audio_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await job.audio_task
            with contextlib.suppress(Exception):
                await job.audio_source.stop()
            self._session_manager.push_audio(job.session, None)
            try:
                await asyncio.wait_for(
                    job.session.finished_event.wait(),
                    timeout=self._session_manager._lifecycle_timeout_seconds,
                )
            except asyncio.TimeoutError:
                _logger.warning("multicast job finish timed out %s:%d", group, port)
            await self._session_manager.stop_session(job.session)
            _logger.info("multicast job stopped %s:%d", group, port)
            return job

    async def stop_all(self) -> None:
        keys = list(self._jobs.keys())
        for group, port in keys:
            await self.stop_job(group, port)

    def _fanout(self, key: tuple[str, int], message: dict[str, Any]) -> None:
        job = self._jobs.get(key)
        if job is None:
            return
        job.publish(message)

    async def _audio_loop(self, key: tuple[str, int]) -> None:
        job = self._jobs.get(key)
        if job is None:
            return
        frames_pushed = 0
        try:
            async for frame in job.audio_source.frames():
                if job.finished:
                    break
                if frame.is_finish:
                    self._session_manager.push_audio(job.session, None)
                    break
                self._session_manager.push_audio(job.session, frame.pcm)
                frames_pushed += 1
                if frames_pushed == 1:
                    job.publish(
                        {
                            "type": "status",
                            "status": "receiving_audio",
                            "lane_id": job.session.lane_id,
                            "message": "first multicast audio frame received",
                        }
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.exception("multicast audio loop error")
            job.publish(
                {
                    "type": "error",
                    "message": f"multicast audio error: {exc}",
                    "lane_id": job.session.lane_id,
                }
            )

    async def _stats_loop(self, key: tuple[str, int], group: str, port: int) -> None:
        last_packets = -1
        while True:
            await asyncio.sleep(3.0)
            job = self._jobs.get(key)
            if job is None or job.finished:
                return
            if not job.listeners:
                # No WS client: keep receiving, skip chatter.
                continue
            source = job.audio_source
            packets = source.packets_received
            bad = source.packets_bad_size
            frames = source.frames_emitted
            heartbeats = source.heartbeats_sent
            if packets == 0:
                joined = ",".join(source._joined_ifs) or "?"
                job.publish(
                    {
                        "type": "status",
                        "status": "waiting_for_packets",
                        "lane_id": job.session.lane_id,
                        "message": (
                            f"no UDP audio on {group}:{port} yet; "
                            f"joined=[{joined}]; "
                            f"CMD_B heartbeats_sent={heartbeats}"
                            + (f" bad_size={bad}" if bad else "")
                        ),
                    }
                )
            elif packets != last_packets:
                job.publish(
                    {
                        "type": "status",
                        "status": "receiving_audio",
                        "lane_id": job.session.lane_id,
                        "message": (
                            f"packets={packets} frames={frames} "
                            f"bad_size={bad} heartbeats={heartbeats}"
                        ),
                    }
                )
            last_packets = packets
