from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multi_stream_asr.audio.multicast import MulticastAudioSource
from multi_stream_asr.session.process_pool import ProcessPoolSessionManager, Session

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

    def __init__(
        self,
        session_manager: ProcessPoolSessionManager,
        *,
        max_jobs: int = 8,
        persist_path: Path | None = None,
    ) -> None:
        self._session_manager = session_manager
        self._max_jobs = max_jobs
        self._persist_path = persist_path
        self._jobs: dict[tuple[str, int], MulticastJob] = {}
        self._lock = asyncio.Lock()

    def set_session_manager(self, manager: ProcessPoolSessionManager) -> None:
        """通道热改（/api/models/load）后替换底层 session 池。

        仅允许在无运行中任务时调用（load 接口已用 409 保证），否则旧任务
        的 session 会悬在已被丢弃的旧池上。
        """
        self._session_manager = manager

    @property
    def max_jobs(self) -> int:
        return self._max_jobs

    def get(self, group: str, port: int) -> MulticastJob | None:
        return self._jobs.get((group, port))

    def list_jobs(self) -> list[dict[str, Any]]:
        """当前在跑的组播任务清单（REST 查询用）。"""
        return [
            {
                "group": group,
                "port": port,
                "lane_id": job.session.lane_id,
                "listeners": len(job.listeners),
                "finished": job.finished,
                "packets_received": job.audio_source.packets_received,
                "frames_emitted": job.audio_source.frames_emitted,
                "heartbeats_sent": job.audio_source.heartbeats_sent,
            }
            for (group, port), job in sorted(self._jobs.items())
        ]

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
        listener: EventSink | None = None,
    ) -> MulticastJob:
        key = (group, port)
        async with self._lock:
            existing = self._jobs.get(key)
            if existing is not None and not existing.finished:
                if listener is not None:
                    existing.listeners.add(listener)
                _logger.info(
                    "multicast job reattached %s:%d lane=%d listeners=%d",
                    group,
                    port,
                    existing.session.lane_id,
                    len(existing.listeners),
                )
                return existing

            if len(self._jobs) >= self._max_jobs:
                raise RuntimeError(
                    f"组播路数已达上限 {self._max_jobs}，请先停止一路再新增"
                )

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
                listeners={listener} if listener is not None else set(),
            )
            self._jobs[key] = job
            self._save_persisted()
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

    async def stop_job(
        self, group: str, port: int, *, update_persist: bool = True
    ) -> MulticastJob | None:
        key = (group, port)
        async with self._lock:
            job = self._jobs.get(key)
            if job is None:
                return None
            job.finished = True
            # 注意：收尾期间不要提前摘除 job、也不要清 listeners。
            # pipeline 在收尾时还会吐最后一句 FINAL 和 speaker_remap，而
            # ``_fanout`` 靠 ``_jobs`` 查表投递，提前摘掉这些事件会被丢掉。
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
            # 收尾事件投递完毕，这时才摘除并断掉监听
            if self._jobs.get(key) is job:
                self._jobs.pop(key, None)
            job.listeners.clear()
            if update_persist:
                self._save_persisted()
            _logger.info("multicast job stopped %s:%d", group, port)
            return job

    async def stop_all(self) -> None:
        # 服务关停不清持久化清单：重启后要按这份清单自动拉起
        keys = list(self._jobs.keys())
        for group, port in keys:
            await self.stop_job(group, port, update_persist=False)

    async def modify_job(
        self,
        *,
        old_group: str,
        old_port: int,
        new_group: str,
        new_port: int | None = None,
        speexdsp_lib_path: str | None = None,
    ) -> MulticastJob | None:
        """修改已有任务的组播地址：沿用旧任务的接收参数，停旧开新。

        返回新任务；旧任务不存在返回 None。注意旧任务上的 WS listener
        不会迁移，客户端需用新地址重新「收听」。
        """
        old = self._jobs.get((old_group, old_port))
        if old is None or old.finished:
            return None
        params: dict[str, Any] = {
            "local_ip": old.audio_source.local_ip,
            "packet_bytes": old.audio_source.packet_bytes,
            "use_full_packet": old.audio_source.use_full_packet,
            "heartbeat_interval": old.audio_source.heartbeat_interval,
            "speexdsp_lib_path": speexdsp_lib_path,
            "listener": None,
        }
        await self.stop_job(old_group, old_port)
        return await self.start_job(
            group=new_group,
            port=new_port if new_port is not None else old_port,
            **params,
        )

    def _save_persisted(self) -> None:
        """把当前在跑的组播地址落盘，供服务重启后自动恢复。"""
        if self._persist_path is None:
            return
        data = [
            {"group": group, "port": port}
            for (group, port), job in sorted(self._jobs.items())
            if not job.finished
        ]
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            _logger.exception("multicast persist save failed: %s", self._persist_path)

    async def restore_jobs(
        self,
        *,
        local_ip: str,
        packet_bytes: int,
        use_full_packet: bool,
        heartbeat_interval: float,
        speexdsp_lib_path: str | None,
    ) -> list[tuple[str, int]]:
        """服务启动时按持久化清单自动拉起上次的组播任务。"""
        if self._persist_path is None or not self._persist_path.exists():
            return []
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except Exception:
            _logger.exception("multicast persist load failed: %s", self._persist_path)
            return []
        restored: list[tuple[str, int]] = []
        for item in data if isinstance(data, list) else []:
            try:
                group = str(item["group"]).strip()
                port = int(item["port"])
                addr = ipaddress.IPv4Address(group)
                if not addr.is_multicast or not (1 <= port <= 65535):
                    raise ValueError("not a valid multicast target")
            except Exception:
                _logger.warning("multicast persist skip invalid entry: %r", item)
                continue
            try:
                await self.start_job(
                    group=group,
                    port=port,
                    local_ip=local_ip,
                    packet_bytes=packet_bytes,
                    use_full_packet=use_full_packet,
                    heartbeat_interval=heartbeat_interval,
                    speexdsp_lib_path=speexdsp_lib_path,
                    listener=None,
                )
                restored.append((group, port))
            except Exception:
                _logger.exception("multicast restore failed %s:%d", group, port)
        if restored:
            _logger.info("multicast restored %d job(s): %s", len(restored), restored)
        return restored

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
