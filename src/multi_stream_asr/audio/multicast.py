from __future__ import annotations

import asyncio
import logging
import socket
import struct
import subprocess
import time
from collections.abc import AsyncIterator
import numpy as np

from multi_stream_asr.audio.base import AudioSource
from multi_stream_asr.audio.resample import SpeexResampler, SpeexResamplerError
from multi_stream_asr.models import AudioFrame

_logger = logging.getLogger(__name__)

DEFAULT_MULTICAST_GROUP = "224.2.1.168"
DEFAULT_MULTICAST_PORT = 6020
MULTICAST_INPUT_RATE = 48000
MULTICAST_OUTPUT_RATE = 16000
MULTICAST_PACKET_BYTES = 1280
MULTICAST_USE_FULL_PACKET = False
DEFAULT_HEARTBEAT_INTERVAL = 5.0
HEARTBEAT_PAYLOAD = b"CMD_B"


def _iface_index_map() -> dict[str, int]:
    try:
        return {name: idx for idx, name in socket.if_nameindex()}
    except OSError:
        return {}


def _list_iface_ipv4() -> list[tuple[str, str, int]]:
    """Return [(ifname, ipv4, ifindex), ...] for global IPv4 NICs."""
    index_by_name = _iface_index_map()
    result: list[tuple[str, str, int]] = []
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        out = ""

    for line in out.splitlines():
        # "3: wlP4p1s0    inet 192.168.1.178/24 ..."
        parts = line.split()
        if len(parts) < 4 or "inet" not in parts:
            continue
        ifname = parts[1].rstrip(":")
        inet_idx = parts.index("inet")
        if inet_idx + 1 >= len(parts):
            continue
        ip = parts[inet_idx + 1].split("/", 1)[0]
        if not ip or ip.startswith("127."):
            continue
        ifindex = index_by_name.get(ifname, 0)
        result.append((ifname, ip, ifindex))

    if result:
        return result

    # Fallback without `ip`
    for ifname, ifindex in ((n, i) for i, n in _iface_index_map().items()):
        if ifname.startswith("lo"):
            continue
        result.append((ifname, "0.0.0.0", ifindex))
    return result


class MulticastAudioSource(AudioSource):
    """UDP multicast → Speex 48k→16k → 100ms frames.

    Join/receive aligned with sm-sdk:
    - bind multicast group address
    - IGMP join per NIC via ``ip_mreqn`` (ifindex), so WiFi traffic is not missed
    - blocking ``recvfrom`` + ``run_in_executor`` (not asyncio sock_recvfrom)
    """

    def __init__(
        self,
        stream_id: int,
        *,
        group: str = DEFAULT_MULTICAST_GROUP,
        port: int = DEFAULT_MULTICAST_PORT,
        local_ip: str = "0.0.0.0",
        packet_bytes: int = MULTICAST_PACKET_BYTES,
        use_full_packet: bool = MULTICAST_USE_FULL_PACKET,
        frame_duration_ms: int = 100,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        speexdsp_lib_path: str | None = None,
    ) -> None:
        self.stream_id = stream_id
        self.group = group
        self.port = port
        self.local_ip = local_ip
        self.packet_bytes = packet_bytes
        self.use_full_packet = use_full_packet
        self.heartbeat_interval = float(heartbeat_interval)
        self.frame_duration_ms = frame_duration_ms
        self._samples_per_frame = MULTICAST_OUTPUT_RATE * frame_duration_ms // 1000
        self._resampler = SpeexResampler(
            input_rate=MULTICAST_INPUT_RATE,
            output_rate=MULTICAST_OUTPUT_RATE,
            channels=1,
            quality=5,
            skip_zeros=True,
            warmup_input_samples=1536,
            lib_path=speexdsp_lib_path,
            allow_fallback=False,
        )
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=100)
        self._buffer = np.asarray([], dtype=np.float32)
        self._socket: socket.socket | None = None
        self._mreqs: list[bytes] = []
        self._joined_ifs: list[str] = []
        self._receive_task: asyncio.Task[None] | None = None
        self._paused = False
        self._stopped = False
        self._frame_index = 0
        self._timestamp_ms = 0
        self.packets_received = 0
        self.packets_bad_size = 0
        self.frames_emitted = 0
        self.heartbeats_sent = 0
        self._last_heartbeat = 0.0

    async def prepare(self) -> None:
        self._stopped = False
        self._paused = False
        await asyncio.to_thread(self._join_group_sync, self.group, self.port)
        self._receive_task = asyncio.create_task(self._receive_loop())

    def _join_targets(self) -> list[tuple[str, str, int]]:
        ifaces = _list_iface_ipv4()
        if self.local_ip and self.local_ip not in ("", "0.0.0.0"):
            matched = [t for t in ifaces if t[1] == self.local_ip]
            if matched:
                return matched
            # User forced an IP that we couldn't map to ifname; still try address join.
            return [("?", self.local_ip, 0)]
        if ifaces:
            return ifaces
        return [("any", "0.0.0.0", 0)]

    def _join_group_sync(self, group: str, port: int) -> None:
        if self._socket is not None:
            self._drop_memberships()
            self._socket.close()
            self._socket = None

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass

        # sm-sdk: bind multicast group address.
        try:
            sock.bind((group, port))
        except OSError as exc:
            _logger.warning("bind(%s,%d) failed (%s); fallback 0.0.0.0", group, port, exc)
            sock.bind(("0.0.0.0", port))

        # Ensure we can receive multicast even when membership bookkeeping is odd.
        if hasattr(socket, "IP_MULTICAST_ALL"):
            try:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_ALL, 1)
            except OSError:
                pass

        joined: list[str] = []
        mreqs: list[bytes] = []
        last_err: OSError | None = None

        for ifname, iface_ip, ifindex in self._join_targets():
            # Prefer Linux ip_mreqn(ifindex); also try classic ip_mreq(iface IP).
            candidates: list[bytes] = []
            if ifindex > 0:
                candidates.append(
                    struct.pack(
                        "4s4si",
                        socket.inet_aton(group),
                        socket.inet_aton("0.0.0.0"),
                        ifindex,
                    )
                )
            if iface_ip not in ("", "0.0.0.0"):
                candidates.append(
                    struct.pack(
                        "4s4s",
                        socket.inet_aton(group),
                        socket.inet_aton(iface_ip),
                    )
                )
            if not candidates:
                candidates.append(
                    struct.pack(
                        "4sl",
                        socket.inet_aton(group),
                        socket.INADDR_ANY,
                    )
                )

            joined_this = False
            for mreq in candidates:
                try:
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                    mreqs.append(mreq)
                    joined_this = True
                except OSError as exc:
                    last_err = exc
            if joined_this:
                label = f"{ifname}/{iface_ip}/if{ifindex}"
                joined.append(label)
            else:
                _logger.warning(
                    "multicast join failed on %s/%s/if%d for %s:%d: %s",
                    ifname,
                    iface_ip,
                    ifindex,
                    group,
                    port,
                    last_err,
                )

        if not mreqs:
            sock.close()
            raise RuntimeError(
                f"failed to join multicast group {group}:{port}"
                + (f": {last_err}" if last_err else "")
            )

        # sm-sdk style: blocking socket with short timeout + executor recv.
        sock.settimeout(1.0)
        self._socket = sock
        self._mreqs = mreqs
        self._joined_ifs = joined
        _logger.info(
            "joined multicast %s:%d memberships=%s",
            group,
            port,
            ",".join(joined),
        )

    def _drop_memberships(self) -> None:
        if self._socket is None:
            self._mreqs = []
            return
        for mreq in self._mreqs:
            try:
                self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
            except OSError:
                pass
        self._mreqs = []
        self._joined_ifs = []

    def _maybe_send_heartbeat(self) -> None:
        if self.heartbeat_interval <= 0 or self._socket is None:
            return
        now = time.time()
        if now - self._last_heartbeat < self.heartbeat_interval:
            return
        try:
            self._socket.sendto(HEARTBEAT_PAYLOAD, (self.group, self.port))
            self._last_heartbeat = now
            self.heartbeats_sent += 1
            if self.heartbeats_sent == 1 or self.heartbeats_sent % 12 == 0:
                _logger.info(
                    "multicast heartbeat CMD_B sent to %s:%d (count=%d)",
                    self.group,
                    self.port,
                    self.heartbeats_sent,
                )
        except OSError as exc:
            _logger.warning("multicast heartbeat send failed: %s", exc)

    def _recv_one(self) -> bytes | None:
        """Blocking recv used inside a worker thread (sm-sdk pattern)."""
        assert self._socket is not None
        try:
            data, _addr = self._socket.recvfrom(self.packet_bytes * 2)
            return data
        except socket.timeout:
            return None
        except OSError:
            if not self._stopped:
                _logger.exception("multicast recvfrom error")
            return None

    async def _receive_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stopped:
            if self._paused or self._socket is None:
                await asyncio.sleep(0.05)
                continue
            self._maybe_send_heartbeat()
            data = await loop.run_in_executor(None, self._recv_one)
            if data is None:
                continue
            if self._paused:
                continue

            if len(data) != self.packet_bytes:
                self.packets_bad_size += 1
                if self.packets_bad_size in (1, 10, 100) or self.packets_bad_size % 500 == 0:
                    _logger.warning(
                        "multicast bad packet size=%d expected=%d (count=%d) head=%r",
                        len(data),
                        self.packet_bytes,
                        self.packets_bad_size,
                        data[:16],
                    )
                continue

            self.packets_received += 1
            if self.packets_received == 1 or self.packets_received % 500 == 0:
                _logger.info(
                    "multicast packets_received=%d frames_emitted=%d group=%s:%d",
                    self.packets_received,
                    self.frames_emitted,
                    self.group,
                    self.port,
                )

            audio_bytes = data if self.use_full_packet else data[:640]
            if not audio_bytes:
                continue
            pcm16 = np.frombuffer(audio_bytes, dtype="<i2")
            try:
                resampled = self._resampler.resample(pcm16)
            except SpeexResamplerError:
                _logger.exception("resample failed")
                continue
            float_pcm = resampled.astype(np.float32) / 32768.0
            self._buffer = np.concatenate([self._buffer, float_pcm])
            while self._buffer.size >= self._samples_per_frame:
                frame = self._buffer[: self._samples_per_frame]
                self._buffer = self._buffer[self._samples_per_frame :]
                self.frames_emitted += 1
                try:
                    self._queue.put_nowait(frame)
                except asyncio.QueueFull:
                    _logger.warning("multicast audio queue full; dropping frame")

    async def frames(self) -> AsyncIterator[AudioFrame]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            frame = AudioFrame(
                stream_id=self.stream_id,
                frame_index=self._frame_index,
                pcm=item,
                timestamp_ms=self._timestamp_ms,
            )
            self._frame_index += 1
            self._timestamp_ms += self.frame_duration_ms
            yield frame

        yield AudioFrame.finish(
            stream_id=self.stream_id,
            frame_index=self._frame_index,
            timestamp_ms=self._timestamp_ms,
        )

    async def stop(self) -> None:
        self._stopped = True
        if self._receive_task is not None and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._socket is not None:
            self._drop_memberships()
            self._socket.close()
            self._socket = None
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        self._resampler.close()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    async def set_group(self, group: str) -> None:
        self.group = group
        await asyncio.to_thread(self._join_group_sync, self.group, self.port)
