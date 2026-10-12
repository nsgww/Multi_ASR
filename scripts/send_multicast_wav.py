#!/usr/bin/env python3
"""Send a WAV file as UDP multicast PCM16 packets for /ws/multicast testing.

Protocol expected by multi-stream-asr MulticastAudioSource:
- 48 kHz mono PCM16 little-endian
- default packet_bytes=1280, use_full_packet=false → first 640 bytes used as audio
  (640 bytes = 320 samples @ 48kHz ≈ 6.67ms)

Example:
  python scripts/send_multicast_wav.py sample.wav --group 224.2.1.209 --port 6020
"""

from __future__ import annotations

import argparse
import socket
import time
import wave
from pathlib import Path


def _read_wav_mono_pcm16(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise SystemExit(f"need 16-bit PCM WAV, got sample_width={width}")
    if channels == 1:
        return raw, rate
    # downmix stereo int16 LE
    import array

    samples = array.array("h")
    samples.frombytes(raw)
    mono = array.array("h")
    for i in range(0, len(samples), channels):
        mono.append(int(sum(samples[i : i + channels]) / channels))
    return mono.tobytes(), rate


def _resample_linear(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm
    import array

    src = array.array("h")
    src.frombytes(pcm)
    if not src:
        return b""
    ratio = src_rate / dst_rate
    out_len = int(len(src) / ratio)
    out = array.array("h")
    for i in range(out_len):
        pos = i * ratio
        i0 = int(pos)
        i1 = min(i0 + 1, len(src) - 1)
        frac = pos - i0
        out.append(int(src[i0] * (1 - frac) + src[i1] * frac))
    return out.tobytes()


def main() -> int:
    parser = argparse.ArgumentParser(description="UDP multicast PCM sender for ASR test")
    parser.add_argument("wav", type=Path, help="input WAV (any common rate; will resample to 48k)")
    parser.add_argument("--group", default="224.2.1.209")
    parser.add_argument("--port", type=int, default=6020)
    parser.add_argument(
        "--audio-bytes",
        type=int,
        default=640,
        help="audio bytes per packet (default 640 = what server reads when use_full_packet=false)",
    )
    parser.add_argument(
        "--packet-bytes",
        type=int,
        default=1280,
        help="UDP datagram size; audio placed at front, rest zero-padded",
    )
    parser.add_argument("--loop", action="store_true", help="loop the file forever")
    parser.add_argument(
        "--ttl",
        type=int,
        default=1,
        help="multicast TTL (1 = same subnet)",
    )
    args = parser.parse_args()

    pcm, rate = _read_wav_mono_pcm16(args.wav)
    pcm = _resample_linear(pcm, rate, 48000)
    if not pcm:
        raise SystemExit("empty audio after resample")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, args.ttl)

    audio_bytes = args.audio_bytes
    packet_bytes = max(args.packet_bytes, audio_bytes)
    # 48kHz * 2 bytes/sample
    seconds_per_packet = audio_bytes / (48000 * 2)
    dest = (args.group, args.port)

    print(
        f"sending {args.wav} -> {args.group}:{args.port} "
        f"audio_bytes={audio_bytes} packet_bytes={packet_bytes} "
        f"interval={seconds_per_packet * 1000:.2f}ms"
    )

    try:
        while True:
            offset = 0
            t0 = time.perf_counter()
            packet_index = 0
            while offset < len(pcm):
                chunk = pcm[offset : offset + audio_bytes]
                offset += audio_bytes
                if len(chunk) < audio_bytes:
                    chunk = chunk + b"\x00" * (audio_bytes - len(chunk))
                packet = chunk + b"\x00" * (packet_bytes - len(chunk))
                sock.sendto(packet, dest)
                packet_index += 1
                target = t0 + packet_index * seconds_per_packet
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            print(f"file sent once ({packet_index} packets)")
            if not args.loop:
                break
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
