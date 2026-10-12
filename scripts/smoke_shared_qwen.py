from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import numpy as np

from multi_stream_asr.asr.offline.qwen3asr import Qwen3AsrClient
from multi_stream_asr.asr.offline.shared_server import (
    SHARED_CONFIG_DIR,
    SharedQwenServerManager,
    build_launch_command,
)
from multi_stream_asr.config import AppConfig


def load_wav(path: Path, seconds: float) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("audio must be mono")
        if wav_file.getframerate() != 16000:
            raise ValueError("audio sample rate must be 16000 Hz")
        if wav_file.getsampwidth() != 2:
            raise ValueError("audio sample width must be 16-bit")
        frame_count = min(wav_file.getnframes(), int(seconds * 16000))
        pcm16 = np.frombuffer(wav_file.readframes(frame_count), dtype="<i2")
    if pcm16.size == 0:
        raise ValueError("audio contains no samples")
    return pcm16.astype(np.float32) / 32768.0


async def run_smoke(audio_path: Path, seconds: float) -> dict[str, object]:
    config = AppConfig()
    pcm = load_wav(audio_path, seconds)
    manager = SharedQwenServerManager(config)
    client = Qwen3AsrClient(shared=config.shared_qwen)
    command = build_launch_command(config, SHARED_CONFIG_DIR)
    print("launch_command=" + json.dumps(command, ensure_ascii=False), flush=True)
    try:
        await asyncio.to_thread(manager.start)
        result = await client.recognize(pcm=pcm, prompt="")
        payload = {
            "text": result.text,
            "audio_seconds": result.audio_seconds,
            "rtf": result.rtf,
            "timings": result.timings,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload
    finally:
        await client.close()
        await asyncio.to_thread(manager.stop)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one real request against the managed shared Qwen3-ASR service."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    asyncio.run(run_smoke(args.audio.resolve(), args.seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
