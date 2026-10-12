from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

from multi_stream_asr.models import AudioFrame

SAMPLE_RATE_HZ = 16000
PCM_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int


def validate_wav_16k_mono_pcm(path: Path) -> WavInfo:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as wav_file:
        info = WavInfo(
            path=wav_path,
            sample_rate=wav_file.getframerate(),
            channels=wav_file.getnchannels(),
            sample_width=wav_file.getsampwidth(),
            frame_count=wav_file.getnframes(),
        )

    if (
        info.sample_rate != SAMPLE_RATE_HZ
        or info.channels != 1
        or info.sample_width != PCM_SAMPLE_WIDTH_BYTES
    ):
        raise ValueError(f"{wav_path} must be a 16 kHz mono PCM WAV")

    return info


def select_input_files(input_dir: Path, lanes: int) -> list[Path]:
    wav_files = sorted(path for path in Path(input_dir).iterdir() if path.is_file() and path.suffix.lower() == ".wav")
    if not wav_files:
        raise ValueError(f"no WAV files found in {input_dir}")

    return [wav_files[index % len(wav_files)] for index in range(lanes)]


class LocalWavSource:
    def __init__(
        self,
        stream_id: int,
        path: Path,
        frame_duration_ms: int = 100,
        *,
        realtime: bool = False,
        loop_audio: bool = False,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.stream_id = stream_id
        self.path = Path(path)
        self.frame_duration_ms = frame_duration_ms
        self.realtime = realtime
        self.loop_audio = loop_audio
        self._sleep = sleep
        self._samples = np.asarray([], dtype=np.float32)
        self._stopped = False
        self._prepared = False

    async def prepare(self) -> None:
        validate_wav_16k_mono_pcm(self.path)
        with wave.open(str(self.path), "rb") as wav_file:
            raw = wav_file.readframes(wav_file.getnframes())
        self._samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if self.loop_audio and self._samples.size == 0:
            raise ValueError(f"cannot loop an empty WAV: {self.path}")
        self._prepared = True
        self._stopped = False

    async def frames(self) -> AsyncIterator[AudioFrame]:
        if not self._prepared:
            raise RuntimeError("LocalWavSource.prepare() must be called before frames()")

        samples_per_frame = SAMPLE_RATE_HZ * self.frame_duration_ms // 1000
        frame_index = 0
        timestamp_ms = 0

        while not self._stopped:
            for start in range(0, self._samples.size, samples_per_frame):
                if self._stopped:
                    break
                stop = start + samples_per_frame
                yield AudioFrame(
                    stream_id=self.stream_id,
                    frame_index=frame_index,
                    pcm=self._samples[start:stop],
                    timestamp_ms=timestamp_ms,
                )
                frame_index += 1
                timestamp_ms += self.frame_duration_ms
                if self.realtime and not self._stopped:
                    await self._sleep(self.frame_duration_ms / 1000.0)
            if not self.loop_audio:
                break

        yield AudioFrame.finish(
            stream_id=self.stream_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    async def stop(self) -> None:
        self._stopped = True
