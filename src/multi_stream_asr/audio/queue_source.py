from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from multi_stream_asr.audio.base import AudioSource
from multi_stream_asr.audio.resample import SpeexResampler, SpeexResamplerError
from multi_stream_asr.models import AudioFrame


class QueueAudioSource(AudioSource):
    """AudioSource fed by a multiprocessing Queue from the parent process.

    Each item read from the queue is either:
    - ``bytes`` of int16 PCM little-endian mono audio
    - ``np.ndarray`` of float32 PCM mono audio at ``sample_rate``
    - ``None`` to signal end of stream

    Items are buffered into frames of ``frame_duration_ms`` at 16kHz.
    Non-16kHz input bytes are resampled with SpeexResampler.
    """

    def __init__(
        self,
        stream_id: int,
        audio_queue: Any,
        *,
        frame_duration_ms: int = 100,
        sample_rate: int = 16000,
        input_sample_rate: int = 16000,
        speexdsp_lib_path: str | None = None,
    ) -> None:
        self.stream_id = stream_id
        self.audio_queue = audio_queue
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.input_sample_rate = input_sample_rate
        self._samples_per_frame = sample_rate * frame_duration_ms // 1000
        self._buffer = np.asarray([], dtype=np.float32)
        self._stopped = False
        self._finished = False
        self._frame_index = 0
        self._timestamp_ms = 0
        self._resampler: SpeexResampler | None = None
        if input_sample_rate != sample_rate:
            self._resampler = SpeexResampler(
                input_rate=input_sample_rate,
                output_rate=sample_rate,
                channels=1,
                quality=5,
                skip_zeros=True,
                warmup_input_samples=0,
                lib_path=speexdsp_lib_path,
                allow_fallback=True,
            )

    async def prepare(self) -> None:
        self._stopped = False
        self._finished = False

    def _convert(self, item: bytes | np.ndarray) -> np.ndarray | None:
        if isinstance(item, bytes):
            pcm16 = np.frombuffer(item, dtype="<i2")
            if self._resampler is not None:
                pcm16 = self._resampler.resample(pcm16)
            return pcm16.astype(np.float32) / 32768.0
        if isinstance(item, np.ndarray):
            float_pcm = item.astype(np.float32)
            if float_pcm.ndim > 1:
                float_pcm = float_pcm.mean(axis=1)
            return float_pcm
        return None

    async def frames(self) -> AsyncIterator[AudioFrame]:
        while True:
            item = await asyncio.to_thread(self.audio_queue.get)
            if item is None:
                break

            float_pcm = self._convert(item)
            if float_pcm is None:
                continue

            self._buffer = np.concatenate([self._buffer, float_pcm])
            while self._buffer.size >= self._samples_per_frame:
                frame = self._buffer[: self._samples_per_frame]
                self._buffer = self._buffer[self._samples_per_frame :]
                yield AudioFrame(
                    stream_id=self.stream_id,
                    frame_index=self._frame_index,
                    pcm=frame,
                    timestamp_ms=self._timestamp_ms,
                )
                self._frame_index += 1
                self._timestamp_ms += self.frame_duration_ms

        if self._buffer.size > 0:
            yield AudioFrame(
                stream_id=self.stream_id,
                frame_index=self._frame_index,
                pcm=self._buffer,
                timestamp_ms=self._timestamp_ms,
            )
            self._frame_index += 1
            self._timestamp_ms += self.frame_duration_ms

        yield AudioFrame.finish(
            stream_id=self.stream_id,
            frame_index=self._frame_index,
            timestamp_ms=self._timestamp_ms,
        )

    async def stop(self) -> None:
        self._stopped = True
        self._finished = True
        if self._resampler is not None:
            self._resampler.close()
            self._resampler = None


class AsyncQueueAudioSource(AudioSource):
    """AudioSource fed by an asyncio.Queue from the same process.

    Each item read from the queue is either:
    - ``bytes`` of int16 PCM little-endian mono audio
    - ``np.ndarray`` of float32 PCM mono audio at ``sample_rate``
    - ``None`` to signal end of stream

    Items are buffered into frames of ``frame_duration_ms`` at 16kHz.
    Non-16kHz input bytes are resampled with SpeexResampler.
    """

    def __init__(
        self,
        stream_id: int,
        *,
        frame_duration_ms: int = 100,
        sample_rate: int = 16000,
        input_sample_rate: int = 16000,
        speexdsp_lib_path: str | None = None,
    ) -> None:
        self.stream_id = stream_id
        self.audio_queue: asyncio.Queue[bytes | np.ndarray | None] = asyncio.Queue()
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.input_sample_rate = input_sample_rate
        self._samples_per_frame = sample_rate * frame_duration_ms // 1000
        self._buffer = np.asarray([], dtype=np.float32)
        self._stopped = False
        self._finished = False
        self._frame_index = 0
        self._timestamp_ms = 0
        self._resampler: SpeexResampler | None = None
        if input_sample_rate != sample_rate:
            self._resampler = SpeexResampler(
                input_rate=input_sample_rate,
                output_rate=sample_rate,
                channels=1,
                quality=5,
                skip_zeros=True,
                warmup_input_samples=0,
                lib_path=speexdsp_lib_path,
                allow_fallback=True,
            )

    async def prepare(self) -> None:
        self._stopped = False
        self._finished = False

    def _convert(self, item: bytes | np.ndarray) -> np.ndarray | None:
        if isinstance(item, bytes):
            pcm16 = np.frombuffer(item, dtype="<i2")
            if self._resampler is not None:
                pcm16 = self._resampler.resample(pcm16)
            return pcm16.astype(np.float32) / 32768.0
        if isinstance(item, np.ndarray):
            float_pcm = item.astype(np.float32)
            if float_pcm.ndim > 1:
                float_pcm = float_pcm.mean(axis=1)
            return float_pcm
        return None

    async def frames(self) -> AsyncIterator[AudioFrame]:
        while True:
            if self._stopped:
                break
            item = await self.audio_queue.get()
            if item is None:
                break

            float_pcm = self._convert(item)
            if float_pcm is None:
                continue

            self._buffer = np.concatenate([self._buffer, float_pcm])
            while self._buffer.size >= self._samples_per_frame:
                frame = self._buffer[: self._samples_per_frame]
                self._buffer = self._buffer[self._samples_per_frame :]
                yield AudioFrame(
                    stream_id=self.stream_id,
                    frame_index=self._frame_index,
                    pcm=frame,
                    timestamp_ms=self._timestamp_ms,
                )
                self._frame_index += 1
                self._timestamp_ms += self.frame_duration_ms

        if self._buffer.size > 0:
            yield AudioFrame(
                stream_id=self.stream_id,
                frame_index=self._frame_index,
                pcm=self._buffer,
                timestamp_ms=self._timestamp_ms,
            )
            self._frame_index += 1
            self._timestamp_ms += self.frame_duration_ms

        yield AudioFrame.finish(
            stream_id=self.stream_id,
            frame_index=self._frame_index,
            timestamp_ms=self._timestamp_ms,
        )

    async def stop(self) -> None:
        self._stopped = True
        self._finished = True
        if self._resampler is not None:
            self._resampler.close()
            self._resampler = None
