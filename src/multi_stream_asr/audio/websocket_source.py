from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator

import numpy as np

from multi_stream_asr.audio.base import AudioSource
from multi_stream_asr.models import AudioFrame


class WebSocketAudioSource(AudioSource):
    """AudioSource fed by binary WebSocket messages.

    Supports PCM16 little-endian mono audio. The sample rate is fixed at 16kHz
    and frames are emitted every ``frame_duration_ms`` (default 100ms).
    """

    def __init__(
        self,
        stream_id: int,
        *,
        frame_duration_ms: int = 100,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> None:
        self.stream_id = stream_id
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self._samples_per_frame = (
            sample_rate * frame_duration_ms * channels // 1000
        )
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._buffer = np.asarray([], dtype=np.float32)
        self._stopped = False
        self._finished = False
        self._frame_index = 0
        self._timestamp_ms = 0

    async def prepare(self) -> None:
        self._stopped = False
        self._finished = False

    def push_bytes(self, data: bytes) -> None:
        """Called by the WebSocket handler for each binary message."""
        if self._finished or self._stopped:
            return
        if self.sample_width == 2:
            pcm = np.frombuffer(data, dtype="<i2")
        elif self.sample_width == 4:
            pcm = np.frombuffer(data, dtype="<i4")
        else:
            raise ValueError(f"unsupported sample width: {self.sample_width}")
        if self.channels > 1:
            pcm = pcm.reshape(-1, self.channels).mean(axis=1)
        float_pcm = pcm.astype(np.float32) / (2 ** (self.sample_width * 8 - 1))
        self._buffer = np.concatenate([self._buffer, float_pcm])
        while self._buffer.size >= self._samples_per_frame:
            frame = self._buffer[: self._samples_per_frame]
            self._buffer = self._buffer[self._samples_per_frame :]
            self._queue.put_nowait(frame)

    def finish(self) -> None:
        """Signal end of audio stream."""
        self._finished = True
        self._queue.put_nowait(None)

    async def frames(self) -> AsyncIterator[AudioFrame]:
        if self._finished and self._queue.empty() and self._buffer.size == 0:
            yield AudioFrame.finish(
                stream_id=self.stream_id,
                frame_index=self._frame_index,
                timestamp_ms=self._timestamp_ms,
            )
            return

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

        # Emit any trailing samples as a final partial frame
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
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
