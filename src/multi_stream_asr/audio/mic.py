from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from multi_stream_asr.models import AudioFrame

from .device import _require_sounddevice


class MicrophoneSource:
    def __init__(
        self,
        stream_id: int,
        device: int | str | None = None,
        frame_duration_ms: int = 100,
    ) -> None:
        self.stream_id = stream_id
        self.device = device
        self.frame_duration_ms = frame_duration_ms
        self._sounddevice = _require_sounddevice(
            "sounddevice is required for microphone input"
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[np.ndarray | RuntimeError | None] | None = None
        self._stream: Any | None = None
        self._stopped = False
        self._lifecycle_lock = threading.Lock()

    async def prepare(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        blocksize = 16000 * self.frame_duration_ms // 1000
        with self._lifecycle_lock:
            if self._stopped:
                self._enqueue(None)
                return
            self._stream = self._sounddevice.InputStream(
                device=self.device,
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=self._on_audio,
            )
            self._stream.start()

    def _on_audio(self, indata: object, _frames: int, _time_info: object, status: object) -> None:
        if self._stopped:
            return
        if status:
            self._enqueue(RuntimeError(str(status)))
            return
        pcm = np.asarray(indata, dtype=np.float32)
        mono = pcm if pcm.ndim == 1 else pcm.mean(axis=1, dtype=np.float32)
        self._enqueue(np.asarray(mono, dtype=np.float32).copy())

    def _enqueue(self, item: np.ndarray | RuntimeError | None) -> None:
        loop = self._loop
        if loop is None or self._queue is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            return

    async def frames(self) -> AsyncIterator[AudioFrame]:
        if self._queue is None:
            raise RuntimeError("MicrophoneSource.prepare() must be called before frames()")
        frame_index = 0
        timestamp_ms = 0
        while True:
            item = await self._queue.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                raise item
            yield AudioFrame(
                stream_id=self.stream_id,
                frame_index=frame_index,
                pcm=item,
                timestamp_ms=timestamp_ms,
            )
            frame_index += 1
            timestamp_ms += self.frame_duration_ms
        yield AudioFrame.finish(
            stream_id=self.stream_id,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    async def stop(self) -> None:
        await asyncio.to_thread(self._stop_stream)
        self._enqueue(None)
        self._loop = None

    def _stop_stream(self) -> None:
        with self._lifecycle_lock:
            if self._stopped and self._stream is None:
                return
            self._stopped = True
            if self._stream is None:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None
