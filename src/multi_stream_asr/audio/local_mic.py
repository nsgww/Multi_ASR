from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Any

_logger = logging.getLogger(__name__)


def find_first_input_device() -> int | None:
    """扫描并返回第一个可用的音频输入设备索引，没有则返回 None。"""
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                _logger.info("[local_mic] auto-selected input device %s: %s", i, info.get("name"))
                return i
        return None
    finally:
        pa.terminate()


class LocalMicSource:
    """Board local microphone source using pyaudio.

    Reads int16 mono PCM from a PortAudio device in a background thread and
    forwards raw bytes (or ``None`` for end-of-stream) through an asyncio queue.
    """

    def __init__(
        self,
        *,
        device: int | str | None = None,
        rate: int = 16000,
        channels: int = 1,
        chunk: int = 512,
        output_queue: asyncio.Queue[bytes | None] | None = None,
    ) -> None:
        self.device = device
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self._queue = output_queue or asyncio.Queue()
        self._pyaudio: Any | None = None
        self._stream: Any | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._started = False

    @property
    def queue(self) -> asyncio.Queue[bytes | None]:
        return self._queue

    def start(self) -> None:
        if self._started:
            return
        import pyaudio

        loop = asyncio.get_running_loop()
        self._pyaudio = pyaudio.PyAudio()
        self._stopped = False
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device,
            frames_per_buffer=self.chunk,
        )
        self._thread = threading.Thread(
            target=self._read_loop, args=(loop,), daemon=True
        )
        self._thread.start()
        self._started = True
        _logger.info(
            "[local_mic] started: device=%s rate=%d channels=%d chunk=%d",
            self.device,
            self.rate,
            self.channels,
            self.chunk,
        )

    def _read_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        while not self._stopped:
            try:
                data = self._stream.read(self.chunk, exception_on_overflow=False)
            except Exception as exc:
                _logger.exception("[local_mic] read error: %s", exc)
                with contextlib.suppress(Exception):
                    loop.call_soon_threadsafe(self._queue.put_nowait, None)
                break
            if not data:
                continue
            try:
                loop.call_soon_threadsafe(self._queue.put_nowait, data)
            except RuntimeError:
                # event loop closed
                break

    async def stop(self) -> None:
        if not self._started:
            return
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            try:
                await asyncio.to_thread(self._stream.stop_stream)
                await asyncio.to_thread(self._stream.close)
            except Exception:
                pass
            self._stream = None
        if self._pyaudio is not None:
            try:
                await asyncio.to_thread(self._pyaudio.terminate)
            except Exception:
                pass
            self._pyaudio = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._started = False
        _logger.info("[local_mic] stopped")
