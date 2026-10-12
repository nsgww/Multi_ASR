from __future__ import annotations

import asyncio
import sys
import threading
import types

import numpy as np
import pytest

from multi_stream_asr.audio.mic import MicrophoneSource


class FakeInputStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def emit(self, samples: np.ndarray, status: object = None) -> None:
        self.callback(samples, len(samples), None, status)


@pytest.fixture
def fake_sounddevice(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("sounddevice")
    streams: list[FakeInputStream] = []

    def make_stream(**kwargs) -> FakeInputStream:
        stream = FakeInputStream(**kwargs)
        streams.append(stream)
        return stream

    module.InputStream = make_stream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return streams


@pytest.mark.asyncio
async def test_microphone_source_emits_float32_100ms_frames_and_one_finish(
    fake_sounddevice,
):
    source = MicrophoneSource(stream_id=2, device=7, frame_duration_ms=100)
    await source.prepare()
    stream = fake_sounddevice[0]
    stream.emit(np.ones((1600, 1), dtype=np.float32))
    await source.stop()

    frames = [frame async for frame in source.frames()]

    assert stream.kwargs == {
        "device": 7,
        "samplerate": 16000,
        "channels": 1,
        "dtype": "float32",
        "blocksize": 1600,
        "callback": stream.callback,
    }
    assert stream.started is True
    assert stream.stopped is True
    assert stream.closed is True
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert [frame.timestamp_ms for frame in frames] == [0, 100]
    assert frames[0].pcm.dtype == np.float32
    assert frames[0].pcm.shape == (1600,)
    assert [frame.is_finish for frame in frames] == [False, True]


@pytest.mark.asyncio
async def test_microphone_source_converts_multichannel_callback_to_mono(fake_sounddevice):
    source = MicrophoneSource(stream_id=0, device=None, frame_duration_ms=100)
    await source.prepare()
    fake_sounddevice[0].emit(
        np.column_stack((np.zeros(1600, dtype=np.float32), np.ones(1600, dtype=np.float32)))
    )
    await source.stop()

    frames = [frame async for frame in source.frames()]

    assert np.allclose(frames[0].pcm, 0.5)


@pytest.mark.asyncio
async def test_microphone_source_raises_callback_status(fake_sounddevice):
    source = MicrophoneSource(stream_id=0)
    await source.prepare()
    fake_sounddevice[0].emit(np.zeros((1600, 1), dtype=np.float32), status="overflow")

    with pytest.raises(RuntimeError, match="overflow"):
        async for _ in source.frames():
            pass

    await source.stop()


@pytest.mark.asyncio
async def test_microphone_stop_ignores_a_closed_producer_loop(fake_sounddevice):
    source = MicrophoneSource(stream_id=0)
    await source.prepare()

    class ClosedLoop:
        def is_closed(self) -> bool:
            return True

        def call_soon_threadsafe(self, *args) -> None:
            raise AssertionError("closed loop must not receive a callback")

    source._loop = ClosedLoop()  # type: ignore[assignment]

    await source.stop()

    assert fake_sounddevice[0].stopped is True
    assert fake_sounddevice[0].closed is True


@pytest.mark.asyncio
async def test_microphone_stop_waits_for_stream_start_before_closing(monkeypatch):
    module = types.ModuleType("sounddevice")
    start_entered = threading.Event()
    release_start = threading.Event()
    events: list[str] = []

    class BlockingStartStream(FakeInputStream):
        def start(self) -> None:
            events.append("start_entered")
            start_entered.set()
            release_start.wait()
            events.append("start_returned")
            super().start()

        def stop(self) -> None:
            events.append("stop")
            super().stop()

        def close(self) -> None:
            events.append("close")
            super().close()

    stream = BlockingStartStream(callback=lambda *_args: None)
    module.InputStream = lambda **_kwargs: stream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    source = MicrophoneSource(stream_id=0)
    prepare_task = asyncio.create_task(
        asyncio.to_thread(lambda: asyncio.run(source.prepare()))
    )
    await asyncio.to_thread(start_entered.wait)

    stop_task = asyncio.create_task(source.stop())
    await asyncio.sleep(0.01)
    release_start.set()
    await asyncio.gather(prepare_task, stop_task)

    assert events == ["start_entered", "start_returned", "stop", "close"]


@pytest.mark.asyncio
async def test_microphone_prepare_after_early_stop_does_not_open_stream(fake_sounddevice):
    source = MicrophoneSource(stream_id=0)

    await source.stop()
    await source.prepare()

    assert fake_sounddevice == []
