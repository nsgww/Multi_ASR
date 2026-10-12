from __future__ import annotations

import builtins
import types
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from multi_stream_asr.audio.local_file import (
    LocalWavSource,
    select_input_files,
    validate_wav_16k_mono_pcm,
)
from multi_stream_asr.audio.device import (
    list_microphone_devices,
    resolve_microphone_device,
    validate_microphone_device,
)
from multi_stream_asr.audio.mic import MicrophoneSource


def write_wav(
    path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
    samples: np.ndarray | None = None,
) -> None:
    pcm = samples
    if pcm is None:
        pcm = np.arange(sample_rate // 5, dtype=np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.astype(np.int16).tobytes())


def test_validate_wav_accepts_16k_mono_pcm(tmp_path: Path):
    wav_path = tmp_path / "valid.wav"
    write_wav(wav_path)

    info = validate_wav_16k_mono_pcm(wav_path)

    assert info.path == wav_path
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.sample_width == 2
    assert info.frame_count == 3200


def test_validate_wav_rejects_wrong_sample_rate(tmp_path: Path):
    wav_path = tmp_path / "bad_rate.wav"
    write_wav(wav_path, sample_rate=8000)

    with pytest.raises(ValueError, match="16 kHz mono PCM WAV"):
        validate_wav_16k_mono_pcm(wav_path)


def test_validate_wav_rejects_stereo(tmp_path: Path):
    wav_path = tmp_path / "stereo.wav"
    write_wav(wav_path, channels=2)

    with pytest.raises(ValueError, match="16 kHz mono PCM WAV"):
        validate_wav_16k_mono_pcm(wav_path)


def test_validate_wav_rejects_unsupported_sample_width(tmp_path: Path):
    wav_path = tmp_path / "bad_width.wav"
    write_wav(wav_path, sample_width=1)

    with pytest.raises(ValueError, match="16 kHz mono PCM WAV"):
        validate_wav_16k_mono_pcm(wav_path)


def test_select_input_files_sorts_and_cycles(tmp_path: Path):
    for name in ("b.wav", "a.wav", "c.wav"):
        write_wav(tmp_path / name)

    selected = select_input_files(tmp_path, lanes=5)

    assert [path.name for path in selected] == ["a.wav", "b.wav", "c.wav", "a.wav", "b.wav"]


def test_list_microphone_devices_raises_when_sounddevice_missing(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sounddevice":
            raise ImportError("missing sounddevice")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="sounddevice is required to list microphone devices"):
        list_microphone_devices()


def test_microphone_source_raises_source_specific_error_when_sounddevice_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sounddevice":
            raise ImportError("missing sounddevice")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="sounddevice is required for microphone input"):
        MicrophoneSource(stream_id=0)


def test_list_microphone_devices_filters_output_only_devices(monkeypatch: pytest.MonkeyPatch):
    fake_sounddevice = types.ModuleType("sounddevice")
    calls: list[str] = []
    fake_sounddevice._terminate = lambda: calls.append("terminate")  # type: ignore[attr-defined]
    fake_sounddevice._initialize = lambda: calls.append("initialize")  # type: ignore[attr-defined]
    fake_sounddevice.query_hostapis = lambda _index: {"name": "PulseAudio"}  # type: ignore[attr-defined]
    fake_sounddevice.query_devices = lambda: [  # type: ignore[attr-defined]
        {
            "name": "output-only",
            "max_input_channels": 0,
            "default_samplerate": 48000,
            "hostapi": 0,
        },
        {
            "name": "mic",
            "max_input_channels": 2,
            "default_samplerate": 16000,
            "hostapi": 0,
        },
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    devices = list_microphone_devices()

    assert devices == [
        {
            "index": 1,
            "id": "mic, PulseAudio",
            "name": "mic",
            "max_input_channels": 2,
            "default_samplerate": 16000,
        }
    ]
    assert calls == ["terminate", "initialize"]


def test_list_microphone_devices_excludes_monitor_and_generic_virtual_devices(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_sounddevice = types.ModuleType("sounddevice")
    fake_sounddevice._terminate = lambda: None  # type: ignore[attr-defined]
    fake_sounddevice._initialize = lambda: None  # type: ignore[attr-defined]
    fake_sounddevice.query_hostapis = lambda _index: {"name": "PulseAudio"}  # type: ignore[attr-defined]
    fake_sounddevice.query_devices = lambda: [  # type: ignore[attr-defined]
        {
            "name": "alsa_output.usb-HM5.monitor",
            "max_input_channels": 2,
            "default_samplerate": 48000,
            "hostapi": 0,
        },
        {
            "name": "default",
            "max_input_channels": 128,
            "default_samplerate": 44100,
            "hostapi": 0,
        },
        {
            "name": "PipeWire",
            "max_input_channels": 128,
            "default_samplerate": 44100,
            "hostapi": 0,
        },
        {
            "name": "Default Source",
            "max_input_channels": 32,
            "default_samplerate": 48000,
            "hostapi": 0,
        },
        {"name": "spdif", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 0},
        {
            "name": "alsa_input.usb-HM5",
            "max_input_channels": 2,
            "default_samplerate": 16000,
            "hostapi": 0,
        },
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    devices = list_microphone_devices()

    assert devices == [
        {
            "index": 5,
            "id": "alsa_input.usb-HM5, PulseAudio",
            "name": "alsa_input.usb-HM5",
            "max_input_channels": 2,
            "default_samplerate": 16000,
        }
    ]


def test_validate_microphone_device_uses_recognition_input_format(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_sounddevice = types.ModuleType("sounddevice")
    calls: list[dict[str, object]] = []

    def check_input_settings(**kwargs: object) -> None:
        calls.append(kwargs)

    fake_sounddevice.check_input_settings = check_input_settings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    validate_microphone_device(7)

    assert calls == [
        {"device": 7, "samplerate": 16000, "channels": 1, "dtype": "float32"}
    ]


def test_resolve_microphone_device_returns_exact_name_after_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_sounddevice = types.ModuleType("sounddevice")
    calls: list[tuple[object, ...]] = []

    def check_input_settings(**kwargs: object) -> None:
        calls.append(("check", kwargs))

    def terminate() -> None:
        calls.append(("terminate",))

    def initialize() -> None:
        calls.append(("initialize",))

    def query_devices(device: object, kind: str) -> dict[str, object]:
        calls.append(("query", device, kind))
        return {
            "name": "alsa_input.usb-HM5_unique.analog-stereo",
            "hostapi": 2,
        }

    def query_hostapis(hostapi: int) -> dict[str, object]:
        calls.append(("hostapi", hostapi))
        return {"name": "PulseAudio"}

    fake_sounddevice.check_input_settings = check_input_settings  # type: ignore[attr-defined]
    fake_sounddevice._terminate = terminate  # type: ignore[attr-defined]
    fake_sounddevice._initialize = initialize  # type: ignore[attr-defined]
    fake_sounddevice.query_devices = query_devices  # type: ignore[attr-defined]
    fake_sounddevice.query_hostapis = query_hostapis  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    resolved = resolve_microphone_device(20)

    assert resolved == "alsa_input.usb-HM5_unique.analog-stereo, PulseAudio"
    assert calls == [
        ("terminate",),
        ("initialize",),
        (
            "check",
            {"device": 20, "samplerate": 16000, "channels": 1, "dtype": "float32"},
        ),
        ("query", 20, "input"),
        ("hostapi", 2),
        ("query", "alsa_input.usb-HM5_unique.analog-stereo, PulseAudio", "input"),
    ]


def test_resolve_microphone_device_uses_stable_id_instead_of_old_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable_id = "alsa_input.usb-HM5_unique.analog-stereo, PulseAudio"
    fake_sounddevice = types.ModuleType("sounddevice")
    calls: list[object] = []
    fake_sounddevice._terminate = lambda: calls.append("terminate")  # type: ignore[attr-defined]
    fake_sounddevice._initialize = lambda: calls.append("initialize")  # type: ignore[attr-defined]
    fake_sounddevice.check_input_settings = lambda **kwargs: calls.append(kwargs["device"])  # type: ignore[attr-defined]
    fake_sounddevice.query_devices = lambda device, _kind: {  # type: ignore[attr-defined]
        "name": "alsa_input.usb-HM5_unique.analog-stereo",
        "hostapi": 2,
    }
    fake_sounddevice.query_hostapis = lambda _hostapi: {"name": "PulseAudio"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    assert resolve_microphone_device(stable_id) == stable_id
    assert calls == ["terminate", "initialize", stable_id]


def test_resolve_default_microphone_device_returns_stable_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sounddevice = types.ModuleType("sounddevice")
    fake_sounddevice.check_input_settings = lambda **_kwargs: None  # type: ignore[attr-defined]
    fake_sounddevice.query_devices = lambda device, _kind: {  # type: ignore[attr-defined]
        "name": "default-source" if device is None else "default-source",
        "hostapi": 1,
    }
    fake_sounddevice.query_hostapis = lambda _hostapi: {"name": "ALSA"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    assert resolve_microphone_device(None) == "default-source, ALSA"


def test_resolve_microphone_device_rejects_ambiguous_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sounddevice = types.ModuleType("sounddevice")
    fake_sounddevice.check_input_settings = lambda **_kwargs: None  # type: ignore[attr-defined]

    def query_devices(device: object, _kind: str) -> dict[str, object]:
        if isinstance(device, str):
            raise ValueError("Multiple input devices found")
        return {"name": "USB Audio", "hostapi": 0}

    fake_sounddevice.query_devices = query_devices  # type: ignore[attr-defined]
    fake_sounddevice.query_hostapis = lambda _hostapi: {"name": "ALSA"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    with pytest.raises(RuntimeError, match="Multiple input devices found"):
        resolve_microphone_device(3)


@pytest.mark.asyncio
async def test_local_wav_source_emits_100ms_frames_and_one_finish(tmp_path: Path):
    wav_path = tmp_path / "stream.wav"
    samples = np.arange(4000, dtype=np.int16)
    write_wav(wav_path, samples=samples)

    source = LocalWavSource(stream_id=3, path=wav_path, frame_duration_ms=100)
    await source.prepare()

    frames = [frame async for frame in source.frames()]

    assert len(frames) == 4
    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3]
    assert [frame.timestamp_ms for frame in frames] == [0, 100, 200, 300]
    assert [frame.is_finish for frame in frames] == [False, False, False, True]

    assert frames[0].stream_id == 3
    assert frames[0].pcm.dtype == np.float32
    assert frames[0].pcm.shape == (1600,)
    assert np.isclose(frames[0].pcm[0], 0.0)
    assert np.isclose(frames[0].pcm[1], 1.0 / 32768.0)

    assert frames[1].pcm.shape == (1600,)
    assert frames[2].pcm.shape == (800,)
    assert frames[3].pcm.size == 0


@pytest.mark.asyncio
async def test_local_wav_source_can_pace_frames_in_realtime_mode(tmp_path: Path):
    wav_path = tmp_path / "stream.wav"
    samples = np.arange(4000, dtype=np.int16)
    write_wav(wav_path, samples=samples)
    sleep_calls: list[float] = []

    async def fake_sleep(duration_seconds: float) -> None:
        sleep_calls.append(duration_seconds)

    source = LocalWavSource(
        stream_id=3,
        path=wav_path,
        frame_duration_ms=100,
        realtime=True,
        sleep=fake_sleep,
    )
    await source.prepare()

    frames = [frame async for frame in source.frames()]

    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3]
    assert [frame.is_finish for frame in frames] == [False, False, False, True]
    assert sleep_calls == [0.1, 0.1, 0.1]


@pytest.mark.asyncio
async def test_local_wav_source_loops_without_resetting_frame_clock(tmp_path: Path):
    wav_path = tmp_path / "loop.wav"
    write_wav(wav_path, samples=np.arange(3200, dtype=np.int16))
    source = LocalWavSource(
        stream_id=3,
        path=wav_path,
        frame_duration_ms=100,
        realtime=True,
        loop_audio=True,
    )
    sleep_calls = 0

    async def stop_after_two_loops(_duration_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 8:
            await source.stop()

    source._sleep = stop_after_two_loops
    await source.prepare()
    frames = [frame async for frame in source.frames()]

    assert len(frames) == 9
    assert [frame.frame_index for frame in frames[:-1]] == list(range(8))
    assert [frame.timestamp_ms for frame in frames[:-1]] == [index * 100 for index in range(8)]
    assert all(not frame.is_finish for frame in frames[:-1])
    assert frames[-1].is_finish


@pytest.mark.asyncio
async def test_local_wav_source_rejects_empty_loop_audio(tmp_path: Path):
    wav_path = tmp_path / "empty.wav"
    write_wav(wav_path, samples=np.asarray([], dtype=np.int16))
    source = LocalWavSource(stream_id=0, path=wav_path, loop_audio=True)

    with pytest.raises(ValueError, match="cannot loop an empty WAV"):
        await source.prepare()
