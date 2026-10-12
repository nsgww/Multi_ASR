from __future__ import annotations

import math
import sys
import unicodedata
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scripts import debug_microphones as debug


def test_parse_device_indexes_accepts_multiple_unique_devices():
    assert debug.parse_device_indexes("22, 24,22") == [22, 24]


@pytest.mark.parametrize(
    ("value", "message"),
    [("", "select at least one"), ("22,x", "invalid device index"), ("-1", "non-negative")],
)
def test_parse_device_indexes_rejects_invalid_values(value, message):
    with pytest.raises(ValueError, match=message):
        debug.parse_device_indexes(value)


def test_amplitude_to_dbfs_uses_float_full_scale():
    assert debug.amplitude_to_dbfs(1.0) == pytest.approx(0.0)
    assert debug.amplitude_to_dbfs(0.5) == pytest.approx(-6.0206, abs=1e-4)
    assert debug.amplitude_to_dbfs(0.0) == -math.inf


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_device_stats_isolates_callbacks_and_marks_stalls(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(debug.time, "monotonic", clock)
    first = debug.DeviceStats()
    second = debug.DeviceStats()
    first.mark_opened()
    second.mark_opened()

    assert first.snapshot(stall_seconds=2.0).state == "WAITING"

    first.update(np.full((1600, 1), 0.5, dtype=np.float32), "")
    second.update(np.zeros((1600, 1), dtype=np.float32), "overflow")

    assert first.snapshot(stall_seconds=2.0).state == "OK"
    clock.advance(2.0)
    assert first.snapshot(stall_seconds=2.0).state == "OK"
    clock.advance(0.1)

    first_snapshot = first.snapshot(stall_seconds=2.0)
    second_snapshot = second.snapshot(stall_seconds=2.0)
    assert first_snapshot.state == "STALLED"
    assert first_snapshot.rms_dbfs == pytest.approx(-6.0206, abs=1e-4)
    assert first_snapshot.peak_dbfs == pytest.approx(-6.0206, abs=1e-4)
    assert first_snapshot.peak_amplitude == pytest.approx(0.5)
    assert first_snapshot.callback_count == 1
    assert first_snapshot.callback_age_seconds == pytest.approx(2.1)
    assert second_snapshot.callback_status == "overflow"
    assert second_snapshot.callback_count == 1
    assert second_snapshot.state == "STALLED"


def test_device_stats_marks_stream_without_callbacks_stalled(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(debug.time, "monotonic", clock)
    stats = debug.DeviceStats()
    stats.mark_opened()

    clock.advance(2.1)

    snapshot = stats.snapshot(stall_seconds=2.0)
    assert snapshot.callback_count == 0
    assert snapshot.callback_age_seconds == pytest.approx(2.1)
    assert snapshot.state == "STALLED"


def test_device_stats_marks_stale_status_callback_stalled_and_preserves_status(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(debug.time, "monotonic", clock)
    stats = debug.DeviceStats()
    stats.mark_opened()
    stats.update(np.zeros((1, 1), dtype=np.float32), "overflow")

    clock.advance(2.1)

    snapshot = stats.snapshot(stall_seconds=2.0)
    assert snapshot.state == "STALLED"
    assert snapshot.callback_status == "overflow"
    assert "STALLED: overflow" in debug.render_snapshot(
        [debug.InputDevice(1, "mono", 1, 16000)], {1: snapshot}
    )


def test_device_stats_preserves_transient_status_after_clean_callback():
    stats = debug.DeviceStats()
    stats.mark_opened()
    stats.update(np.zeros((1, 1), dtype=np.float32), "input overflow")

    stats.update(np.zeros((1, 1), dtype=np.float32), "")

    snapshot = stats.snapshot(stall_seconds=2.0)
    assert snapshot.callback_count == 2
    assert snapshot.callback_status == "input overflow"
    assert snapshot.state == "STATUS"


def test_device_stats_retains_run_max_peak_after_lower_callback():
    stats = debug.DeviceStats()
    stats.mark_opened()

    stats.update(np.full((16, 1), 0.8, dtype=np.float32), "")
    stats.update(np.full((16, 1), 0.2, dtype=np.float32), "")

    snapshot = stats.snapshot(stall_seconds=2.0)
    assert snapshot.peak_amplitude == pytest.approx(0.2)
    assert snapshot.peak_dbfs == pytest.approx(debug.amplitude_to_dbfs(0.2))
    assert snapshot.max_peak_amplitude == pytest.approx(0.8)
    assert snapshot.max_peak_dbfs == pytest.approx(debug.amplitude_to_dbfs(0.8))


def test_device_snapshot_is_immutable():
    snapshot = debug.DeviceStats().snapshot(stall_seconds=2.0)

    with pytest.raises(FrozenInstanceError):
        snapshot.state = "OK"


class FakeStream:
    def __init__(self, *, fail_to_start=False, **kwargs):
        self.kwargs = kwargs
        self.fail_to_start = fail_to_start
        self.events: list[str] = []

    def start(self):
        self.events.append("start")
        if self.fail_to_start:
            raise RuntimeError("cannot start")
        return self

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")


class BaseExceptionCleanupStream:
    def __init__(self, stop_error=None, close_error=None):
        self.events: list[str] = []
        self.stop_error = stop_error
        self.close_error = close_error

    def stop(self):
        self.events.append("stop")
        if self.stop_error is not None:
            raise self.stop_error

    def close(self):
        self.events.append("close")
        if self.close_error is not None:
            raise self.close_error


class FakeSoundDevice:
    def __init__(self, fail_index: int | None = None):
        self.devices = [
            {"name": "speakers", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "mono", "max_input_channels": 1, "default_samplerate": 16000},
            {"name": "stereo", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "array", "max_input_channels": 4, "default_samplerate": 48000},
        ]
        self.fail_index = fail_index
        self.streams: list[FakeStream] = []
        self.checked_settings: list[dict[str, object]] = []

    def query_devices(self):
        return self.devices

    def check_input_settings(self, **kwargs):
        self.checked_settings.append(kwargs)

    def InputStream(self, **kwargs):
        stream = FakeStream(
            fail_to_start=kwargs["device"] == self.fail_index,
            **kwargs,
        )
        self.streams.append(stream)
        return stream


class ValidationFailureSoundDevice(FakeSoundDevice):
    def check_input_settings(self, **kwargs):
        super().check_input_settings(**kwargs)
        if kwargs["device"] == 2:
            raise RuntimeError("unsupported sample rate")


class OpenFailureSoundDevice(FakeSoundDevice):
    def InputStream(self, **kwargs):
        if kwargs["device"] == 2:
            raise RuntimeError("unable to open stream")
        return super().InputStream(**kwargs)


class KeyboardInterruptStartSoundDevice(FakeSoundDevice):
    def __init__(self):
        super().__init__()
        self.interrupt = KeyboardInterrupt("interrupt during start")

    def InputStream(self, **kwargs):
        stream = super().InputStream(**kwargs)
        if kwargs["device"] == 2:
            def start():
                stream.events.append("start")
                raise self.interrupt

            stream.start = start
        return stream


def test_list_input_devices_and_interactive_selection(monkeypatch):
    sounddevice = FakeSoundDevice()

    devices = debug.list_input_devices(sounddevice)
    assert devices == [
        debug.InputDevice(1, "mono", 1, 16000),
        debug.InputDevice(2, "stereo", 2, 44100),
        debug.InputDevice(3, "array", 4, 48000),
    ]

    monkeypatch.setattr("builtins.input", lambda _: "1,3")
    assert debug.select_input_devices(devices, None) == [devices[0], devices[2]]


def test_interactive_selection_sanitizes_device_names(capsys, monkeypatch):
    device = debug.InputDevice(1, "Room\n\r\t\x1b[31m麦克风", 1, 16000)

    monkeypatch.setattr("builtins.input", lambda _: "1")

    assert debug.select_input_devices([device], None) == [device]

    output = capsys.readouterr().out
    assert output == "1: Room    [31m麦克风 (1 input channels, default 16000 Hz)\n"
    assert all(control not in output.replace("\n", "") for control in ("\r", "\t", "\x1b"))


def test_main_uses_explicit_devices_without_prompt(monkeypatch):
    sounddevice = FakeSoundDevice()
    selected_devices = []

    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("input() was called"))
    monkeypatch.setattr(
        debug,
        "open_streams",
        lambda _module, devices, *_args: selected_devices.extend(devices) or [],
    )
    monkeypatch.setattr(debug.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt))

    assert debug.main(["--devices", "1,3"]) == 0
    assert selected_devices == [
        debug.InputDevice(1, "mono", 1, 16000),
        debug.InputDevice(3, "array", 4, 48000),
    ]


def _assert_sanitized_stderr(stderr, expected_fragments):
    assert stderr.endswith("\n")
    message = stderr.removesuffix("\n")
    assert "\n" not in message
    assert all(not unicodedata.category(char).startswith("C") for char in message)
    for fragment in expected_fragments:
        assert fragment in message


def test_main_sanitizes_import_error_before_writing_stderr(capsys, monkeypatch):
    real_import = __import__

    def malicious_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("missing\nbackend\x1b[31m\x1b]0;owned\x07")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", malicious_import)

    assert debug.main([]) == 1

    _assert_sanitized_stderr(
        capsys.readouterr().err,
        ["Unable to import sounddevice:", "missing backend [31m ]0;owned "],
    )


def test_main_sanitizes_device_and_backend_error_before_writing_stderr(
    capsys, monkeypatch
):
    class MaliciousFailureSoundDevice(FakeSoundDevice):
        def __init__(self):
            super().__init__()
            self.devices[1]["name"] = "HM5\nmic\x1b[32m\x1b]2;device-owned\x07"

        def check_input_settings(self, **kwargs):
            super().check_input_settings(**kwargs)
            raise RuntimeError("backend\nfailed\x1b[31m\x1b]0;backend-owned\x07")

    monkeypatch.setitem(sys.modules, "sounddevice", MaliciousFailureSoundDevice())

    assert debug.main(["--devices", "1"]) == 1

    _assert_sanitized_stderr(
        capsys.readouterr().err,
        [
            "Microphone debugger error:",
            "HM5 mic [32m ]2;device-owned ",
            "backend failed [31m ]0;backend-owned ",
        ],
    )


def test_open_streams_uses_shared_audio_parameters_and_isolated_callbacks():
    sounddevice = FakeSoundDevice()
    devices = debug.list_input_devices(sounddevice)
    stats = {device.index: debug.DeviceStats() for device in devices[:2]}

    streams = debug.open_streams(sounddevice, devices[:2], stats, 16000, 100)

    assert streams == sounddevice.streams
    for stream, device in zip(streams, devices[:2], strict=True):
        assert stream.kwargs == {
            "device": device.index,
            "samplerate": 16000,
            "channels": 1,
            "dtype": "float32",
            "blocksize": 1600,
            "callback": stream.kwargs["callback"],
        }
    assert sounddevice.checked_settings == [
        {"device": 1, "samplerate": 16000, "channels": 1, "dtype": "float32"},
        {"device": 2, "samplerate": 16000, "channels": 1, "dtype": "float32"},
    ]

    streams[0].kwargs["callback"](np.full((1, 1), 0.5), 1, None, "")
    streams[1].kwargs["callback"](np.zeros((1, 1)), 1, None, "overflow")
    assert stats[1].snapshot(2.0).callback_count == 1
    assert stats[2].snapshot(2.0).callback_status == "overflow"


def test_open_streams_preserves_callback_delivered_synchronously_by_start(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(debug.time, "monotonic", clock)
    sounddevice = FakeSoundDevice()
    device = debug.list_input_devices(sounddevice)[0]
    stats = {device.index: debug.DeviceStats()}

    original_input_stream = sounddevice.InputStream

    def input_stream(**kwargs):
        stream = original_input_stream(**kwargs)
        original_start = stream.start

        def start():
            kwargs["callback"](np.full((1, 1), 0.5), 1, None, "")
            return original_start()

        stream.start = start
        return stream

    sounddevice.InputStream = input_stream

    debug.open_streams(sounddevice, [device], stats, 16000, 100)

    snapshot = stats[device.index].snapshot(stall_seconds=2.0)
    assert snapshot.callback_count == 1
    assert snapshot.state == "OK"


def test_open_streams_cleans_up_every_created_stream_when_start_fails():
    sounddevice = FakeSoundDevice(fail_index=2)
    devices = debug.list_input_devices(sounddevice)
    stats = {device.index: debug.DeviceStats() for device in devices[:2]}

    with pytest.raises(RuntimeError, match="cannot start"):
        debug.open_streams(sounddevice, devices[:2], stats, 16000, 100)

    assert sounddevice.streams[0].events == ["start", "stop", "close"]
    assert sounddevice.streams[1].events == ["start", "stop", "close"]


def test_open_streams_rolls_back_when_start_raises_base_exception():
    sounddevice = KeyboardInterruptStartSoundDevice()
    devices = debug.list_input_devices(sounddevice)
    stats = {device.index: debug.DeviceStats() for device in devices[:2]}

    with pytest.raises(KeyboardInterrupt) as exc_info:
        debug.open_streams(sounddevice, devices[:2], stats, 16000, 100)

    assert exc_info.value is sounddevice.interrupt
    assert sounddevice.streams[0].events == ["start", "stop", "close"]
    assert sounddevice.streams[1].events == ["start", "stop", "close"]


@pytest.mark.parametrize(
    "error",
    [KeyboardInterrupt("stop"), SystemExit("close"), BaseException("other")],
)
def test_stop_and_close_continues_after_base_exception_cleanup_errors(error):
    first = BaseExceptionCleanupStream(stop_error=error)
    second = BaseExceptionCleanupStream(close_error=error)
    third = BaseExceptionCleanupStream()

    debug._stop_and_close([first, second, third])

    assert first.events == ["stop", "close"]
    assert second.events == ["stop", "close"]
    assert third.events == ["stop", "close"]


@pytest.mark.parametrize(
    ("sounddevice", "cause_message"),
    [
        (ValidationFailureSoundDevice(), "unsupported sample rate"),
        (OpenFailureSoundDevice(), "unable to open stream"),
        (FakeSoundDevice(fail_index=2), "cannot start"),
    ],
    ids=["validation", "open", "start"],
)
def test_open_streams_wraps_device_failures_with_context(sounddevice, cause_message):
    device = debug.list_input_devices(sounddevice)[1]
    stats = {device.index: debug.DeviceStats()}

    with pytest.raises(RuntimeError, match=r"device 2 \(stereo\)") as exc_info:
        debug.open_streams(sounddevice, [device], stats, 16000, 100)

    assert str(exc_info.value.__cause__) == cause_message


def test_render_snapshot_formats_stable_device_rows():
    device = debug.InputDevice(1, "mono", 1, 16000)
    snapshot = debug.DeviceSnapshot(
        rms_dbfs=-math.inf,
        peak_dbfs=-6.0206,
        peak_amplitude=0.5,
        callback_count=7,
        callback_status="overflow",
        callback_age_seconds=0.125,
        state="STATUS",
        max_peak_dbfs=-1.9382,
        max_peak_amplitude=0.8,
    )

    rendered = debug.render_snapshot([device], {1: snapshot})

    assert "RMS dBFS" in rendered
    assert "-inf" in rendered
    assert "-6.0" in rendered
    assert "0.5000" in rendered
    assert "MAX dBFS" in rendered
    assert "MAX PEAK" in rendered
    assert "-1.9" in rendered
    assert "0.8000" in rendered
    assert "overflow" in rendered


def test_render_snapshot_clips_extreme_values_without_shifting_columns():
    normal_device = debug.InputDevice(1, "mono", 1, 16000)
    normal_snapshot = debug.DeviceSnapshot(
        rms_dbfs=-math.inf,
        peak_dbfs=-6.0206,
        peak_amplitude=0.5,
        callback_count=7,
        callback_status="overflow",
        callback_age_seconds=0.125,
        state="STATUS",
        max_peak_dbfs=-1.9382,
        max_peak_amplitude=0.8,
    )
    extreme_device = debug.InputDevice(123456789, "x" * 100, 1, 16000)
    extreme_snapshot = debug.DeviceSnapshot(
        rms_dbfs=1e100,
        peak_dbfs=-1e100,
        peak_amplitude=1e100,
        callback_count=10**100,
        callback_status="status" * 100,
        callback_age_seconds=1e100,
        state="STATUS",
        max_peak_dbfs=1e100,
        max_peak_amplitude=1e100,
    )

    normal_lines = debug.render_snapshot(
        [normal_device], {normal_device.index: normal_snapshot}
    ).splitlines()
    extreme_lines = debug.render_snapshot(
        [extreme_device], {extreme_device.index: extreme_snapshot}
    ).splitlines()
    header = normal_lines[0]

    assert len(normal_lines[1]) == len(header)
    assert len(extreme_lines[1]) == len(header)
    column_starts = [
        0,
        header.index("NAME"),
        header.index("RMS dBFS"),
        header.index("PEAK dBFS"),
        header.index("LIN PEAK"),
        header.index("MAX dBFS"),
        header.index("MAX PEAK"),
        header.index("CALLBACKS"),
        header.index("AGE(s)"),
        header.index("STATE/STATUS"),
    ]
    for start, end in zip(column_starts, column_starts[1:] + [len(header)], strict=True):
        assert len(extreme_lines[1][start:end]) == len(normal_lines[1][start:end])
    assert extreme_lines[1][0:4] == "6789"
    assert extreme_lines[1][column_starts[1] : column_starts[2]] == "x" * 24 + "  "


@pytest.mark.parametrize("control", ["\n", "\r", "\t", "\x1b"])
def test_render_snapshot_sanitizes_control_characters_in_device_name(control):
    device = debug.InputDevice(1, f"before{control}after", 1, 16000)
    snapshot = debug.DeviceSnapshot(-math.inf, -math.inf, 0.0, 0, "", 0.0, "WAITING")

    rendered = debug.render_snapshot([device], {1: snapshot})
    lines = rendered.splitlines()

    assert len(lines) == 2
    assert all(debug._terminal_cell_width(line) == debug.SNAPSHOT_LINE_WIDTH for line in lines)
    assert control not in rendered.replace("\n", "")


@pytest.mark.parametrize("control", ["\n", "\r", "\t", "\x1b"])
def test_render_snapshot_sanitizes_control_characters_in_callback_status(control):
    device = debug.InputDevice(1, "mono", 1, 16000)
    snapshot = debug.DeviceSnapshot(
        -math.inf, -math.inf, 0.0, 1, f"before{control}after", 0.0, "STATUS"
    )

    rendered = debug.render_snapshot([device], {1: snapshot})
    lines = rendered.splitlines()

    assert len(lines) == 2
    assert all(debug._terminal_cell_width(line) == debug.SNAPSHOT_LINE_WIDTH for line in lines)
    assert control not in rendered.replace("\n", "")


def test_render_snapshot_keeps_wide_unicode_name_within_fixed_width_row():
    device = debug.InputDevice(1, "麦克风" * 20, 1, 16000)
    snapshot = debug.DeviceSnapshot(-math.inf, -math.inf, 0.0, 0, "", 0.0, "WAITING")

    rendered = debug.render_snapshot([device], {1: snapshot})
    lines = rendered.splitlines()

    assert len(lines) == 2
    assert all(debug._terminal_cell_width(line) == debug.SNAPSHOT_LINE_WIDTH for line in lines)
    assert debug._terminal_cell_width(debug._fit_field(device.name, 24)) == 24
    assert unicodedata.east_asian_width("麦") == "W"


@pytest.mark.parametrize(
    ("name", "display_name"),
    [
        ("coffee ☕\ufe0f", "coffee ?"),
        ("keycap 1\ufe0f\u20e3", "keycap ?"),
        ("family 👨\u200d👩\u200d👧\u200d👦", "family ?"),
        ("flag 🇨🇳", "flag ?"),
        ("麦克风", "麦克风"),
    ],
)
def test_render_snapshot_sanitizes_emoji_graphemes_without_narrowing_cjk(name, display_name):
    device = debug.InputDevice(1, name, 1, 16000)
    snapshot = debug.DeviceSnapshot(-math.inf, -math.inf, 0.0, 0, "", 0.0, "WAITING")

    rendered = debug.render_snapshot([device], {1: snapshot})
    lines = rendered.splitlines()
    name_field = debug._fit_field(name, 24)

    assert name_field == display_name + " " * (24 - debug._terminal_cell_width(display_name))
    assert display_name in rendered
    assert all(debug._terminal_cell_width(line) == debug.SNAPSHOT_LINE_WIDTH for line in lines)
    assert all(
        not unicodedata.category(char).startswith("C")
        for char in rendered.replace("\n", "")
    )
    assert "\ufe0f" not in rendered
    assert "\u20e3" not in rendered


class CapturingStdout:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty
        self.output = ""
        self.flush_count = 0

    def isatty(self):
        return self._is_tty

    def write(self, text):
        self.output += text

    def flush(self):
        self.flush_count += 1


@pytest.mark.parametrize(
    ("first_columns", "second_columns"),
    [(40, 160), (160, 40)],
    ids=["narrow-to-wide", "wide-to-narrow"],
)
def test_write_snapshot_redraws_tty_from_saved_origin_across_resizes(
    monkeypatch, first_columns, second_columns
):
    stdout = CapturingStdout(is_tty=True)
    monkeypatch.setattr(debug.sys, "stdout", stdout)

    first_frame = "A" * debug.SNAPSHOT_LINE_WIDTH + "\n" + "B" * debug.SNAPSHOT_LINE_WIDTH
    second_frame = "C" * debug.SNAPSHOT_LINE_WIDTH + "\n" + "D" * debug.SNAPSHOT_LINE_WIDTH

    stdout.columns = first_columns
    cursor_saved = debug._write_snapshot(first_frame, False)
    assert cursor_saved is True
    stdout.columns = second_columns
    assert debug._write_snapshot(second_frame, cursor_saved) is True

    assert stdout.output == (
        f"\x1b[s{'A' * debug.SNAPSHOT_LINE_WIDTH}\n"
        f"{'B' * debug.SNAPSHOT_LINE_WIDTH}\n"
        f"\x1b[u\x1b[J{'C' * debug.SNAPSHOT_LINE_WIDTH}\n"
        f"{'D' * debug.SNAPSHOT_LINE_WIDTH}\n"
    )
    assert "\x1b[H" not in stdout.output
    assert "\x1b[2K" not in stdout.output
    assert stdout.flush_count == 2


def test_write_snapshot_appends_timestamped_redirected_output(monkeypatch):
    class FixedDatetime:
        @classmethod
        def now(cls):
            return cls()

        def astimezone(self):
            return self

        def isoformat(self, *, timespec):
            assert timespec == "seconds"
            return "2026-07-14T12:34:56+08:00"

    stdout = CapturingStdout(is_tty=False)
    monkeypatch.setattr(debug, "datetime", FixedDatetime)
    monkeypatch.setattr(debug.sys, "stdout", stdout)

    debug._write_snapshot("first", cursor_saved=False)
    debug._write_snapshot("second", cursor_saved=True)

    assert stdout.output == (
        "2026-07-14T12:34:56+08:00\nfirst\n"
        "2026-07-14T12:34:56+08:00\nsecond\n"
    )
    assert "\x1b" not in stdout.output
    assert stdout.flush_count == 2


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--sample-rate", "0"),
        ("--frame-ms", "-1"),
        ("--stall-seconds", "0"),
        ("--stall-seconds", "nan"),
        ("--stall-seconds", "inf"),
        ("--stall-seconds", "-inf"),
    ],
)
def test_main_rejects_non_positive_numeric_options(option, value):
    with pytest.raises(SystemExit, match="2"):
        debug.main([option, value])
