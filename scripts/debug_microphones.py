from __future__ import annotations

import argparse
import math
import sys
import threading
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np


SAMPLE_FORMAT = "float32"
CHANNELS = 1
SNAPSHOT_LINE_WIDTH = 125


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float


def parse_device_indexes(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("select at least one microphone device")

    indexes: list[int] = []
    for token in tokens:
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid device index: {token}") from exc
        if index < 0:
            raise ValueError("device indexes must be non-negative")
        if index not in indexes:
            indexes.append(index)
    return indexes


def amplitude_to_dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return -math.inf
    return 20.0 * math.log10(amplitude)


@dataclass(frozen=True)
class DeviceSnapshot:
    rms_dbfs: float
    peak_dbfs: float
    peak_amplitude: float
    callback_count: int
    callback_status: str
    callback_age_seconds: float
    state: str
    max_peak_dbfs: float = -math.inf
    max_peak_amplitude: float = 0.0


class DeviceStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._opened_at: float | None = None
        self._last_callback_at: float | None = None
        self._rms_dbfs = -math.inf
        self._peak_dbfs = -math.inf
        self._peak_amplitude = 0.0
        self._max_peak_dbfs = -math.inf
        self._max_peak_amplitude = 0.0
        self._callback_count = 0
        self._callback_status = ""

    def mark_opened(self) -> None:
        with self._lock:
            opened_at = time.monotonic()
            self._opened_at = opened_at
            self._last_callback_at = None

    def update(self, samples: object, callback_status: object) -> None:
        values = np.asarray(samples, dtype=np.float64)
        if values.size:
            rms = float(np.sqrt(np.mean(np.square(values))))
            peak = float(np.max(np.abs(values)))
        else:
            rms = 0.0
            peak = 0.0

        with self._lock:
            self._rms_dbfs = amplitude_to_dbfs(rms)
            self._peak_dbfs = amplitude_to_dbfs(peak)
            self._peak_amplitude = peak
            if peak > self._max_peak_amplitude:
                self._max_peak_amplitude = peak
                self._max_peak_dbfs = self._peak_dbfs
            self._callback_count += 1
            if callback_status:
                self._callback_status = str(callback_status)
            self._last_callback_at = time.monotonic()

    def snapshot(self, stall_seconds: float) -> DeviceSnapshot:
        with self._lock:
            now = time.monotonic()
            if self._opened_at is None:
                age = 0.0
                state = "WAITING"
            else:
                reference = self._last_callback_at or self._opened_at
                age = now - reference
                if age > stall_seconds:
                    state = "STALLED"
                elif self._callback_status:
                    state = "STATUS"
                elif self._last_callback_at is None:
                    state = "WAITING"
                else:
                    state = "OK"

            return DeviceSnapshot(
                rms_dbfs=self._rms_dbfs,
                peak_dbfs=self._peak_dbfs,
                peak_amplitude=self._peak_amplitude,
                callback_count=self._callback_count,
                callback_status=self._callback_status,
                callback_age_seconds=age,
                state=state,
                max_peak_dbfs=self._max_peak_dbfs,
                max_peak_amplitude=self._max_peak_amplitude,
            )


def list_input_devices(sounddevice_module: object) -> list[InputDevice]:
    devices = []
    for index, device in enumerate(sounddevice_module.query_devices()):
        input_channels = int(device["max_input_channels"])
        if input_channels > 0:
            devices.append(
                InputDevice(
                    index=index,
                    name=str(device["name"]),
                    max_input_channels=input_channels,
                    default_samplerate=float(device["default_samplerate"]),
                )
            )
    return devices


def select_input_devices(
    devices: Sequence[InputDevice], selected_indexes: Sequence[int] | None
) -> list[InputDevice]:
    if selected_indexes is None:
        for device in devices:
            print(
                f"{device.index}: {_sanitize_terminal_text(device.name)} "
                f"({device.max_input_channels} input channels, "
                f"default {device.default_samplerate:g} Hz)"
            )
        selected_indexes = parse_device_indexes(input("Select microphone device indexes: "))

    devices_by_index = {device.index: device for device in devices}
    selected = []
    for index in selected_indexes:
        try:
            selected.append(devices_by_index[index])
        except KeyError as exc:
            raise ValueError(f"device index {index} is not an input device") from exc
    return selected


def _stop_and_close(streams: Sequence[object]) -> None:
    for stream in reversed(streams):
        try:
            stream.stop()
        except BaseException:
            pass
        try:
            stream.close()
        except BaseException:
            pass


def open_streams(
    sounddevice_module: object,
    devices: Sequence[InputDevice],
    stats_by_index: Mapping[int, DeviceStats],
    sample_rate: int,
    frame_ms: int,
) -> list[object]:
    available_devices = {device.index: device for device in list_input_devices(sounddevice_module)}
    blocksize = max(1, int(sample_rate * frame_ms / 1000))
    streams: list[object] = []

    try:
        for requested_device in devices:
            device = available_devices.get(requested_device.index)
            if device is None:
                raise ValueError(f"device index {requested_device.index} is not an input device")
            try:
                stats = stats_by_index[device.index]
            except KeyError as exc:
                raise ValueError(f"missing statistics for device index {device.index}") from exc

            try:
                sounddevice_module.check_input_settings(
                    device=device.index,
                    samplerate=sample_rate,
                    channels=CHANNELS,
                    dtype=SAMPLE_FORMAT,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"device {device.index} ({device.name}) validation failed: {exc}"
                ) from exc

            def callback(indata, _frames, _time_info, status, *, device_stats=stats):
                device_stats.update(indata, status)

            try:
                stream = sounddevice_module.InputStream(
                    device=device.index,
                    samplerate=sample_rate,
                    channels=CHANNELS,
                    dtype=SAMPLE_FORMAT,
                    blocksize=blocksize,
                    callback=callback,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"device {device.index} ({device.name}) open failed: {exc}"
                ) from exc
            streams.append(stream)
            stats.mark_opened()
            try:
                stream.start()
            except Exception as exc:
                raise RuntimeError(
                    f"device {device.index} ({device.name}) start failed: {exc}"
                ) from exc
    except BaseException:
        _stop_and_close(streams)
        raise

    return streams


def _format_dbfs(value: float) -> str:
    if value == -math.inf:
        return _fit_field("-inf", 8, align="right")
    return _fit_field(f"{value:.1f}", 8, align="right")


def _fit_field(value: object, width: int, *, align: str = "left") -> str:
    text = _sanitize_terminal_text(str(value))
    if _terminal_cell_width(text) > width:
        if align == "right":
            text = _clip_terminal_text_from_right(text, width)
        else:
            text = _clip_terminal_text_from_left(text, width)
    if align == "right":
        return _pad_terminal_text(text, width, align="right")
    return _pad_terminal_text(text, width, align="left")


def _is_emoji_base(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or codepoint
        in {
            0x00A9,
            0x00AE,
            0x203C,
            0x2049,
            0x2122,
            0x2139,
            0x3030,
            0x303D,
            0x3297,
            0x3299,
        }
    )


def _is_regional_indicator(char: str) -> bool:
    return 0x1F1E6 <= ord(char) <= 0x1F1FF


def _is_emoji_modifier(char: str) -> bool:
    return 0x1F3FB <= ord(char) <= 0x1F3FF


def _is_variation_selector(char: str) -> bool:
    return ord(char) in {0xFE0E, 0xFE0F}


def _is_keycap_sequence(text: str, start: int) -> bool:
    if text[start] not in "#*0123456789":
        return False
    index = start + 1
    if index < len(text) and _is_variation_selector(text[index]):
        index += 1
    return index < len(text) and text[index] == "\u20e3"


def _consume_emoji_sequence(text: str, start: int) -> int:
    if _is_regional_indicator(text[start]):
        return start + 2 if start + 1 < len(text) and _is_regional_indicator(text[start + 1]) else start + 1

    if _is_keycap_sequence(text, start):
        index = start + 1
        if _is_variation_selector(text[index]):
            index += 1
        return index + 1

    index = start + 1
    if text[start] == "\u200d" and index < len(text) and _is_emoji_base(text[index]):
        index += 1
    while index < len(text):
        char = text[index]
        if _is_variation_selector(char) or _is_emoji_modifier(char) or char == "\u20e3":
            index += 1
        elif char == "\u200d" and index + 1 < len(text) and _is_emoji_base(text[index + 1]):
            index += 2
        else:
            break
    return index


def _sanitize_terminal_text(text: str) -> str:
    sanitized: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if (
            _is_emoji_base(char)
            or _is_regional_indicator(char)
            or _is_keycap_sequence(text, index)
            or char in {"\u200d", "\u20e3"}
            or _is_variation_selector(char)
            or _is_emoji_modifier(char)
        ):
            sanitized.append("?")
            index = _consume_emoji_sequence(text, index)
        else:
            sanitized.append(" " if unicodedata.category(char).startswith("C") else char)
            index += 1
    return "".join(sanitized)


def _char_cell_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _terminal_cell_width(text: str) -> int:
    return sum(_char_cell_width(char) for char in text)


def _clip_terminal_text_from_left(text: str, width: int) -> str:
    cells = 0
    clipped: list[str] = []
    for char in text:
        char_width = _char_cell_width(char)
        if cells + char_width > width:
            break
        clipped.append(char)
        cells += char_width
    return "".join(clipped)


def _clip_terminal_text_from_right(text: str, width: int) -> str:
    cells = 0
    clipped: list[str] = []
    for char in reversed(text):
        char_width = _char_cell_width(char)
        if cells + char_width > width:
            break
        clipped.append(char)
        cells += char_width
    return "".join(reversed(clipped))


def _pad_terminal_text(text: str, width: int, *, align: str) -> str:
    padding = " " * (width - _terminal_cell_width(text))
    if align == "right":
        return padding + text
    return text + padding


def render_snapshot(
    devices: Sequence[InputDevice], snapshots: Mapping[int, DeviceSnapshot]
) -> str:
    header = _fit_field(
        " IDX  NAME                      RMS dBFS  PEAK dBFS  LIN PEAK  MAX dBFS  MAX PEAK  CALLBACKS  AGE(s)  STATE/STATUS",
        SNAPSHOT_LINE_WIDTH,
    )
    rows = [header]
    for device in devices:
        snapshot = snapshots[device.index]
        state = snapshot.state
        if snapshot.callback_status:
            state = f"{state}: {snapshot.callback_status}"
        rows.append(
            f"{_fit_field(device.index, 4, align='right')}  "
            f"{_fit_field(device.name, 24)}  "
            f"{_format_dbfs(snapshot.rms_dbfs)}  "
            f"{_format_dbfs(snapshot.peak_dbfs)}  "
            f"{_fit_field(f'{snapshot.peak_amplitude:.4f}', 8, align='right')}  "
            f"{_format_dbfs(snapshot.max_peak_dbfs)}  "
            f"{_fit_field(f'{snapshot.max_peak_amplitude:.4f}', 8, align='right')}  "
            f"{_fit_field(snapshot.callback_count, 9, align='right')}  "
            f"{_fit_field(f'{snapshot.callback_age_seconds:.2f}', 6, align='right')}  "
            f"{_fit_field(state, 24)}"
        )
    return "\n".join(rows)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number: {value}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Display live per-device microphone PCM statistics.")
    parser.add_argument("--devices", type=parse_device_indexes, help="comma-separated input device indexes")
    parser.add_argument("--sample-rate", type=_positive_int, default=16000, help="sample rate in Hz")
    parser.add_argument("--frame-ms", type=_positive_int, default=100, help="callback frame size in ms")
    parser.add_argument(
        "--stall-seconds",
        type=_positive_float,
        default=2.0,
        help="seconds without a callback before reporting STALLED",
    )
    return parser


def _write_snapshot(rendered: str, cursor_saved: bool = False) -> bool:
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[u\x1b[J" if cursor_saved else "\x1b[s")
        sys.stdout.write(f"{rendered}\n")
        sys.stdout.flush()
        return True

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{timestamp}\n{rendered}", flush=True)
    return False


def _print_stderr(message: str) -> None:
    print(_sanitize_terminal_text(message), file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        import sounddevice
    except ImportError as exc:
        _print_stderr(f"Unable to import sounddevice: {exc}")
        return 1

    streams: list[object] = []
    try:
        available_devices = list_input_devices(sounddevice)
        if not available_devices:
            raise ValueError("no input devices found")
        selected_devices = select_input_devices(available_devices, args.devices)
        stats_by_index = {device.index: DeviceStats() for device in selected_devices}
        streams = open_streams(
            sounddevice,
            selected_devices,
            stats_by_index,
            args.sample_rate,
            args.frame_ms,
        )

        cursor_saved = False
        while True:
            snapshots = {
                device.index: stats_by_index[device.index].snapshot(args.stall_seconds)
                for device in selected_devices
            }
            cursor_saved = _write_snapshot(
                render_snapshot(selected_devices, snapshots), cursor_saved
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        _print_stderr(f"Microphone debugger error: {exc}")
        return 1
    finally:
        _stop_and_close(streams)


if __name__ == "__main__":
    raise SystemExit(main())
