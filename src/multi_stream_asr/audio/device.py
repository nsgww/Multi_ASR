from __future__ import annotations

import threading


_EXCLUDED_INPUT_DEVICE_NAMES = frozenset(
    {
        "default",
        "default source",
        "pipewire",
        "spdif",
        "sysdefault",
    }
)
_SOUNDDEVICE_LOCK = threading.Lock()


def _require_sounddevice(
    error_message: str = "sounddevice is required to list microphone devices",
):
    try:
        import sounddevice  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(error_message) from exc

    return sounddevice


def _refresh_sounddevice(sounddevice: object) -> None:
    terminate = getattr(sounddevice, "_terminate", None)
    initialize = getattr(sounddevice, "_initialize", None)
    if callable(terminate) and callable(initialize):
        terminate()
        initialize()


def _device_selector(sounddevice: object, device: dict[str, object]) -> str:
    name = str(device["name"]).strip()
    hostapi = sounddevice.query_hostapis(int(device["hostapi"]))  # type: ignore[attr-defined]
    hostapi_name = str(hostapi["name"]).strip()
    if not name or not hostapi_name:
        raise ValueError("device or host API name is empty")
    return f"{name}, {hostapi_name}"


def list_microphone_devices() -> list[dict[str, object]]:
    sounddevice = _require_sounddevice()

    with _SOUNDDEVICE_LOCK:
        _refresh_sounddevice(sounddevice)
        devices: list[dict[str, object]] = []
        for index, device in enumerate(sounddevice.query_devices()):
            if not _is_selectable_microphone_device(device):
                continue
            devices.append(
                {
                    "index": index,
                    "id": _device_selector(sounddevice, device),
                    "name": device["name"],
                    "max_input_channels": device["max_input_channels"],
                    "default_samplerate": device["default_samplerate"],
                }
            )
        return devices


def validate_microphone_device(device: int | None) -> None:
    sounddevice = _require_sounddevice(
        "sounddevice is required to validate microphone input"
    )
    try:
        sounddevice.check_input_settings(
            device=device,
            samplerate=16000,
            channels=1,
            dtype="float32",
        )
    except Exception as exc:
        label = "default" if device is None else str(device)
        raise RuntimeError(
            f"device {label} does not support 16 kHz mono float32 input: {exc}"
        ) from exc


def resolve_microphone_device(device: int | str | None) -> str:
    """Validate an index and return a process-independent PortAudio selector."""
    sounddevice = _require_sounddevice(
        "sounddevice is required to resolve microphone input"
    )
    try:
        with _SOUNDDEVICE_LOCK:
            _refresh_sounddevice(sounddevice)
            sounddevice.check_input_settings(
                device=device,
                samplerate=16000,
                channels=1,
                dtype="float32",
            )
            device_info = sounddevice.query_devices(device, "input")
            selector = _device_selector(sounddevice, device_info)
            sounddevice.query_devices(selector, "input")
            return selector
    except Exception as exc:
        label = "default" if device is None else str(device)
        raise RuntimeError(f"unable to resolve microphone device {label}: {exc}") from exc


def _is_selectable_microphone_device(device: dict[str, object]) -> bool:
    name = str(device["name"]).strip().lower()
    return (
        int(device["max_input_channels"]) > 0
        and name not in _EXCLUDED_INPUT_DEVICE_NAMES
        and not name.endswith(".monitor")
    )
