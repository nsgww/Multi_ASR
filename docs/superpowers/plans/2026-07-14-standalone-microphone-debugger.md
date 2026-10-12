# Standalone Multi-Microphone Debugger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python script that interactively selects and concurrently monitors multiple microphones while printing live per-device dBFS and callback health.

**Architecture:** `scripts/debug_microphones.py` is self-contained and imports only the standard library, NumPy, and sounddevice. PortAudio callbacks update lock-protected per-device state; the main thread renders snapshots and detects callback stalls without blocking audio threads.

**Tech Stack:** Python 3.13, `sounddevice`, NumPy, argparse, pytest.

## Global Constraints

- The script must not import `multi_stream_asr` or start recognition, model, pipeline, or Web components.
- Default capture format is 16 kHz, mono `float32`, with 100ms callback blocks.
- Default callback stall threshold is 2.0 seconds.
- Interactive selection accepts one or more comma-separated PortAudio device indexes.
- `--devices 22,24` bypasses interactive selection.
- Use the project-local `.venv313` and Python 3.13 for local and K3 validation.
- Remote package installation, if required, must follow `~/.codex/docs/k3/remote-device-python.md`.

---

### Task 1: Pure parsing, signal measurement, and device state

**Files:**
- Create: `scripts/debug_microphones.py`
- Create: `tests/test_debug_microphones.py`

**Interfaces:**
- Produces: `parse_device_indexes(value: str) -> list[int]`.
- Produces: `amplitude_to_dbfs(amplitude: float) -> float`.
- Produces: `DeviceStats` with `update()`, `snapshot()`, and `mark_opened()`.
- Produces: `DeviceSnapshot` immutable values consumed by terminal rendering in Task 2.

- [ ] **Step 1: Write failing tests for selection parsing and dBFS**

Add tests that import the script through `scripts.debug_microphones` and assert:

```python
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
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PATH="$PWD/.venv313/bin:$PATH" PYTHONPATH=. \
  .venv313/bin/python -m pytest -q tests/test_debug_microphones.py
```

Expected: collection fails because `scripts.debug_microphones` does not exist.

- [ ] **Step 3: Implement parsing and dBFS**

Create the script with constants and pure functions:

```python
SAMPLE_FORMAT = "float32"
CHANNELS = 1


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
```

- [ ] **Step 4: Add failing tests for independent callback state and stalls**

Use a fake monotonic clock and two `DeviceStats` objects. Verify RMS/peak, callback counts, callback status isolation, elapsed time, and `STALLED` only after the configured threshold:

```python
first.update(np.full((1600, 1), 0.5, dtype=np.float32), "")
second.update(np.zeros((1600, 1), dtype=np.float32), "overflow")
clock.advance(2.1)

assert first.snapshot(stall_seconds=2.0).state == "STALLED"
assert first.snapshot(stall_seconds=2.0).rms_dbfs == pytest.approx(-6.0206, abs=1e-4)
assert second.snapshot(stall_seconds=2.0).callback_status == "overflow"
assert second.snapshot(stall_seconds=2.0).callback_count == 1
```

- [ ] **Step 5: Run the focused state tests and verify RED**

Run the test file again. Expected: parsing and dBFS tests pass; state tests fail because `DeviceStats` is absent.

- [ ] **Step 6: Implement thread-safe state**

Add a frozen `DeviceSnapshot` dataclass and a `DeviceStats` class. `update()` converts callback input with `np.asarray`, calculates RMS in float64 to avoid precision loss, calculates absolute peak, stores callback status, increments count, and records monotonic time while holding a `threading.Lock`. `snapshot(stall_seconds)` returns `WAITING`, `OK`, `STATUS`, or `STALLED` without exposing mutable state.

- [ ] **Step 7: Run Task 1 tests and commit**

Run the test file and expect all Task 1 tests to pass, then commit:

```bash
git add scripts/debug_microphones.py tests/test_debug_microphones.py
git commit -m "feat: add microphone debug signal metrics"
```

---

### Task 2: Device discovery, concurrent streams, and live CLI

**Files:**
- Modify: `scripts/debug_microphones.py`
- Modify: `tests/test_debug_microphones.py`

**Interfaces:**
- Consumes: `parse_device_indexes`, `DeviceStats`, and `DeviceSnapshot` from Task 1.
- Produces: `list_input_devices(sounddevice_module) -> list[InputDevice]`.
- Produces: `open_streams(...) -> list[object]` with transactional cleanup.
- Produces: `render_snapshot(...) -> str` and `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing tests for discovery and interactive selection**

Create a fake sounddevice module whose `query_devices()` returns output-only, mono, stereo, and four-channel entries. Verify only devices with `max_input_channels > 0` are listed and that the prompt result `"1,3"` selects both devices. Verify explicit `--devices` avoids calling `input()`.

- [ ] **Step 2: Write failing tests for stream parameters and cleanup**

Use fake `InputStream` objects and assert each selected device receives:

```python
{
    "device": index,
    "samplerate": 16000,
    "channels": 1,
    "dtype": "float32",
    "blocksize": 1600,
    "callback": callback,
}
```

Make the second stream raise from `start()` and assert the first stream is stopped and closed before the exception propagates.

- [ ] **Step 3: Run Task 2 tests and verify RED**

Run:

```bash
PATH="$PWD/.venv313/bin:$PATH" PYTHONPATH=. \
  .venv313/bin/python -m pytest -q tests/test_debug_microphones.py
```

Expected: new tests fail because discovery, stream lifecycle, and CLI functions are absent.

- [ ] **Step 4: Implement discovery and stream lifecycle**

Add `InputDevice(index, name, max_input_channels, default_samplerate)`. Validate selected indexes against fresh discovery and call `sounddevice.check_input_settings()` before opening. Build one callback closure per device so callback status and PCM update only its `DeviceStats`. Start all streams transactionally; on any failure stop and close every stream already created.

- [ ] **Step 5: Implement stable terminal rendering**

Render one header and one fixed-width row per device. Format finite dBFS with one decimal place and negative infinity as `-inf`. Include current RMS dBFS, peak dBFS, linear peak, callback count, callback age, and state/status. When `sys.stdout.isatty()` is true, move the cursor up by the prior table height; otherwise prefix each snapshot with an ISO timestamp and print normally.

- [ ] **Step 6: Implement argparse and shutdown**

Add `--devices`, `--sample-rate`, `--frame-ms`, and `--stall-seconds`. Reject non-positive numeric values through `argparse.ArgumentTypeError`. Import sounddevice inside `main()` so `--help` and pure tests remain usable without PortAudio. Loop at 10Hz until `KeyboardInterrupt`, then stop and close streams in reverse order and return 0. Print operational errors to stderr and return 1.

- [ ] **Step 7: Verify CLI behavior without hardware**

Run:

```bash
.venv313/bin/python scripts/debug_microphones.py --help
```

Expected: usage lists all four options and exits 0 without importing project modules.

- [ ] **Step 8: Run focused and complete tests, then commit**

Run:

```bash
PATH="$PWD/.venv313/bin:$PATH" PYTHONPATH=. \
  .venv313/bin/python -m pytest -q tests/test_debug_microphones.py
PATH="$PWD/.venv313/bin:$PATH" PYTHONPATH=. \
  .venv313/bin/python -m pytest -q
```

Expected: all tests pass. Commit only the script and its tests:

```bash
git add scripts/debug_microphones.py tests/test_debug_microphones.py
git commit -m "feat: add standalone multi-microphone debugger"
```

---

### Task 3: K3 dual-microphone validation

**Files:**
- No tracked file changes.

**Interfaces:**
- Consumes: `scripts/debug_microphones.py --devices <HM5>,<SPV>`.
- Produces: terminal evidence that both callback counters and dBFS values update independently for at least 20 seconds.

- [ ] **Step 1: Synchronize committed code to the development board**

Push the local branch, then on `bianbu@10.0.91.51` pull the same commit in `~/Workspace/projects/multi-stream-asr`. Do not overwrite board-only model files or `.venv313`.

- [ ] **Step 2: Stop the existing ASR service for an isolated microphone test**

Record the existing service command, stop `python -m multi_stream_asr.main`, and verify no process holds the selected capture devices. Leave PipeWire and WirePlumber running.

- [ ] **Step 3: Discover current device indexes**

Run under the board-local Python 3.13 environment:

```bash
cd ~/Workspace/projects/multi-stream-asr
. .venv313/bin/activate
python scripts/debug_microphones.py
```

At the prompt select the current HM5 and SPV PulseAudio source indexes shown by the script.

- [ ] **Step 4: Run the non-interactive 20-second validation**

Run with `timeout --signal=INT 20s` and explicit indexes. Expected for both rows: callback count rises by roughly 10 per second, callback age remains below 0.5 seconds, state stays `OK`, and speaking into each microphone raises that row's RMS/peak dBFS. If a device disconnects, preserve the script output and matching `journalctl -k` USB lines as root-cause evidence.

- [ ] **Step 5: Restore the Web service and verify health**

Restart the exact recorded ASR service command from Task 3 Step 2. Verify `/health` returns HTTP 200 and model state reaches `loaded` before completing the task.
