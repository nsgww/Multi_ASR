# Web Configuration and Microphone Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Web UI configure 2/4/6/8 lanes, local files or microphones, and 0.6B/1.7B shared Qwen models safely, with a working microphone source and lane-local scrollable results.

**Architecture:** Keep `AppConfig` immutable at runtime. A Web request is validated into a new `AppConfig` while unloaded; the process manager serializes a lane source descriptor in the existing start command, and the lane runtime selects `LocalWavSource` or `MicrophoneSource`. The static HTML keeps an unapplied configuration draft, preflights model files, drives controls from lifecycle state, and renders independent lane panels.

**Tech Stack:** Python 3.13, dataclasses, Pydantic, FastAPI, asyncio, multiprocessing spawn, sounddevice, NumPy, pytest, HTML/CSS/vanilla JavaScript.

## Global Constraints

- Preserve the current main-owned single `llama-multi-asr`, lane-process isolation, shared Qwen timing schema, Zipformer configuration, Qwen request payload, and 100 ms producer cadence.
- Do not change `OnlineFrameQueue(maxsize=1)`, Qwen core/thread allocation, `n_parallel`, `n_ctx`, or endpoint rules.
- Use only the current worktree-local Qwen model directories; no API may accept a missing model file.
- Accept only `qwen3-asr-0.6b-q40` and `qwen3-asr-1.7b-q4km` as offline model identifiers.
- The 0.6B model requires q40 GGUF plus its two dynq encoder files and uses dimensions `896/1024/1024`.
- The 1.7B model requires q4km GGUF plus its two dynq encoder files and uses dimensions `1024/2048/2048`.
- A user-visible model validation error must prevent the browser from calling `/api/models/load`; the server must repeat the same validation.
- Supported lane inputs are exactly `default_files`, `local_wav`, and `microphone`.
- `default_files` preserves sorted WAV discovery and cyclic assignment from `AppConfig.input_dir`.
- Microphone audio is mono float32 at 16 kHz, emitted as one 100 ms `AudioFrame` per block with consecutive frame indexes and exactly one finish frame on stop.
- Configuration is editable only while the manager is `unloaded`; loaded/running configurations are frozen.
- Do not commit model files, `.venv313`, `runs/`, the user-owned untracked task-note file, or board-local artifacts.
- Board validation uses the project-local Python 3.13 `.venv313`; do not run the full multi-run performance benchmark.

---

## File Structure

Create:

- `tests/test_microphone_source.py`: deterministic callback/queue tests for microphone frames and stop semantics.

Modify:

- `src/multi_stream_asr/config.py`: offline model catalog, immutable lane input configuration, default lane input expansion, and model-file validation helpers.
- `src/multi_stream_asr/asr/offline/shared_server.py`: derive required files and SMT config dimensions from the selected offline model.
- `src/multi_stream_asr/audio/mic.py`: real `sounddevice.InputStream` producer implementation.
- `src/multi_stream_asr/pipeline/process_manager.py`: resolve each configured lane input into a JSON-safe start command.
- `src/multi_stream_asr/pipeline/lane_runtime.py`: create the configured local or microphone source from the command descriptor.
- `src/multi_stream_asr/web/server.py`: configuration request schemas, model preflight API, device listing API, and conversion to a new immutable `AppConfig` before load.
- `src/multi_stream_asr/web/static/index.html`: control state machine, editable configuration draft, validation errors, lane input controls, fixed grids, and independent results scrolling.
- `tests/test_config.py`, `tests/test_shared_server.py`, `tests/test_process_manager.py`, `tests/test_lane_runtime.py`, `tests/test_web_server.py`: focused regression coverage.
- `README.md`: Web configuration and microphone prerequisites.

---

### Task 1: Add Offline Model and Lane Input Configuration

**Files:**
- Modify: `src/multi_stream_asr/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces `OfflineModelSpec`, `OFFLINE_MODELS`, `LaneInputConfig`, `AppConfig.lane_inputs`, `AppConfig.input_for_lane(stream_id)`, and `validate_offline_model_files(spec)`.
- `OfflineModelSpec` contains `model_id`, `model_dir`, `text_model_filename`, `frontend_filename`, `backend_filename`, `d_model`, `hidden_size`, and `output_dim`.
- `LaneInputConfig` contains `kind`, `local_wav_path`, and `microphone_device`.

- [ ] **Step 1: Write failing configuration tests**

Add these tests to `tests/test_config.py`:

```python
from dataclasses import replace

from multi_stream_asr.config import (
    AppConfig,
    LaneInputConfig,
    ModelConfig,
    OFFLINE_MODELS,
    validate_offline_model_files,
)


def test_offline_model_catalog_contains_q40_and_q4km(tmp_path: Path):
    q40 = OFFLINE_MODELS["qwen3-asr-0.6b-q40"]
    q4km = OFFLINE_MODELS["qwen3-asr-1.7b-q4km"]

    assert q40.text_model_filename == "Qwen3-ASR-0.6B-text-q40.gguf"
    assert (q40.d_model, q40.hidden_size, q40.output_dim) == (896, 1024, 1024)
    assert q4km.text_model_filename == "Qwen3-ASR-1.7B-text-q4km.gguf"
    assert (q4km.d_model, q4km.hidden_size, q4km.output_dim) == (1024, 2048, 2048)


def test_app_config_expands_default_file_inputs_for_every_lane():
    config = AppConfig(lanes=2)

    assert config.input_for_lane(0) == LaneInputConfig(kind="default_files")
    assert config.input_for_lane(1) == LaneInputConfig(kind="default_files")


def test_lane_input_rejects_missing_local_path_and_invalid_kind():
    with pytest.raises(ValueError, match="local_wav requires local_wav_path"):
        LaneInputConfig(kind="local_wav")
    with pytest.raises(ValueError, match="unsupported lane input kind"):
        LaneInputConfig(kind="network")


def test_validate_offline_model_files_returns_all_missing_paths(tmp_path: Path):
    spec = replace(OFFLINE_MODELS["qwen3-asr-0.6b-q40"], model_dir=tmp_path)

    missing = validate_offline_model_files(spec)

    assert missing == [tmp_path / name for name in spec.required_filenames]
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv313/bin/python -m pytest tests/test_config.py -k 'offline_model_catalog or expands_default or lane_input_rejects or validate_offline' -v
```

Expected: collection fails because the new configuration symbols do not exist.

- [ ] **Step 3: Implement immutable catalog and input configuration**

Add to `src/multi_stream_asr/config.py` after the path constants:

```python
@dataclass(frozen=True)
class OfflineModelSpec:
    model_id: str
    model_dir: Path
    text_model_filename: str
    frontend_filename: str
    backend_filename: str
    d_model: int
    hidden_size: int
    output_dim: int

    @property
    def required_filenames(self) -> tuple[str, str, str]:
        return (
            self.text_model_filename,
            self.frontend_filename,
            self.backend_filename,
        )


OFFLINE_MODELS = {
    "qwen3-asr-0.6b-q40": OfflineModelSpec(
        model_id="qwen3-asr-0.6b-q40",
        model_dir=PROJECT_ROOT / "models/asr/qwen3asr",
        text_model_filename="Qwen3-ASR-0.6B-text-q40.gguf",
        frontend_filename="Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
        backend_filename="Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
        d_model=896,
        hidden_size=1024,
        output_dim=1024,
    ),
    "qwen3-asr-1.7b-q4km": OfflineModelSpec(
        model_id="qwen3-asr-1.7b-q4km",
        model_dir=PROJECT_ROOT / "models/asr/qwen3asr-1.7b",
        text_model_filename="Qwen3-ASR-1.7B-text-q4km.gguf",
        frontend_filename="Qwen3-ASR-1.7B-encoder-frontend.dynq.onnx",
        backend_filename="Qwen3-ASR-1.7B-encoder-backend.dynq.onnx",
        d_model=1024,
        hidden_size=2048,
        output_dim=2048,
    ),
}


@dataclass(frozen=True)
class LaneInputConfig:
    kind: str = "default_files"
    local_wav_path: Path | None = None
    microphone_device: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"default_files", "local_wav", "microphone"}:
            raise ValueError(f"unsupported lane input kind: {self.kind}")
        if self.kind == "local_wav" and self.local_wav_path is None:
            raise ValueError("local_wav requires local_wav_path")
        if self.kind != "local_wav" and self.local_wav_path is not None:
            raise ValueError("only local_wav accepts local_wav_path")
        if self.kind != "microphone" and self.microphone_device is not None:
            raise ValueError("only microphone accepts microphone_device")


def validate_offline_model_files(spec: OfflineModelSpec) -> list[Path]:
    return [
        spec.model_dir / filename
        for filename in spec.required_filenames
        if not (spec.model_dir / filename).is_file()
    ]
```

Import `replace` from `dataclasses`. Change `ModelConfig` to hold `offline_model_id` and derive its selected spec:

```python
@dataclass(frozen=True)
class ModelConfig:
    online_model_dir: Path = DEFAULT_ONLINE_MODEL_DIR
    offline_model_id: str = "qwen3-asr-0.6b-q40"
    offline_model_dir: Path = DEFAULT_OFFLINE_MODEL_DIR
    online_num_threads: int = 1

    @property
    def offline_model(self) -> OfflineModelSpec:
        template = OFFLINE_MODELS[self.offline_model_id]
        return replace(template, model_dir=self.offline_model_dir)
```

Add `lane_inputs: tuple[LaneInputConfig, ...] = ()` to `AppConfig`, validate that it is empty or exactly `lanes` long, and add:

```python
def input_for_lane(self, stream_id: int) -> LaneInputConfig:
    if self.lane_inputs:
        return self.lane_inputs[stream_id]
    return LaneInputConfig()
```

- [ ] **Step 4: Run focused configuration tests and format check**

Run:

```bash
.venv313/bin/python -m pytest tests/test_config.py -q
.venv313/bin/python -m compileall -q src
git diff --check
```

Expected: all configuration tests pass and no compile or whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/config.py tests/test_config.py
git commit -m "feat: add selectable offline models and lane inputs"
```

---

### Task 2: Make Shared Qwen Configuration Model-Specific

**Files:**
- Modify: `src/multi_stream_asr/asr/offline/shared_server.py`
- Modify: `tests/test_shared_server.py`

**Interfaces:**
- Consumes `config.models.offline_model` from Task 1.
- Produces `build_shared_config(config)` and `build_launch_command(config, config_dir)` using selected filenames and dimensions.

- [ ] **Step 1: Write failing 1.7B tests**

Add to `tests/test_shared_server.py`:

```python
def test_shared_config_uses_q4km_1_7b_filenames_and_dimensions(tmp_path: Path):
    config = AppConfig(
        models=ModelConfig(
            offline_model_id="qwen3-asr-1.7b-q4km",
            offline_model_dir=tmp_path,
        )
    )

    payload = build_shared_config(config)

    assert payload["audio_model"]["frontend_model_path"] == str(
        tmp_path / "Qwen3-ASR-1.7B-encoder-frontend.dynq.onnx"
    )
    assert payload["audio_model"]["backend_model_path"] == str(
        tmp_path / "Qwen3-ASR-1.7B-encoder-backend.dynq.onnx"
    )
    assert payload["audio_model"]["d_model"] == 1024
    assert payload["audio_model"]["hidden_size"] == 2048
    assert payload["audio_model"]["output_dim"] == 2048
    assert payload["text_model"]["model_path"] == str(
        tmp_path / "Qwen3-ASR-1.7B-text-q4km.gguf"
    )
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
.venv313/bin/python -m pytest tests/test_shared_server.py::test_shared_config_uses_q4km_1_7b_filenames_and_dimensions -v
```

Expected: FAIL because `shared_server.py` still hard-codes 0.6B filenames and dimensions.

- [ ] **Step 3: Replace hard-coded model metadata with selected spec**

In `build_shared_config`, assign `model = config.models.offline_model` and replace the fixed paths and dimensions:

```python
"frontend_model_path": str(model.model_dir / model.frontend_filename),
"backend_model_path": str(model.model_dir / model.backend_filename),
"d_model": model.d_model,
"hidden_size": model.hidden_size,
"output_dim": model.output_dim,
...
"model_path": str(model.model_dir / model.text_model_filename),
"hidden_size": model.hidden_size,
```

In `build_launch_command`, use `model.model_dir / model.text_model_filename`. In
`_validate_files`, iterate `config.models.offline_model.required_filenames` rather than fixed
constants. Keep all existing shared thread, affinity, health, stale-cleanup, `--n-parallel`, and
`--n-ctx` behavior unchanged.

- [ ] **Step 4: Run shared-server regression tests**

Run:

```bash
.venv313/bin/python -m pytest tests/test_shared_server.py -q
```

Expected: all shared-server tests pass for both model selections.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/asr/offline/shared_server.py tests/test_shared_server.py
git commit -m "feat: configure shared qwen from selected model"
```

---

### Task 3: Implement Microphone Capture and Lane Source Descriptors

**Files:**
- Create: `tests/test_microphone_source.py`
- Modify: `src/multi_stream_asr/audio/mic.py`
- Modify: `src/multi_stream_asr/pipeline/process_manager.py`
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py`
- Modify: `tests/test_process_manager.py`
- Modify: `tests/test_lane_runtime.py`

**Interfaces:**
- `MicrophoneSource(stream_id, device, frame_duration_ms)` implements `AudioSource`.
- Start commands contain exactly one source descriptor: `input_kind=local_wav` plus `input_path`, or `input_kind=microphone` plus optional `microphone_device`.
- `LaneRuntime._build_source()` dispatches on `input_kind`.

- [ ] **Step 1: Write failing microphone source tests**

Create `tests/test_microphone_source.py` with a fake `sounddevice.InputStream` whose constructor
captures `samplerate`, `channels`, `dtype`, `blocksize`, `device`, and callback, and whose
`emit(samples)` invokes that callback. Include:

```python
@pytest.mark.asyncio
async def test_microphone_source_emits_float32_100ms_frames_and_one_finish(fake_sounddevice):
    source = MicrophoneSource(stream_id=2, device=7, frame_duration_ms=100)
    await source.prepare()
    fake_sounddevice.stream.emit(np.ones((1600, 1), dtype=np.float32))
    await source.stop()

    frames = [frame async for frame in source.frames()]

    assert fake_sounddevice.stream.kwargs == {
        "device": 7, "samplerate": 16000, "channels": 1,
        "dtype": "float32", "blocksize": 1600,
    }
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert frames[0].pcm.dtype == np.float32
    assert frames[0].pcm.shape == (1600,)
    assert [frame.is_finish for frame in frames] == [False, True]


@pytest.mark.asyncio
async def test_microphone_source_converts_multichannel_callback_to_mono(fake_sounddevice):
    source = MicrophoneSource(stream_id=0, device=None, frame_duration_ms=100)
    await source.prepare()
    fake_sounddevice.stream.emit(np.column_stack((np.zeros(1600), np.ones(1600))))
    await source.stop()

    frames = [frame async for frame in source.frames()]

    assert np.allclose(frames[0].pcm, 0.5)
```

Extend `tests/test_process_manager.py` to assert mixed source configuration produces commands:

```python
assert commands[0]["input_kind"] == "local_wav"
assert commands[0]["input_path"] == str(explicit_wav.resolve())
assert commands[1] == {
    "type": "start", "run_dir": str(run_dir),
    "input_kind": "microphone", "microphone_device": None,
}
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv313/bin/python -m pytest tests/test_microphone_source.py tests/test_process_manager.py -k 'microphone or mixed_source' -v
```

Expected: FAIL because microphone frames are unimplemented and start commands only contain
`input_path`.

- [ ] **Step 3: Implement `MicrophoneSource` with a callback-to-async queue bridge**

Replace `src/multi_stream_asr/audio/mic.py` with an implementation that:

```python
class MicrophoneSource:
    def __init__(self, stream_id: int, device: int | None = None, frame_duration_ms: int = 100):
        self.stream_id = stream_id
        self.device = device
        self.frame_duration_ms = frame_duration_ms
        self._sounddevice = _require_sounddevice(...)
        self._queue: asyncio.Queue[np.ndarray | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream = None
        self._stopped = False

    async def prepare(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        blocksize = 16000 * self.frame_duration_ms // 1000
        self._stream = self._sounddevice.InputStream(
            device=self.device, samplerate=16000, channels=1,
            dtype="float32", blocksize=blocksize, callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, RuntimeError(str(status)))
            return
        pcm = np.asarray(indata, dtype=np.float32)
        mono = pcm if pcm.ndim == 1 else pcm.mean(axis=1, dtype=np.float32)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, mono.copy())
```

`frames()` must raise a callback error, emit each queued buffer as an `AudioFrame` with sequential
index/timestamp, and emit exactly one finish frame after the queue sentinel. `stop()` must be
idempotent, stop/close the stream, and enqueue the sentinel through `call_soon_threadsafe`.

- [ ] **Step 4: Serialize and consume lane input descriptors**

Replace `_resolve_input_paths()` with `_resolve_input_commands()` in `ProcessPipelineManager`.
For each `config.input_for_lane(stream_id)`:

```python
if source.kind == "microphone":
    command.update({"input_kind": "microphone", "microphone_device": source.microphone_device})
else:
    path = explicit_local_path if source.kind == "local_wav" else default_paths[stream_id]
    command.update({"input_kind": "local_wav", "input_path": str(path.resolve(strict=True))})
```

Resolve and validate all local WAV paths before sending any start command. In
`LaneRuntime._build_source()`, select:

```python
if command["input_kind"] == "microphone":
    return MicrophoneSource(self.lane.stream_id, command.get("microphone_device"), self.app_config.frame_duration_ms)
return LocalWavSource(..., realtime=True)
```

Reject unknown/malformed descriptors with `ValueError` so the lane reports fatal rather than
silently choosing a source.

- [ ] **Step 5: Run audio, process, and lane source tests**

Run:

```bash
.venv313/bin/python -m pytest tests/test_microphone_source.py tests/test_local_file_audio.py tests/test_process_manager.py tests/test_lane_runtime.py -q
```

Expected: all source creation, command serialization, stop, and existing local-WAV tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/multi_stream_asr/audio/mic.py src/multi_stream_asr/pipeline/process_manager.py \
  src/multi_stream_asr/pipeline/lane_runtime.py tests/test_microphone_source.py \
  tests/test_process_manager.py tests/test_lane_runtime.py
git commit -m "feat: add configurable microphone lane input"
```

---

### Task 4: Add Web Configuration and Validation APIs

**Files:**
- Modify: `src/multi_stream_asr/web/server.py`
- Modify: `tests/test_web_server.py`

**Interfaces:**
- `GET /api/audio-devices` returns `{"devices": [...]}` from `list_microphone_devices()`.
- `POST /api/models/validate` accepts `offline_model_id` and `offline_model_dir`, returns
  `{"valid": bool, "missing_files": [str, ...]}`.
- `POST /api/models/load` accepts `lanes`, offline model selection, and ordered lane inputs.

- [ ] **Step 1: Write failing Web API tests**

Add tests to `tests/test_web_server.py`:

```python
def test_model_validation_reports_missing_files_without_creating_manager(tmp_path, monkeypatch):
    created = []
    app = create_app(manager_factory=lambda config, bus: created.append(config) or RecordingManager())
    with TestClient(app) as client:
        response = client.post("/api/models/validate", json={
            "offline_model_id": "qwen3-asr-1.7b-q4km",
            "offline_model_dir": str(tmp_path / "missing"),
        })
    assert response.json() == {
        "valid": False,
        "missing_files": [
            str(tmp_path / "missing/Qwen3-ASR-1.7B-text-q4km.gguf"),
            str(tmp_path / "missing/Qwen3-ASR-1.7B-encoder-frontend.dynq.onnx"),
            str(tmp_path / "missing/Qwen3-ASR-1.7B-encoder-backend.dynq.onnx"),
        ],
    }
    assert created == []


def test_load_rejects_missing_model_files_before_manager_load(tmp_path):
    manager = RecordingManager()
    app = create_app(config=AppConfig(), manager=manager)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/models/load", json={
            "offline_model_id": "qwen3-asr-0.6b-q40",
            "offline_model_dir": str(tmp_path / "missing"),
            "lane_inputs": [{"kind": "default_files"}] * 4,
        })
    assert response.status_code == 422
    assert manager.calls == []


def test_audio_devices_returns_enumerated_input_devices(monkeypatch):
    monkeypatch.setattr(server, "list_microphone_devices", lambda: [{"index": 2, "name": "USB"}])
    with TestClient(create_app(manager=RecordingManager())) as client:
        assert client.get("/api/audio-devices").json() == {"devices": [{"index": 2, "name": "USB"}]}
```

- [ ] **Step 2: Run the API tests and verify RED**

Run:

```bash
.venv313/bin/python -m pytest tests/test_web_server.py -k 'model_validation or missing_model or audio_devices' -v
```

Expected: FAIL because the request fields and two API routes do not exist.

- [ ] **Step 3: Define request schemas and conversion helpers**

Add Pydantic models:

```python
class LaneInputRequest(BaseModel):
    kind: Literal["default_files", "local_wav", "microphone"]
    local_wav_path: Path | None = None
    microphone_device: int | None = None


class ModelSelectionRequest(BaseModel):
    offline_model_id: Literal["qwen3-asr-0.6b-q40", "qwen3-asr-1.7b-q4km"] | None = None
    offline_model_dir: Path | None = None


class LoadModelsRequest(ModelSelectionRequest):
    lanes: int | None = None
    lane_inputs: list[LaneInputRequest] | None = None
```

Implement `_config_from_request(base, request)` to use the current `base.models` values when the
optional selection fields are absent, otherwise resolve the supplied directory and create
`ModelConfig(offline_model_id=..., offline_model_dir=...)`; then convert requests to
`tuple[LaneInputConfig, ...]`, verify the input count equals lanes when present, and return
`replace(base, lanes=..., models=..., lane_inputs=...)`.

`/api/models/validate` requires both selection fields and returns a 200 payload for both valid and
invalid directories. `/api/models/load` uses the same helper, preserves the existing empty-body
default-load behavior, raises `HTTPException(422, detail={"missing_files": [...]})` before creating
or loading a manager, and only permits configuration replacement while current state is `UNLOADED`.
`/api/audio-devices` calls `list_microphone_devices`; turn a missing sounddevice RuntimeError into
HTTP 503 with its message.

- [ ] **Step 4: Run Web API regression tests**

Run:

```bash
.venv313/bin/python -m pytest tests/test_web_server.py -q
```

Expected: existing lifecycle tests and new preflight/device tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/web/server.py tests/test_web_server.py
git commit -m "feat: add web model and input validation APIs"
```

---

### Task 5: Implement the Web Configuration Workspace and Result Panels

**Files:**
- Modify: `src/multi_stream_asr/web/static/index.html`
- Modify: `tests/test_web_server.py`

**Interfaces:**
- Browser calls `/api/models/validate`, `/api/audio-devices`, and `/api/models/load` from Task 4.
- Browser sends `{lanes, offline_model_id, offline_model_dir, lane_inputs}` to `/api/models/load` only after a valid preflight.

- [ ] **Step 1: Write failing static-page assertions**

Add assertions to `tests/test_web_server.py` for the required markup and named functions:

```python
assert 'id="offline-model"' in response.text
assert 'id="offline-model-dir"' in response.text
assert 'id="configuration-error"' in response.text
assert 'function renderLaneInputControls()' in response.text
assert 'function updateControlAvailability()' in response.text
assert 'function validateModelSelection()' in response.text
assert 'function scrollResultsIfPinnedToBottom' in response.text
assert '.lane-grid[data-lanes="6"]' in response.text
assert '.lane .results {' in response.text
assert 'overflow-y: auto;' in response.text
```

Add a control-state assertion that requires explicit disabled assignments for all four action buttons,
not only `laneSelect.disabled`.

- [ ] **Step 2: Run the assertions and verify RED**

Run:

```bash
.venv313/bin/python -m pytest tests/test_web_server.py -k 'index_html' -v
```

Expected: FAIL because the current page has no configuration controls, validation function, or
per-result scrolling behavior.

- [ ] **Step 3: Replace the static page with a state-driven configuration UI**

In `index.html`, add these elements above the lane grid:

```html
<section id="configuration-panel" class="configuration-panel">
  <label>路数 <select id="lane-count">...</select></label>
  <label>整段模型 <select id="offline-model">
    <option value="qwen3-asr-0.6b-q40">Qwen3-ASR 0.6B q40</option>
    <option value="qwen3-asr-1.7b-q4km">Qwen3-ASR 1.7B q4km</option>
  </select></label>
  <label>模型目录 <input id="offline-model-dir" type="text" /></label>
  <div id="lane-input-config"></div>
</section>
<p id="operation-status" role="status"></p>
<pre id="configuration-error" class="hidden" role="alert"></pre>
```

Keep a `draft` object in `appState` with `offlineModelId`, `offlineModelDir`, and one lane input
object per lane. `renderLaneInputControls()` must create one row per lane containing a source type
select. It reveals exactly one of these controls:

```html
<input class="local-wav-path" type="text" />
<select class="microphone-device"></select>
```

When microphone is selected, fetch `/api/audio-devices` once per unloaded editing session and
populate the device selector with an automatic option whose value is empty. When default files is
selected, hide both controls.

`validateModelSelection()` posts the selected ID and directory to `/api/models/validate`. On
`valid: false`, it writes `missing_files.join("\n")` into `configuration-error`, returns `false`,
and does not invoke `postJson("/api/models/load", ...)`. On success it clears the error and returns
`true`.

Implement `updateControlAvailability()` with this exact state policy:

```javascript
const editable = appState.state === "unloaded";
const busy = ["loading", "unloading", "starting", "stopping"].includes(appState.state);
loadButton.disabled = !editable || busy;
unloadButton.disabled = !["loaded", "stopped", "fatal"].includes(appState.state) || busy;
startButton.disabled = !["loaded", "stopped"].includes(appState.state) || busy;
stopButton.disabled = !["starting", "running"].includes(appState.state) || busy;
for (const element of configurationPanel.querySelectorAll("input, select")) {
  element.disabled = !editable || busy;
}
```

All handlers await requests inside `try/catch`, set an operation message before the request, and
render `error.message` into the persistent error element on failure. The load handler must call
`validateModelSelection()` before posting the full draft configuration.

Use lane-grid attributes and CSS:

```css
.app-shell { height: 100dvh; overflow: hidden; }
.lane-grid { min-height: 0; overflow: hidden; }
.lane-grid[data-lanes="2"] { grid-template-columns: repeat(2, minmax(0, 1fr)); grid-template-rows: 1fr; }
.lane-grid[data-lanes="4"] { grid-template-columns: repeat(2, minmax(0, 1fr)); grid-template-rows: repeat(2, 1fr); }
.lane-grid[data-lanes="6"] { grid-template-columns: repeat(3, minmax(0, 1fr)); grid-template-rows: repeat(2, 1fr); }
.lane-grid[data-lanes="8"] { grid-template-columns: repeat(4, minmax(0, 1fr)); grid-template-rows: repeat(2, 1fr); }
.lane { min-height: 0; }
.lane .results { min-height: 0; overflow-y: auto; }
```

Before rebuilding one lane's result DOM, record:

```javascript
const pinned = results.scrollTop + results.clientHeight >= results.scrollHeight - 2;
```

After appending rows, set `results.scrollTop = results.scrollHeight` only when `pinned` was true.
Assign `laneGrid.dataset.lanes = String(appState.lanes)` whenever lane count changes.

- [ ] **Step 4: Run Web UI unit assertions**

Run:

```bash
.venv313/bin/python -m pytest tests/test_web_server.py -q
```

Expected: static-page assertions and existing API/lifecycle tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/web/static/index.html tests/test_web_server.py
git commit -m "feat: add configurable web asr controls"
```

---

### Task 6: Document and Validate the Complete Workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/thread-model.md`
- Test: local focused suite and board smoke commands

**Interfaces:**
- Documents the Web APIs, microphone dependency, model selection, and board-local model directories delivered by Tasks 1-5.

- [ ] **Step 1: Add README Web configuration instructions**

Add a Web-mode section documenting:

```text
1. Keep the service unloaded while editing lanes, model, model directory, and lane inputs.
2. Select Qwen3-ASR 0.6B q40 or 1.7B q4km; the browser preflights required local files.
3. Select default files, a 16 kHz mono WAV path, or a microphone for each lane.
4. Microphone input requires sounddevice and an available PortAudio input device.
5. Load models, start recognition, stop recognition, then unload before changing configuration.
```

Document the required local directories:

```text
models/asr/qwen3asr/
models/asr/qwen3asr-1.7b/
```

- [ ] **Step 2: Update the thread model for microphone producer behavior**

In `docs/thread-model.md`, add that a microphone lane uses a sounddevice callback producer feeding
the lane's 100 ms source stream; it does not move PCM through main-process IPC and sends a finish
frame exactly once on stop.

- [ ] **Step 3: Run local focused verification**

Run:

```bash
.venv313/bin/python -m compileall -q src
.venv313/bin/python -m pytest tests/test_config.py tests/test_shared_server.py tests/test_microphone_source.py tests/test_local_file_audio.py tests/test_process_manager.py tests/test_lane_runtime.py tests/test_web_server.py -q
.venv313/bin/python -m multi_stream_asr.main --headless --fake --lanes 4
```

Expected: compilation succeeds, all selected tests pass, and fake headless exits 0.

- [ ] **Step 4: Synchronize for board testing without committing models**

From the local worktree, synchronize code only:

```bash
rsync -a --delete \
  --exclude='.git/' --exclude='.venv313/' --exclude='models/' --exclude='runs/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  ./ bianbu@10.0.91.51:~/Workspace/projects/multi-stream-asr/
```

Do not synchronize model files; the board keeps its verified local model directories.

- [ ] **Step 5: Run board model smokes**

Run on the board after activating `.venv313`:

```bash
python scripts/smoke_shared_qwen.py ~/Music/wav16k/00_simple_news_16000hz_1ch.wav --seconds 10
```

Then run the 1.7B equivalent by constructing `AppConfig` with:

```python
ModelConfig(
    offline_model_id="qwen3-asr-1.7b-q4km",
    offline_model_dir=Path("models/asr/qwen3asr-1.7b").resolve(),
)
```

Start `SharedQwenServerManager`, issue one 10-second `Qwen3AsrClient` request, assert nonempty
text, and always stop the manager in `finally`. Verify `pgrep -ax llama-multi-asr` is empty after
each smoke.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md docs/thread-model.md
git commit -m "docs: describe web inputs and model selection"
```

---

## Plan Self-Review

- Spec coverage: Task 1 covers immutable configuration and file validation; Task 2 covers model-specific shared service configuration; Task 3 covers microphone capture and lane dispatch; Task 4 covers server APIs; Task 5 covers all missing browser controls, configuration visibility, fixed grids, scrolling, and errors; Task 6 covers documentation and board verification.
- No placeholders: every task names exact files, interfaces, tests, commands, and expected outcomes.
- Type consistency: the Web request becomes `LaneInputConfig`/`ModelConfig`, `ProcessPipelineManager` serializes `input_kind`, and `LaneRuntime` consumes that same descriptor.
