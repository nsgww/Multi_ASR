# Multi-Stream ASR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Python 3.13 multi-lane ASR service with WebSocket UI, headless mode, sherpa-onnx online recognition, Qwen3-ASR `llama-server` offline correction, and final real-device headless validation.

**Architecture:** Implement a layered application under `src/multi_stream_asr`: typed config and models first, then buffers, process/model managers, ASR adapters, pipeline orchestration, event bus, Web server, and headless runner. Tests use fake audio/online/offline adapters for deterministic coverage; real model adapters are isolated behind interfaces.

**Tech Stack:** Python 3.13, FastAPI, uvicorn, WebSocket, asyncio, pytest, httpx, numpy, standard-library `wave`, optional `sounddevice`, optional `sherpa-onnx`.

## Global Constraints

- Remote K3 development, validation, and inference must use Python 3.13 only.
- Remote K3 commands must use the project-local `.venv313` virtual environment.
- Dependency installation must activate `.venv313` and export `PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"` and `PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"` first.
- Supported lane counts are exactly `2`, `4`, `6`, and `8`; default is `4`.
- Audio input is 16 kHz mono; local files must be 16 kHz mono PCM WAV.
- Default frame duration is `100 ms`.
- Default online model path is `~/Workspace/sm-sdk/models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`.
- Default offline model path is `~/Workspace/sm-sdk/models/asr/qwen3asr`.
- Each sherpa-onnx online lane uses `num_threads=1`.
- Default zipformer endpoint values are `rule1_min_trailing_silence=1.0s`, `rule2_min_trailing_silence=0.5s`, and `rule3_min_utterance_length=10.0s`.
- Offline segment cache capacity is `1.5 * rule3_min_utterance_length`, default `15.0s`.
- Offline `llama-server` ports start at `8063`; each lane uses a dedicated server and fixed `id_slot=0`.
- A100 affinity is configured for sherpa-onnx and `llama-server`/SMT backends only; Python application threads must not implement explicit Linux thread binding.
- A100 default CPU set is `8,9,10,11,12,13,14,15`.
- `SPACEMIT_PERFER_CORE_ID` uses comma-separated cores; SMT ONNX `SPACEMIT_EP_INTRA_THREAD_AFFINITY` uses semicolon-separated cores.
- `llama-server -t` and `-tb` must equal the number of cores assigned to that lane.
- `SPACEMIT_EP_INTER_THREAD_NUM` is always `1`.
- `pending` is a frontend-visible and headless-visible event and must refresh sentence state.
- Offline empty string is a valid `final` result and must replace pending text.
- Unit and fake-adapter integration tests are required.
- Final validation after development is `python -m multi_stream_asr.main --headless --lanes 4 --input-dir ~/Music/wav16k`.
- Ideal final headless performance targets are online average frame latency approximately `50 ms` and offline RTF approximately `0.4`.

---

## File Structure

Create these files:

- `pyproject.toml`: package metadata, runtime dependencies, optional hardware dependencies, pytest config.
- `README.md`: setup, Web mode, headless mode, and validation commands.
- `src/multi_stream_asr/__init__.py`: package version.
- `src/multi_stream_asr/main.py`: CLI entry point for Web and headless modes.
- `src/multi_stream_asr/config.py`: typed config, defaults, lane/core allocation, path expansion.
- `src/multi_stream_asr/models.py`: dataclasses/enums for frames, ASR events, lane config, metrics, and result state.
- `src/multi_stream_asr/pipeline/state.py`: lifecycle states and legal transitions.
- `src/multi_stream_asr/buffer/audio_queue.py`: capacity-one online queue with drop accounting.
- `src/multi_stream_asr/buffer/segment_cache.py`: PCM segment cache bounded by seconds.
- `src/multi_stream_asr/log/metrics.py`: latency and RTF metrics.
- `src/multi_stream_asr/log/recorder.py`: JSON result/config/metric persistence.
- `src/multi_stream_asr/utils/paths.py`: `~` expansion and directory helpers.
- `src/multi_stream_asr/utils/time.py`: monotonic timestamp helpers.
- `src/multi_stream_asr/utils/process.py`: subprocess helper functions.
- `src/multi_stream_asr/asr/online/base.py`: online recognizer protocol and result type.
- `src/multi_stream_asr/asr/online/zipformer.py`: sherpa-onnx adapter.
- `src/multi_stream_asr/asr/offline/base.py`: offline recognizer protocol and result type.
- `src/multi_stream_asr/asr/offline/llama_server.py`: per-lane `llama-server` config, launch, health check, cleanup.
- `src/multi_stream_asr/asr/offline/qwen3asr.py`: Qwen3-ASR HTTP request, slot cleanup, response parsing.
- `src/multi_stream_asr/audio/base.py`: audio source protocol.
- `src/multi_stream_asr/audio/local_file.py`: WAV validation and frame generation.
- `src/multi_stream_asr/audio/mic.py`: microphone source wrapper.
- `src/multi_stream_asr/audio/device.py`: microphone enumeration.
- `src/multi_stream_asr/web/event_bus.py`: event queue, partial merging, WebSocket broadcast, internal result state.
- `src/multi_stream_asr/pipeline/pipeline.py`: one-lane orchestration.
- `src/multi_stream_asr/pipeline/manager.py`: global lifecycle manager.
- `src/multi_stream_asr/headless/runner.py`: headless app flow.
- `src/multi_stream_asr/web/server.py`: FastAPI app, routes, WebSocket endpoint.
- `src/multi_stream_asr/web/static/index.html`: static UI.
- `tests/conftest.py`: shared fixtures and fake adapters.
- `tests/test_config.py`
- `tests/test_models_state.py`
- `tests/test_buffers_metrics.py`
- `tests/test_llama_server.py`
- `tests/test_qwen3asr.py`
- `tests/test_local_file_audio.py`
- `tests/test_event_bus.py`
- `tests/test_pipeline.py`
- `tests/test_manager_headless.py`
- `tests/test_web_server.py`

Do not modify `docs/superpowers/specs/design.md`; it is the user's original design artifact.

---

### Task 1: Project Scaffold, Config, Models, and State

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/multi_stream_asr/__init__.py`
- Create: `src/multi_stream_asr/config.py`
- Create: `src/multi_stream_asr/models.py`
- Create: `src/multi_stream_asr/pipeline/state.py`
- Create: `tests/test_config.py`
- Create: `tests/test_models_state.py`

**Interfaces:**
- Produces: `AppConfig`, `EndpointConfig`, `ModelConfig`, `OfflineServerConfig`, `LaneConfig`, `CoreAllocation`, `build_lane_configs(lanes: int, base_port: int = 8063) -> list[LaneConfig]`.
- Produces: `AudioFrame`, `OnlineRecognition`, `ASREvent`, `EventStatus`, `ResultType`, `LaneMetrics`, `LaneFinalState`.
- Produces: `PipelineState`, `assert_transition(current: PipelineState, target: PipelineState) -> None`.

- [ ] **Step 1: Write config tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from multi_stream_asr.config import (
    AppConfig,
    EndpointConfig,
    build_lane_configs,
    core_allocations_for_lanes,
)


def test_default_config_values():
    config = AppConfig()

    assert config.lanes == 4
    assert config.frame_duration_ms == 100
    assert config.input_dir == Path("~/Music/wav16k").expanduser()
    assert config.models.online_num_threads == 1
    assert config.models.online_model_dir == Path(
        "~/Workspace/sm-sdk/models/asr/"
        "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
    ).expanduser()
    assert config.models.offline_model_dir == Path(
        "~/Workspace/sm-sdk/models/asr/qwen3asr"
    ).expanduser()
    assert config.endpoint == EndpointConfig(
        rule1_min_trailing_silence=1.0,
        rule2_min_trailing_silence=0.5,
        rule3_min_utterance_length=10.0,
    )
    assert config.segment_cache_seconds == 15.0
    assert config.offline.request_timeout_ms == 300000
    assert config.offline.slot_cleanup_timeout_ms == 30000


@pytest.mark.parametrize("lanes", [2, 4, 6, 8])
def test_supported_lane_counts(lanes):
    assert AppConfig(lanes=lanes).lanes == lanes


@pytest.mark.parametrize("lanes", [0, 1, 3, 5, 7, 9])
def test_rejects_unsupported_lane_counts(lanes):
    with pytest.raises(ValueError, match="lanes must be one of"):
        AppConfig(lanes=lanes)


def test_core_allocations_for_each_supported_lane_count():
    assert core_allocations_for_lanes(2) == [
        ("8,9,10,11", "8;9;10;11"),
        ("12,13,14,15", "12;13;14;15"),
    ]
    assert core_allocations_for_lanes(4) == [
        ("8,9", "8;9"),
        ("10,11", "10;11"),
        ("12,13", "12;13"),
        ("14,15", "14;15"),
    ]
    assert core_allocations_for_lanes(6) == [
        ("10", "10"),
        ("11", "11"),
        ("12", "12"),
        ("13", "13"),
        ("14", "14"),
        ("15", "15"),
    ]
    assert core_allocations_for_lanes(8) == [
        ("8", "8"),
        ("9", "9"),
        ("10", "10"),
        ("11", "11"),
        ("12", "12"),
        ("13", "13"),
        ("14", "14"),
        ("15", "15"),
    ]


def test_build_lane_configs_assigns_ports_and_threads():
    lanes = build_lane_configs(4, base_port=8063)

    assert [lane.stream_id for lane in lanes] == [0, 1, 2, 3]
    assert [lane.offline_port for lane in lanes] == [8063, 8064, 8065, 8066]
    assert [lane.gguf_cores for lane in lanes] == ["8,9", "10,11", "12,13", "14,15"]
    assert [lane.onnx_cores for lane in lanes] == ["8;9", "10;11", "12;13", "14;15"]
    assert [lane.thread_count for lane in lanes] == [2, 2, 2, 2]
```

- [ ] **Step 2: Write model and state tests**

Create `tests/test_models_state.py`:

```python
import numpy as np
import pytest

from multi_stream_asr.models import (
    ASREvent,
    AudioFrame,
    EventStatus,
    LaneFinalState,
    LaneMetrics,
    ResultType,
)
from multi_stream_asr.pipeline.state import PipelineState, assert_transition


def test_audio_frame_finish_has_empty_pcm():
    frame = AudioFrame.finish(stream_id=1, frame_index=42, timestamp_ms=1000)

    assert frame.stream_id == 1
    assert frame.frame_index == 42
    assert frame.is_finish is True
    assert frame.pcm.dtype == np.float32
    assert frame.pcm.size == 0


def test_asr_event_final_flag_matches_status():
    pending = ASREvent.asr_result(
        stream_id=0,
        sentence_id=3,
        revision=2,
        status=EventStatus.PENDING,
        text="hello",
        timestamp_ms=123,
    )
    final = ASREvent.asr_result(
        stream_id=0,
        sentence_id=3,
        revision=3,
        status=EventStatus.FINAL,
        text="hello world",
        timestamp_ms=124,
    )

    assert pending.result_type == ResultType.ONLINE
    assert pending.is_final is False
    assert final.result_type == ResultType.OFFLINE
    assert final.is_final is True


def test_lane_metrics_defaults():
    metrics = LaneMetrics(stream_id=2)

    assert metrics.processed_frames == 0
    assert metrics.dropped_frames == 0
    assert metrics.offline_requests == 0
    assert metrics.final_state == LaneFinalState.STOPPED


def test_state_allows_normal_lifecycle():
    assert_transition(PipelineState.UNLOADED, PipelineState.LOADING)
    assert_transition(PipelineState.LOADING, PipelineState.LOADED)
    assert_transition(PipelineState.LOADED, PipelineState.STARTING)
    assert_transition(PipelineState.STARTING, PipelineState.RUNNING)
    assert_transition(PipelineState.RUNNING, PipelineState.STOPPING)
    assert_transition(PipelineState.STOPPING, PipelineState.STOPPED)
    assert_transition(PipelineState.STOPPED, PipelineState.UNLOADING)
    assert_transition(PipelineState.UNLOADING, PipelineState.UNLOADED)


def test_state_rejects_illegal_transition():
    with pytest.raises(ValueError, match="illegal pipeline state transition"):
        assert_transition(PipelineState.UNLOADED, PipelineState.RUNNING)
```

- [ ] **Step 3: Run tests and confirm they fail because package is absent**

Run:

```bash
pytest tests/test_config.py tests/test_models_state.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'multi_stream_asr'`.

- [ ] **Step 4: Create package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "multi-stream-asr"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "httpx>=0.27",
  "numpy>=2.0",
  "pydantic>=2.8",
  "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3",
  "pytest-asyncio>=0.23",
]
hardware = [
  "sounddevice>=0.5",
  "sherpa-onnx>=1.10",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"multi_stream_asr.web" = ["static/*.html"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

Create `src/multi_stream_asr/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `README.md`:

```markdown
# Multi-Stream ASR

Python 3.13 application for 2/4/6/8-lane online and offline ASR on K3/A100.

## Remote setup

```bash
python3.13 -m venv .venv313
. .venv313/bin/activate
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Headless validation

```bash
. .venv313/bin/activate
python -m multi_stream_asr.main --headless --lanes 4 --input-dir ~/Music/wav16k
```

Target final metrics: online average frame latency about 50 ms and offline RTF about 0.4.
```

- [ ] **Step 5: Implement config**

Create `src/multi_stream_asr/config.py` with these public names:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_LANES = (2, 4, 6, 8)
DEFAULT_ONLINE_MODEL_DIR = Path(
    "~/Workspace/sm-sdk/models/asr/"
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
).expanduser()
DEFAULT_OFFLINE_MODEL_DIR = Path("~/Workspace/sm-sdk/models/asr/qwen3asr").expanduser()
DEFAULT_INPUT_DIR = Path("~/Music/wav16k").expanduser()


@dataclass(frozen=True)
class EndpointConfig:
    rule1_min_trailing_silence: float = 1.0
    rule2_min_trailing_silence: float = 0.5
    rule3_min_utterance_length: float = 10.0


@dataclass(frozen=True)
class ModelConfig:
    online_model_dir: Path = DEFAULT_ONLINE_MODEL_DIR
    offline_model_dir: Path = DEFAULT_OFFLINE_MODEL_DIR
    online_num_threads: int = 1


@dataclass(frozen=True)
class OfflineServerConfig:
    base_port: int = 8063
    request_timeout_ms: int = 300000
    slot_cleanup_timeout_ms: int = 30000
    host: str = "127.0.0.1"


@dataclass(frozen=True)
class CoreAllocation:
    gguf_cores: str
    onnx_cores: str

    @property
    def thread_count(self) -> int:
        return len([core for core in self.gguf_cores.split(",") if core])


@dataclass(frozen=True)
class LaneConfig:
    stream_id: int
    offline_port: int
    gguf_cores: str
    onnx_cores: str
    thread_count: int


@dataclass(frozen=True)
class AppConfig:
    lanes: int = 4
    frame_duration_ms: int = 100
    input_dir: Path = DEFAULT_INPUT_DIR
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    offline: OfflineServerConfig = field(default_factory=OfflineServerConfig)
    output_dir: Path = Path("runs").resolve()
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.lanes not in SUPPORTED_LANES:
            raise ValueError(f"lanes must be one of {SUPPORTED_LANES}, got {self.lanes}")

    @property
    def segment_cache_seconds(self) -> float:
        return 1.5 * self.endpoint.rule3_min_utterance_length

    @property
    def lane_configs(self) -> list[LaneConfig]:
        return build_lane_configs(self.lanes, self.offline.base_port)


def core_allocations_for_lanes(lanes: int) -> list[tuple[str, str]]:
    if lanes == 2:
        return [("8,9,10,11", "8;9;10;11"), ("12,13,14,15", "12;13;14;15")]
    if lanes == 4:
        return [("8,9", "8;9"), ("10,11", "10;11"), ("12,13", "12;13"), ("14,15", "14;15")]
    if lanes == 6:
        return [(str(core), str(core)) for core in range(10, 16)]
    if lanes == 8:
        return [(str(core), str(core)) for core in range(8, 16)]
    raise ValueError(f"lanes must be one of {SUPPORTED_LANES}, got {lanes}")


def build_lane_configs(lanes: int, base_port: int = 8063) -> list[LaneConfig]:
    configs: list[LaneConfig] = []
    for stream_id, (gguf_cores, onnx_cores) in enumerate(core_allocations_for_lanes(lanes)):
        allocation = CoreAllocation(gguf_cores=gguf_cores, onnx_cores=onnx_cores)
        configs.append(
            LaneConfig(
                stream_id=stream_id,
                offline_port=base_port + stream_id,
                gguf_cores=allocation.gguf_cores,
                onnx_cores=allocation.onnx_cores,
                thread_count=allocation.thread_count,
            )
        )
    return configs
```

- [ ] **Step 6: Implement models and state**

Create `src/multi_stream_asr/models.py` with dataclasses/enums used by tests. Create `src/multi_stream_asr/pipeline/state.py` with `PipelineState` and `assert_transition`. Keep enum values exactly as strings used in tests: `partial`, `pending`, `final`, `fatal_error`, `lane_finished`, `stats_updated`, `online`, `offline`, `finished`, `stopped`, `fatal`.

- [ ] **Step 7: Run tests**

Run:

```bash
pytest tests/test_config.py tests/test_models_state.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml README.md src/multi_stream_asr tests/test_config.py tests/test_models_state.py
git commit -m "feat: add project config and shared models"
```

---

### Task 2: Buffers, Metrics, Recorder, and Utilities

**Files:**
- Create: `src/multi_stream_asr/buffer/audio_queue.py`
- Create: `src/multi_stream_asr/buffer/segment_cache.py`
- Create: `src/multi_stream_asr/log/metrics.py`
- Create: `src/multi_stream_asr/log/recorder.py`
- Create: `src/multi_stream_asr/utils/paths.py`
- Create: `src/multi_stream_asr/utils/time.py`
- Create: `src/multi_stream_asr/utils/process.py`
- Create: `tests/test_buffers_metrics.py`

**Interfaces:**
- Consumes: `AudioFrame`, `LaneMetrics`, `LaneFinalState`.
- Produces: `OnlineFrameQueue(maxsize: int = 1)`, `SegmentCache(sample_rate: int, capacity_seconds: float)`.
- Produces: `LatencyTracker`, `RtfTracker`, `RunRecorder`.

- [ ] **Step 1: Write buffer and metrics tests**

Create `tests/test_buffers_metrics.py` with tests for capacity-one replacement, drop count, segment cache capacity of 15 seconds at 16 kHz, reset behavior, average latency, RTF, and JSON recorder output.

- [ ] **Step 2: Run tests and confirm missing modules**

Run:

```bash
pytest tests/test_buffers_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `multi_stream_asr.buffer`.

- [ ] **Step 3: Implement utilities**

Create:

```python
# src/multi_stream_asr/utils/paths.py
from pathlib import Path


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_dir(path: str | Path) -> Path:
    resolved = expand_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
```

```python
# src/multi_stream_asr/utils/time.py
import time


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)
```

```python
# src/multi_stream_asr/utils/process.py
import subprocess


def run_best_effort(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)
```

- [ ] **Step 4: Implement `OnlineFrameQueue` and `SegmentCache`**

`OnlineFrameQueue.put(frame)` returns `True` if no drop happened and `False` if an old frame was replaced. Finish frames must be enqueued and must not be discarded behind normal audio frames. `SegmentCache.audio()` returns concatenated `float32` PCM.

- [ ] **Step 5: Implement metrics and recorder**

`LatencyTracker.add(duration_ms: float)` updates count/total/average. `RtfTracker.add(audio_seconds: float, elapsed_ms: float)` stores per-request RTF as `(elapsed_ms / 1000) / audio_seconds`, with zero-audio guarded as `0.0`. `RunRecorder.save(...)` writes `config.json`, `metrics.json`, final transcripts under `results/lane<N>.txt`, and model diagnostics under `model_outputs/lane<N>.jsonl` in a run directory.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_buffers_metrics.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/multi_stream_asr/buffer src/multi_stream_asr/log src/multi_stream_asr/utils tests/test_buffers_metrics.py
git commit -m "feat: add buffers metrics and recorder"
```

---

### Task 3: Offline `llama-server` Manager

**Files:**
- Create: `src/multi_stream_asr/asr/offline/llama_server.py`
- Create: `tests/test_llama_server.py`

**Interfaces:**
- Consumes: `AppConfig`, `LaneConfig`.
- Produces: `LlamaServerSpec`, `LlamaServerManager`.
- Produces: `build_smt_config(config: AppConfig, lane: LaneConfig) -> dict`.
- Produces: `build_launch_command(config: AppConfig, lane: LaneConfig, config_dir: Path) -> list[str]`.

- [ ] **Step 1: Write `llama-server` tests**

Create tests proving:

- generated config contains Qwen3-ASR audio/text/prompt fields and lane-local `ep_config`;
- model directory config is not modified;
- `SPACEMIT_PERFER_CORE_ID`, `-t`, `-tb`, `--smt-config-dir`, `--media-backend smt`, `--slot-save-path /tmp`, and `--warmup` are present;
- health success returns loaded specs;
- health timeout stops started processes.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/test_llama_server.py -v
```

Expected: FAIL because `multi_stream_asr.asr.offline.llama_server` is missing.

- [ ] **Step 3: Implement config rendering**

Implement `build_smt_config()` using the exact Qwen3-ASR fields from the spec, including frontend/backend ONNX paths, text GGUF path `Qwen3-ASR-0.6B-text-q40.gguf`, `context_size=65536`, `logical_batch_size=2048`, `predict_token=256`, and prompt tokens.

- [ ] **Step 4: Implement launch command and environment**

`LlamaServerSpec` must include `stream_id`, `port`, `config_dir`, `log_path`, `pid_path`, `env`, and `command`. `env["SPACEMIT_PERFER_CORE_ID"]` equals `lane.gguf_cores`. The command must use `-t` and `-tb` equal to `lane.thread_count`.

- [ ] **Step 5: Implement manager lifecycle**

`LlamaServerManager.cleanup_before_start()` executes the three cleanup operations from the spec. `start_all()` writes per-port configs under `/tmp/spacemit-qwen3asr-<port>`, starts each process, polls `/health` until OK or timeout, and stops started processes on failure. `stop_all()` terminates processes and escalates to kill after a timeout.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_llama_server.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/multi_stream_asr/asr/offline/llama_server.py tests/test_llama_server.py
git commit -m "feat: manage per-lane llama servers"
```

---

### Task 4: Qwen3-ASR Offline Client

**Files:**
- Create: `src/multi_stream_asr/asr/offline/base.py`
- Create: `src/multi_stream_asr/asr/offline/qwen3asr.py`
- Create: `tests/test_qwen3asr.py`

**Interfaces:**
- Consumes: `LaneConfig`, `OfflineServerConfig`.
- Produces: `OfflineRecognitionResult(text: str, elapsed_ms: float, audio_seconds: float, rtf: float)`.
- Produces: `Qwen3AsrClient.recognize(pcm: np.ndarray, prompt: str) -> OfflineRecognitionResult`.
- Produces: `encode_wav_base64(pcm: np.ndarray, sample_rate: int = 16000) -> str`.
- Produces: `clean_qwen_text(payload: dict) -> str`.

- [ ] **Step 1: Write offline client tests**

Create tests for WAV/base64 encoding, request JSON shape, text parsing from `choices[0].message.content`, fallback parsing from top-level `content` and `text`, `<asr_text>` cleanup, empty final preservation, hard failure on missing text, and slot cleanup failure.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/test_qwen3asr.py -v
```

Expected: FAIL because the offline client is missing.

- [ ] **Step 3: Implement base protocol and result**

Create an async `OfflineRecognizer` protocol with `recognize(pcm: np.ndarray, prompt: str) -> OfflineRecognitionResult` and `close() -> None`.

- [ ] **Step 4: Implement WAV/base64 and request body**

Use standard-library `wave` and `io.BytesIO`. Convert clipped float32 PCM `[-1.0, 1.0]` to signed 16-bit PCM before writing WAV. Request body must match the spec and use `id_slot=0`.

- [ ] **Step 5: Implement response parsing and cleanup**

If a request reaches the server, always call `/slots/0?action=erase` in a `finally` block. Raise a lane offline error when cleanup fails. Treat empty cleaned text as a successful final result.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_qwen3asr.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/multi_stream_asr/asr/offline tests/test_qwen3asr.py
git commit -m "feat: add qwen3 asr offline client"
```

---

### Task 5: Audio Input Implementations

**Files:**
- Create: `src/multi_stream_asr/audio/base.py`
- Create: `src/multi_stream_asr/audio/local_file.py`
- Create: `src/multi_stream_asr/audio/mic.py`
- Create: `src/multi_stream_asr/audio/device.py`
- Create: `tests/test_local_file_audio.py`

**Interfaces:**
- Consumes: `AudioFrame`, `AppConfig`.
- Produces: `AudioSource` protocol with `prepare()`, `frames()`, and `stop()`.
- Produces: `validate_wav_16k_mono_pcm(path: Path) -> WavInfo`.
- Produces: `select_input_files(input_dir: Path, lanes: int) -> list[Path]`.
- Produces: `LocalWavSource(stream_id: int, path: Path, frame_duration_ms: int = 100)`.

- [ ] **Step 1: Write local file tests**

Create tests that generate temporary WAV files and verify:

- 16 kHz mono PCM WAV passes validation;
- wrong sample rate fails;
- stereo fails;
- non-PCM encoding fails when represented by unsupported sample width;
- selected input files are sorted and cycle when fewer than lane count;
- `LocalWavSource.frames()` emits 100 ms frames and exactly one finish frame.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/test_local_file_audio.py -v
```

Expected: FAIL because audio modules are missing.

- [ ] **Step 3: Implement audio base protocol**

Use an async frame iterator interface:

```python
class AudioSource(Protocol):
    async def prepare(self) -> None: ...
    async def frames(self) -> AsyncIterator[AudioFrame]: ...
    async def stop(self) -> None: ...
```

- [ ] **Step 4: Implement local WAV validation and frame generation**

Read PCM WAV with `wave.open`. Convert signed 16-bit PCM to `float32` in `[-1.0, 1.0]`. For 100 ms at 16 kHz, emit frames of 1600 samples, incrementing `frame_index`. Emit a finish frame after data frames.

- [ ] **Step 5: Implement microphone placeholders**

`device.py` exposes `list_microphone_devices() -> list[dict[str, object]]`. `mic.py` exposes `MicrophoneSource` and raises a clear `RuntimeError` if optional `sounddevice` is unavailable.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_local_file_audio.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/multi_stream_asr/audio tests/test_local_file_audio.py
git commit -m "feat: add audio input sources"
```

---

### Task 6: Frontend Event Bus

**Files:**
- Create: `src/multi_stream_asr/web/event_bus.py`
- Create: `tests/test_event_bus.py`

**Interfaces:**
- Consumes: `ASREvent`.
- Produces: `FrontendEventBus.publish(event: ASREvent) -> None`.
- Produces: `FrontendEventBus.next_event() -> ASREvent`.
- Produces: `FrontendEventBus.register(websocket)`, `unregister(websocket)`, `broadcast_loop()`.
- Produces: `FrontendEventBus.snapshot_results() -> dict`.

- [ ] **Step 1: Write event bus tests**

Create async tests proving:

- newer revisions replace older text for the same `stream_id + sentence_id`;
- older revisions are ignored;
- `partial` events may merge when queued;
- `pending`, `final`, `fatal_error`, `lane_finished`, and `stats_updated` are not dropped;
- `pending` updates internal state;
- `final` empty string replaces pending text.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/test_event_bus.py -v
```

Expected: FAIL because `FrontendEventBus` is missing.

- [ ] **Step 3: Implement internal result state**

Use a dictionary keyed by `(stream_id, sentence_id)` storing latest `revision`, `text`, `status`, `result_type`, and `is_final`.

- [ ] **Step 4: Implement queue behavior**

Use `asyncio.Queue`. When publishing a replaceable `partial`, update a pending-partial map so that a later partial for the same sentence supersedes earlier text. Publish non-replaceable events directly.

- [ ] **Step 5: Implement WebSocket registration and broadcast**

`broadcast_loop()` reads events and sends JSON to all registered sockets. On send error, remove only that socket. Do not propagate socket failures into pipeline code.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_event_bus.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/multi_stream_asr/web/event_bus.py tests/test_event_bus.py
git commit -m "feat: add frontend event bus"
```

---

### Task 7: Online Recognizer Interface and One-Lane Pipeline

**Files:**
- Create: `src/multi_stream_asr/asr/online/base.py`
- Create: `src/multi_stream_asr/asr/online/zipformer.py`
- Create: `src/multi_stream_asr/pipeline/pipeline.py`
- Create: `tests/conftest.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `AudioFrame`, `SegmentCache`, `FrontendEventBus`, `OfflineRecognizer`.
- Produces: `OnlineRecognizer.accept(frame: AudioFrame) -> OnlineRecognition`.
- Produces: `OnlineRecognizer.reset() -> None`.
- Produces: `LanePipeline.run(source: AudioSource) -> None`.

- [ ] **Step 1: Write fake adapters**

In `tests/conftest.py`, create:

```python
class FakeOnlineRecognizer:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.reset_count = 0

    async def accept(self, frame):
        if frame.is_finish:
            return None
        return self.outputs.pop(0) if self.outputs else None

    async def reset(self):
        self.reset_count += 1


class FakeOfflineRecognizer:
    def __init__(self, text="offline final"):
        self.text = text
        self.requests = []

    async def recognize(self, pcm, prompt):
        self.requests.append((pcm.copy(), prompt))
        from multi_stream_asr.asr.offline.base import OfflineRecognitionResult
        return OfflineRecognitionResult(text=self.text, elapsed_ms=400, audio_seconds=1.0, rtf=0.4)

    async def close(self):
        return None
```

- [ ] **Step 2: Write pipeline tests**

Create tests proving:

- partial is emitted when online text changes;
- endpoint with empty online text clears cache and does not call offline;
- endpoint with non-empty text emits pending and later final for the same sentence;
- final empty string replaces pending;
- finish resets online recognizer and emits lane finished;
- sentence ID increments after endpoint handling.

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
pytest tests/test_pipeline.py -v
```

Expected: FAIL because pipeline and recognizer interfaces are missing.

- [ ] **Step 4: Implement online base protocol**

Define `OnlineRecognizer` with async `accept(frame)` and `reset()`. `OnlineRecognition` already comes from `models.py`; if Task 1 did not include `text` and `is_endpoint`, add those exact fields now and update Task 1 tests.

- [ ] **Step 5: Implement sherpa adapter wrapper**

`ZipformerRecognizer` imports `sherpa_onnx` lazily in `__init__`. If unavailable, raise `RuntimeError("sherpa-onnx is required for ZipformerRecognizer")`. Map config endpoint rules and `num_threads=1` into the recognizer constructor.

- [ ] **Step 6: Implement `LanePipeline`**

`LanePipeline` consumes frames from an `AudioSource`, writes normal frames to `SegmentCache`, calls online recognizer, publishes partial/pending/final events, submits offline recognition with `asyncio.create_task`, waits for submitted offline tasks before lane completion, and records metrics.

- [ ] **Step 7: Run tests**

Run:

```bash
pytest tests/test_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/multi_stream_asr/asr/online src/multi_stream_asr/pipeline/pipeline.py tests/conftest.py tests/test_pipeline.py
git commit -m "feat: add lane pipeline orchestration"
```

---

### Task 8: Pipeline Manager and Headless Runner with Fake Integration

**Files:**
- Create: `src/multi_stream_asr/pipeline/manager.py`
- Create: `src/multi_stream_asr/headless/runner.py`
- Create: `src/multi_stream_asr/main.py`
- Create: `tests/test_manager_headless.py`

**Interfaces:**
- Consumes: `AppConfig`, `LanePipeline`, `FrontendEventBus`, `LlamaServerManager`, recognizer factories.
- Produces: `PipelineManager.load_models()`, `start_recognition()`, `stop_recognition()`, `unload_models()`.
- Produces: `HeadlessRunner.run() -> int`.
- Produces: CLI args `--headless`, `--lanes`, `--input-dir`, `--host`, `--port`, `--fake`.

- [ ] **Step 1: Write manager and headless tests**

Tests must prove:

- illegal lifecycle transitions raise clear errors;
- `load_models()` starts offline servers and only reaches loaded after success;
- load failure stops already-started servers;
- fake headless mode emits partial/pending/final, finishes all lanes, saves output, unloads models, and returns exit code 0.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/test_manager_headless.py -v
```

Expected: FAIL because manager/headless/main are missing.

- [ ] **Step 3: Implement `PipelineManager`**

Serialize public lifecycle methods with `asyncio.Lock`. Maintain global `PipelineState`. Build lane pipelines from lane configs and source/recognizer factories. On fatal lane error, enter global stop flow.

- [ ] **Step 4: Implement `HeadlessRunner`**

For normal mode, select local files and use real recognizer factories. For `--fake`, use deterministic fake sources and recognizers so integration tests do not need model files.

- [ ] **Step 5: Implement CLI**

`main.py` parses args. `--headless` runs `HeadlessRunner`. Without `--headless`, it starts the FastAPI app from Task 9. `--fake` is accepted only for tests and local smoke checks.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_manager_headless.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/multi_stream_asr/pipeline/manager.py src/multi_stream_asr/headless src/multi_stream_asr/main.py tests/test_manager_headless.py
git commit -m "feat: add pipeline manager and headless runner"
```

---

### Task 9: FastAPI Server and Static Web UI

**Files:**
- Create: `src/multi_stream_asr/web/server.py`
- Create: `src/multi_stream_asr/web/static/index.html`
- Create: `tests/test_web_server.py`

**Interfaces:**
- Consumes: `PipelineManager`, `FrontendEventBus`, `AppConfig`.
- Produces: `create_app(config: AppConfig | None = None) -> FastAPI`.
- Produces routes: `GET /`, `GET /health`, `POST /api/models/load`, `POST /api/models/unload`, `POST /api/recognition/start`, `POST /api/recognition/stop`, `WS /ws`.

- [ ] **Step 1: Write Web tests**

Tests must prove:

- `GET /health` returns state and lane count;
- load/start/stop/unload endpoints call manager methods;
- WebSocket receives an ASR event published through `FrontendEventBus`;
- static HTML contains lane selector values `2`, `4`, `6`, `8` and controls for load/unload/start/stop.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/test_web_server.py -v
```

Expected: FAIL because Web server is missing.

- [ ] **Step 3: Implement FastAPI app**

`create_app()` constructs or accepts app services, serves `index.html`, exposes JSON control endpoints, and registers WebSocket clients with the event bus. Startup creates the event bus broadcast task; shutdown stops recognition, unloads models, closes sockets, and flushes recorder state.

- [ ] **Step 4: Implement static UI**

Create a single HTML file with a fixed-height app shell, top toolbar, lane grid, hidden inactive config controls, WebSocket connection, event revision filtering, partial gray text, pending endpointed text, final black text, and metrics display.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_web_server.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/multi_stream_asr/web tests/test_web_server.py
git commit -m "feat: add web server and ui"
```

---

### Task 10: Full Test Suite, Fake Smoke Command, and Real-Device Validation Path

**Files:**
- Modify: `README.md`
- Modify: `src/multi_stream_asr/main.py`
- Modify: `src/multi_stream_asr/headless/runner.py`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: stable local fake smoke command and documented real-device validation command.

- [ ] **Step 1: Run the full pytest suite**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 2: Run fake headless smoke**

Run:

```bash
python -m multi_stream_asr.main --headless --fake --lanes 4
```

Expected: exit code 0; output directory contains `config.json`, `metrics.json`, `results/lane<N>.txt`, and `model_outputs/lane<N>.jsonl`; events include `partial`, `pending`, and `final`.

- [ ] **Step 3: Update README with exact commands**

Ensure README includes:

```bash
. .venv313/bin/activate
pytest -v
python -m multi_stream_asr.main --headless --fake --lanes 4
python -m multi_stream_asr.main --headless --lanes 4 --input-dir ~/Music/wav16k
```

- [ ] **Step 4: Run real-device validation after implementation is complete**

On `bianbu@10.0.91.51:~/Workspace/projects/multi-stream-asr`, run:

```bash
. .venv313/bin/activate
python -m multi_stream_asr.main --headless --lanes 4 --input-dir ~/Music/wav16k
```

Expected:

- Command completes without fatal errors.
- All four WAV lanes finish.
- All submitted offline tasks complete.
- Results, config, logs, and metrics are saved.
- All app-started `llama-server` processes stop.
- `pgrep -x llama-server` shows no leftover app-started server.
- Metrics report online average frame latency near 50 ms.
- Metrics report offline RTF near 0.4.

- [ ] **Step 5: Commit final docs or tuning changes**

```bash
git add README.md src/multi_stream_asr/main.py src/multi_stream_asr/headless/runner.py
git commit -m "docs: add validation workflow"
```

---

## Self-Review

Spec coverage:

- Python 3.13 remote environment is covered in Global Constraints and README setup.
- 2/4/6/8 lane config, default model paths, endpoint rules, offline ports, A100 config, pending/final semantics, and headless validation are covered.
- pytest coverage is assigned across config, models, buffers, llama manager, Qwen3-ASR, audio, event bus, pipeline, manager/headless, and Web server.
- The final real-device validation command and ideal metrics are included.

Placeholder scan:

- The plan has no deferred-work markers or unspecified file paths.
- Each task includes exact files, public interfaces, verification commands, expected results, and commit commands.

Type consistency:

- `AppConfig`, `LaneConfig`, `AudioFrame`, `ASREvent`, `EventStatus`, `ResultType`, `OnlineRecognition`, `OfflineRecognitionResult`, `FrontendEventBus`, `LanePipeline`, `PipelineManager`, and `HeadlessRunner` are named consistently across tasks.
