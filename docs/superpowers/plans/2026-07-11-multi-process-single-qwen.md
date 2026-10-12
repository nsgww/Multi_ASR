# Multi-Process Zipformer + Single Shared Qwen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep `main`'s per-lane Python process isolation for audio and Zipformer while replacing four lane-owned Qwen servers with one main-process-owned `llama-multi-asr`, then validate three real 4-lane runs with every lane below 10 dropped frames and combined offline RTF at or below 0.471.

**Architecture:** The main `ProcessPipelineManager` starts and watches one shared Qwen service before spawning lane processes. Lane children retain audio, thread-owned Zipformer, lane artifacts, and metrics, but connect directly to the shared HTTP port; PCM never crosses main-process IPC. Shared service timings are summarized in each lane, transported as JSON metrics, and merged into schema v3.

**Tech Stack:** Python 3.13, asyncio, multiprocessing `spawn`, threading, httpx, FastAPI/WebSocket, sherpa-onnx, SpaceMiT EP, `llama-multi-asr`, pytest.

## Global Constraints

- Base all implementation work on the latest `main` containing design commit `8fb3e5c` or its descendant.
- Create branch `multi-process-single-qwen` and local worktree `~/Workspace/projects/multi-process-single-qwen` using `superpowers:using-git-worktrees` before Task 1.
- Create `models/asr` in the new worktree as a symlink to `~/Workspace/projects/multi-stream-asr/models/asr`; do not copy or commit models.
- Use Python 3.13 only and a project-local `.venv313` locally and on the remote device.
- On the remote device, follow `~/.codex/docs/k3/remote-device-python.md`; activate `.venv313` and export the Spacemit/Aliyun index variables before installing dependencies.
- Remote code and test directory: `bianbu@10.0.91.51:~/Workspace/projects/multi-process-single-qwen`.
- Reuse remote Qwen models from `~/Workspace/sm-sdk/models/asr/qwen3asr` and runtime from `~/Workspace/projects/multi-asr-runtime`.
- Use `Qwen3-ASR-0.6B-text-q40.gguf`, `--n-parallel 4`, and `--n-ctx 8192` exactly.
- Keep the existing 100 ms producer and `OnlineFrameQueue(maxsize=1)` semantics.
- Do not pass PCM, NumPy arrays, segment cache data, raw Qwen responses, or artifact contents through main/lane IPC.
- Do not add direct Linux thread binding in this project.
- Do not retain a runtime switch back to per-lane `llama-server`.
- Keep `results/laneX.txt` and `model_outputs/laneX.jsonl` lane-owned; model-output records must retain exactly six keys.
- If real validation fails, do not tune core allocation, thread counts, `n_parallel`, or Zipformer EP configuration in this task.

## Execution Workspace

Before Task 1, invoke `superpowers:using-git-worktrees`, then run from the original repository:

```bash
git status --short
git worktree add -b multi-process-single-qwen \
  /home/liangff/Workspace/projects/multi-process-single-qwen main
cd /home/liangff/Workspace/projects/multi-process-single-qwen
mkdir -p models
ln -s /home/liangff/Workspace/projects/multi-stream-asr/models/asr models/asr
python3.13 -m venv .venv313
. .venv313/bin/activate
python -m pip install -e '.[dev]'
python --version
pytest -q
```

Expected: the branch is checked out only in the new worktree, `models/asr` resolves to the main worktree's models, Python reports 3.13.x, and the baseline suite reports 191 passed.

---

## File Structure

Create:

- `src/multi_stream_asr/asr/offline/shared_server.py`: one shared service config, scoped stale cleanup, process lifecycle, health, and liveness.
- `tests/test_shared_server.py`: shared config/command/lifecycle/cleanup unit tests.
- `scripts/smoke_shared_qwen.py`: real one-request q40/8192 smoke command.
- `scripts/evaluate_shared_qwen_runs.py`: strict three-run artifact and performance evaluator that emits JSON and Markdown.
- `tests/test_evaluate_shared_qwen_runs.py`: evaluator acceptance/failure tests.
- `docs/superpowers/reports/2026-07-11-multi-process-single-qwen-report.md`: generated real validation report.

Modify:

- `src/multi_stream_asr/config.py`: add shared service configuration and remove per-lane Qwen allocation after cutover.
- `src/multi_stream_asr/asr/offline/base.py`: add optional shared timing payload.
- `src/multi_stream_asr/asr/offline/qwen3asr.py`: shared request format and timing parsing.
- `src/multi_stream_asr/log/metrics.py`: shared timing tracker and summary merge.
- `src/multi_stream_asr/log/recorder.py`: schema v3 global shared timing output.
- `src/multi_stream_asr/pipeline/pipeline.py`: record timing values from completed offline requests.
- `src/multi_stream_asr/pipeline/lane_runtime.py`: remove lane server ownership and emit timing summaries.
- `src/multi_stream_asr/pipeline/process_manager.py`: own and watch the shared service.
- `docs/thread-model.md`: document 1 main + N lanes + 1 shared Qwen topology.
- `tests/test_config.py`, `tests/test_qwen3asr.py`, `tests/test_buffers_metrics.py`, `tests/test_pipeline.py`, `tests/test_lane_runtime.py`, `tests/test_process_manager.py`, `tests/test_manager_headless.py`, `tests/test_web_server.py`: focused and regression coverage.

Delete after cutover:

- `src/multi_stream_asr/asr/offline/llama_server.py`
- `tests/test_llama_server.py`

---

### Task 1: Add Shared Qwen Configuration Without Breaking the Baseline

**Files:**
- Modify: `src/multi_stream_asr/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `SharedQwenConfig`, `AppConfig.shared_qwen`.
- Preserves temporarily: `OfflineServerConfig`, `AppConfig.offline`, and existing `LaneConfig`, until the atomic cutover in Task 5.

- [ ] **Step 1: Write failing shared config tests**

Append to `tests/test_config.py`:

```python
from multi_stream_asr.config import SharedQwenConfig


def test_shared_qwen_defaults_match_validated_four_lane_service():
    shared = SharedQwenConfig()

    assert shared.host == "127.0.0.1"
    assert shared.port == 8090
    assert shared.runtime_dir == Path(
        "~/Workspace/projects/multi-asr-runtime"
    ).expanduser()
    assert shared.encoder_thread_num == 2
    assert shared.encoder_affinity == "8;9"
    assert shared.decoder_cores == "10,11,12,13,14,15"
    assert shared.decoder_threads == 6
    assert shared.n_parallel == 4
    assert shared.n_ctx == 8192
    assert shared.request_timeout_ms == 120000


def test_app_config_exposes_shared_qwen_config():
    config = AppConfig()
    assert config.shared_qwen == SharedQwenConfig()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv313/bin/pytest tests/test_config.py::test_shared_qwen_defaults_match_validated_four_lane_service tests/test_config.py::test_app_config_exposes_shared_qwen_config -v
```

Expected: collection fails because `SharedQwenConfig` does not exist.

- [ ] **Step 3: Add the shared configuration**

Add after `OfflineServerConfig` in `src/multi_stream_asr/config.py`:

```python
DEFAULT_SHARED_RUNTIME_DIR = Path(
    "~/Workspace/projects/multi-asr-runtime"
).expanduser()


@dataclass(frozen=True)
class SharedQwenConfig:
    host: str = "127.0.0.1"
    port: int = 8090
    runtime_dir: Path = DEFAULT_SHARED_RUNTIME_DIR
    encoder_thread_num: int = 2
    encoder_affinity: str = "8;9"
    decoder_cores: str = "10,11,12,13,14,15"
    decoder_threads: int = 6
    n_parallel: int = 4
    n_ctx: int = 8192
    request_timeout_ms: int = 120000
```

Add the transitional field to `AppConfig`:

```python
shared_qwen: SharedQwenConfig = field(default_factory=SharedQwenConfig)
```

Do not remove `offline` or alter `LaneConfig` in this task, so every existing test remains green.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
.venv313/bin/pytest tests/test_config.py -q
.venv313/bin/pytest -q
```

Expected: config tests pass and the full baseline remains green.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/config.py tests/test_config.py
git commit -m "feat: add shared qwen configuration"
```

---

### Task 2: Implement the Main-Owned Shared Qwen Server

**Files:**
- Create: `src/multi_stream_asr/asr/offline/shared_server.py`
- Create: `tests/test_shared_server.py`

**Interfaces:**
- Consumes: `AppConfig.shared_qwen`, `AppConfig.models.offline_model_dir`.
- Produces: `SharedQwenServerSpec`, `build_shared_config(config)`, `build_launch_command(config, config_dir)`, `SharedQwenServerManager.start()`, `.stop()`, `.is_alive()`, `.exit_code`.

- [ ] **Step 1: Write failing config and launch-command tests**

Create `tests/test_shared_server.py` with fake process helpers and these first tests:

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from multi_stream_asr.config import AppConfig, ModelConfig, SharedQwenConfig
from multi_stream_asr.asr.offline.shared_server import (
    SharedQwenServerManager,
    build_launch_command,
    build_shared_config,
)


class FakeProcess:
    def __init__(self, pid: int = 4321, *, timeout_until_kill: bool = False):
        self.pid = pid
        self.returncode = None
        self.timeout_until_kill = timeout_until_kill
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout=None):
        if self.timeout_until_kill and self.kill_calls == 0:
            raise subprocess.TimeoutExpired("llama-multi-asr", timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_shared_config_uses_flat_lowercase_ep_fields_and_q40(tmp_path: Path):
    config = AppConfig(models=ModelConfig(offline_model_dir=tmp_path))

    payload = build_shared_config(config)

    audio = payload["audio_model"]
    assert audio["frontend_model_path"] == str(
        tmp_path / "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx"
    )
    assert audio["backend_model_path"] == str(
        tmp_path / "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx"
    )
    assert audio["spacemit_ep_intra_thread_num"] == 2
    assert audio["spacemit_ep_inter_thread_num"] == 1
    assert audio["spacemit_ep_intra_thread_affinity"] == "8;9"
    assert "ep_config" not in audio
    assert payload["text_model"]["model_path"].endswith(
        "Qwen3-ASR-0.6B-text-q40.gguf"
    )
    assert payload["prompt"]["asr_text_tag"] == "<asr_text>"
    assert "Chinese" in payload["support_languages"]


def test_launch_command_pins_parallelism_context_and_decoder_cores(tmp_path: Path):
    shared = SharedQwenConfig(runtime_dir=tmp_path / "runtime")
    config = AppConfig(
        models=ModelConfig(offline_model_dir=tmp_path / "models"),
        shared_qwen=shared,
    )

    command = build_launch_command(config, tmp_path / "config")

    assert command[0] == str(tmp_path / "runtime/bin/llama-multi-asr")
    assert command[command.index("--n-parallel") + 1] == "4"
    assert command[command.index("--n-ctx") + 1] == "8192"
    assert command[command.index("--decoder-cores") + 1] == "10,11,12,13,14,15"
    assert command[command.index("--decoder-n-threads") + 1] == "6"
```

- [ ] **Step 2: Run and verify RED**

```bash
.venv313/bin/pytest tests/test_shared_server.py -v
```

Expected: import fails because `shared_server.py` does not exist.

- [ ] **Step 3: Implement config and command builders**

Create `src/multi_stream_asr/asr/offline/shared_server.py` with these constants and
the complete config builder:

```python
SHARED_CONFIG_DIR = Path("/tmp/multi-stream-asr/shared-qwen")
SHARED_PID_PATH = SHARED_CONFIG_DIR / "llama-multi-asr.pid"
SHARED_LOG_PATH = SHARED_CONFIG_DIR / "llama-multi-asr.log"
TEXT_MODEL_NAME = "Qwen3-ASR-0.6B-text-q40.gguf"
FRONTEND_MODEL_NAME = "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx"
BACKEND_MODEL_NAME = "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx"
SUPPORT_LANGUAGES = (
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch",
    "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian",
    "Greek", "Romanian", "Hungarian", "Macedonian",
)


def build_shared_config(config: AppConfig) -> dict[str, object]:
    model_dir = config.models.offline_model_dir
    shared = config.shared_qwen
    return {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "support_languages": list(SUPPORT_LANGUAGES),
        "audio_model": {
            "frontend_model_path": str(model_dir / FRONTEND_MODEL_NAME),
            "backend_model_path": str(model_dir / BACKEND_MODEL_NAME),
            "spacemit_ep_intra_thread_num": shared.encoder_thread_num,
            "spacemit_ep_inter_thread_num": 1,
            "spacemit_ep_intra_thread_affinity": shared.encoder_affinity,
            "num_mel_bins": 128,
            "sample_rate": 16000,
            "n_fft": 400,
            "window_len": 400,
            "hop_len": 160,
            "chunk_frames": 100,
            "chunk_tokens": 13,
            "d_model": 896,
            "hidden_size": 1024,
            "output_dim": 1024,
            "special_tokens": {
                "audio_start_token": "<|audio_start|>",
                "audio_end_token": "<|audio_end|>",
                "audio_token": "<|audio_pad|>",
                "audio_start_token_id": 151669,
                "audio_end_token_id": 151670,
                "audio_token_id": 151676,
            },
        },
        "text_model": {
            "model_path": str(model_dir / TEXT_MODEL_NAME),
            "context_size": 65536,
            "logical_batch_size": 2048,
            "predict_token": 256,
            "vocab_size": 151936,
            "hidden_size": 1024,
        },
        "prompt": {
            "system_prefix_token": "<|im_start|>system",
            "user_prefix_token": "<|im_start|>user",
            "assistant_prefix_token": "<|im_start|>assistant",
            "message_suffix_token": "<|im_end|>",
            "asr_text_tag": "<asr_text>",
        },
    }
```

The full command returned by `build_launch_command` must be:

```python
return [
    str(shared.runtime_dir / "bin" / "llama-multi-asr"),
    "-m", str(config.models.offline_model_dir / TEXT_MODEL_NAME),
    "--smt-config-dir", str(config_dir),
    "--host", shared.host,
    "--port", str(shared.port),
    "--decoder-cores", shared.decoder_cores,
    "--decoder-n-threads", str(shared.decoder_threads),
    "--n-parallel", str(shared.n_parallel),
    "--n-ctx", str(shared.n_ctx),
    "--warmup",
]
```

- [ ] **Step 4: Add failing lifecycle and scoped-cleanup tests**

Append tests that assert:

```python
def test_scoped_cleanup_ignores_pid_with_nonmatching_cmdline(tmp_path, monkeypatch):
    manager = SharedQwenServerManager(AppConfig())
    pid_path = tmp_path / "service.pid"
    pid_path.write_text("1234\n")
    killed = []
    monkeypatch.setattr(manager, "_pid_path", pid_path)
    monkeypatch.setattr(manager, "_read_cmdline", lambda pid: ["other-service"])
    monkeypatch.setattr("os.kill", lambda pid, signal: killed.append((pid, signal)))

    manager.cleanup_before_start()

    assert killed == []
    assert not pid_path.exists()


def test_start_writes_config_waits_health_and_exposes_liveness(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    runtime_dir = tmp_path / "runtime"
    (runtime_dir / "bin").mkdir(parents=True)
    (runtime_dir / "lib").mkdir()
    (runtime_dir / "ort_lib").mkdir()
    (runtime_dir / "bin/llama-multi-asr").write_text("binary")
    model_dir.mkdir()
    for name in (
        "Qwen3-ASR-0.6B-text-q40.gguf",
        "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
        "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
    ):
        (model_dir / name).write_text("model")
    config = AppConfig(
        models=ModelConfig(offline_model_dir=model_dir),
        shared_qwen=SharedQwenConfig(runtime_dir=runtime_dir),
    )
    process = FakeProcess()
    manager = SharedQwenServerManager(config)
    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(manager, "_config_dir", tmp_path / "generated")
    monkeypatch.setattr(manager, "_pid_path", tmp_path / "service.pid")
    monkeypatch.setattr(manager, "_log_path", tmp_path / "service.log")
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.is_server_healthy",
        lambda url, timeout: True,
    )

    spec = manager.start()

    assert spec.port == 8090
    assert manager.is_alive()
    assert manager.exit_code is None
    assert json.loads((tmp_path / "generated/config.json").read_text())
    manager.stop()
    assert process.terminate_calls == 1
    assert not manager.is_alive()


def test_stop_escalates_to_kill(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    runtime_dir = tmp_path / "runtime"
    (runtime_dir / "bin").mkdir(parents=True)
    (runtime_dir / "lib").mkdir()
    (runtime_dir / "ort_lib").mkdir()
    (runtime_dir / "bin/llama-multi-asr").write_text("binary")
    model_dir.mkdir()
    for name in (
        "Qwen3-ASR-0.6B-text-q40.gguf",
        "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
        "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
    ):
        (model_dir / name).write_text("model")
    config = AppConfig(
        models=ModelConfig(offline_model_dir=model_dir),
        shared_qwen=SharedQwenConfig(runtime_dir=runtime_dir),
    )
    process = FakeProcess(timeout_until_kill=True)
    manager = SharedQwenServerManager(config, stop_timeout_seconds=0.01)
    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(manager, "_config_dir", tmp_path / "generated")
    monkeypatch.setattr(manager, "_pid_path", tmp_path / "service.pid")
    monkeypatch.setattr(manager, "_log_path", tmp_path / "service.log")
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.is_server_healthy",
        lambda url, timeout: True,
    )

    manager.start()
    manager.stop()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
```

- [ ] **Step 5: Implement lifecycle and scoped stale cleanup**

Implement:

```python
@dataclass(frozen=True)
class SharedQwenServerSpec:
    port: int
    config_dir: Path
    log_path: Path
    pid_path: Path
    env: dict[str, str]
    command: list[str]


class SharedQwenServerManager:
    def __init__(self, config: AppConfig, *, health_timeout_seconds=120.0,
                 poll_interval_seconds=1.0, stop_timeout_seconds=5.0):
        self._config = config
        self._health_timeout_seconds = health_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_timeout_seconds = stop_timeout_seconds
        self._config_dir = SHARED_CONFIG_DIR
        self._pid_path = SHARED_PID_PATH
        self._log_path = SHARED_LOG_PATH
        self._process = None
        self._log_handle = None

    @property
    def exit_code(self) -> int | None:
        return None if self._process is None else self._process.poll()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None
```

`cleanup_before_start()` must read only `_pid_path`, parse one positive PID, read
`/proc/<pid>/cmdline`, and signal it only when both the resolved runtime binary and
`--port 8090` match. Remove invalid/stale pid files. Use SIGTERM, poll for at most
`stop_timeout_seconds`, then SIGKILL. It may call `spacemit-tcm-smi -c` after scoped cleanup.

`start()` must validate the binary and all three model files, create the config/log directories,
write config, prepend runtime `lib` and `ort_lib` to `LD_LIBRARY_PATH`, launch once, write PID,
wait for health, and stop on any exception. `stop()` must be idempotent and always close the log
handle and unlink the owned PID file.

- [ ] **Step 6: Run focused and full tests**

```bash
.venv313/bin/pytest tests/test_shared_server.py -q
.venv313/bin/pytest -q
```

Expected: all tests pass; no real process is started.

- [ ] **Step 7: Commit**

```bash
git add src/multi_stream_asr/asr/offline/shared_server.py tests/test_shared_server.py
git commit -m "feat: add shared qwen server manager"
```

---

### Task 3: Add the Shared Qwen Client Protocol and Timing Result

**Files:**
- Modify: `src/multi_stream_asr/asr/offline/base.py`
- Modify: `src/multi_stream_asr/asr/offline/qwen3asr.py`
- Modify: `tests/test_qwen3asr.py`

**Interfaces:**
- Produces: `OfflineRecognitionResult.timings: dict[str, float] | None`, `parse_timings(payload)`, and `Qwen3AsrClient(shared=SharedQwenConfig)`.
- Preserves temporarily: the legacy constructor path only until Task 5 atomically rewires lane runtime; remove it in Task 5.

- [ ] **Step 1: Write failing request and timing tests**

Port these tests from `single-process-share-qwen:tests/test_qwen3asr.py` into the current test file:

```python
def test_build_chat_request_matches_shared_service_format():
    pcm = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)
    payload = build_chat_request(pcm)
    assert payload["model"] == "qwen3-asr"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 256
    assert "id_slot" not in payload
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"][1] == {
        "type": "text",
        "text": "language Chinese<asr_text>",
    }


@pytest.mark.asyncio
async def test_recognize_posts_once_and_parses_shared_timings(monkeypatch):
    client = FakeAsyncClient(post_responses=[FakeResponse(200, {
        "choices": [{"message": {"content": "<asr_text> hello </asr_text>"}}],
        "multi_asr_timings": {
            "queue_ms": 12.5,
            "encode_ms": 900.0,
            "prefill_ms": 300.0,
            "decode_ms": 800.0,
            "total_ms": 2012.5,
        },
    })])
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient",
        lambda: client,
    )
    recognizer = Qwen3AsrClient(
        shared=SharedQwenConfig(port=9090, request_timeout_ms=2000)
    )

    result = await recognizer.recognize(np.asarray([0.0, 0.5], dtype=np.float32), "ignored")

    assert result.text == "hello"
    assert result.timings["queue_ms"] == 12.5
    assert len(client.calls) == 1
    assert client.calls[0][1] == "http://127.0.0.1:9090/v1/chat/completions"
    assert client.calls[0][3] == 2.0
```

Keep existing WAV encoding and text-cleaning tests. Remove assertions that require `id_slot` or slot erase.

- [ ] **Step 2: Run and verify RED**

```bash
.venv313/bin/pytest tests/test_qwen3asr.py -q
```

Expected: shared request and constructor assertions fail.

- [ ] **Step 3: Implement the result and shared request**

Add to `OfflineRecognitionResult`:

```python
timings: dict[str, float] | None = None
```

Replace request constants and payload with:

```python
QWEN_USER_PROMPT = "language Chinese<asr_text>"
QWEN_MAX_TOKENS = 256


def build_chat_request(pcm, max_tokens=QWEN_MAX_TOKENS):
    return {
        "model": "qwen3-asr",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {
                    "data": encode_wav_base64(pcm), "format": "wav"
                }},
                {"type": "text", "text": QWEN_USER_PROMPT},
            ],
        }],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def parse_timings(payload: dict[str, Any]) -> dict[str, float] | None:
    raw = payload.get("multi_asr_timings")
    if not isinstance(raw, dict):
        return None
    allowed = {"queue_ms", "encode_ms", "prefill_ms", "decode_ms", "total_ms"}
    values = {
        key: float(value)
        for key, value in raw.items()
        if key in allowed and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return values or None
```

Make `Qwen3AsrClient(shared=SharedQwenConfig())` post exactly once and return
`timings=parse_timings(raw_response)`.
For this intermediate commit, support the old `lane/offline` keyword pair only to keep existing lane tests
green; mark that branch for removal in Task 5. The shared path must never erase a slot.

- [ ] **Step 4: Run focused and full tests**

```bash
.venv313/bin/pytest tests/test_qwen3asr.py -q
.venv313/bin/pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/asr/offline/base.py src/multi_stream_asr/asr/offline/qwen3asr.py tests/test_qwen3asr.py
git commit -m "feat: use shared qwen request protocol"
```

---

### Task 4: Track, Transport, and Merge Shared Service Timings

**Files:**
- Modify: `src/multi_stream_asr/log/metrics.py`
- Modify: `src/multi_stream_asr/log/recorder.py`
- Modify: `src/multi_stream_asr/pipeline/pipeline.py`
- Modify: `tests/test_buffers_metrics.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `SHARED_TIMING_FIELDS`, `SharedTimingsTracker.add()`, `.as_dict()`,
  `merge_shared_timing_summaries()`, and the
  `LanePipeline(shared_timings_tracker=SharedTimingsTracker())` constructor contract.
- Summary shape: overall `count`, then each field has `count`, `total_ms`, `average_ms`, `max_ms`.

- [ ] **Step 1: Write failing tracker and merge tests**

Append to `tests/test_buffers_metrics.py`:

```python
def test_shared_timing_tracker_records_only_known_finite_fields():
    tracker = SharedTimingsTracker()
    tracker.add({
        "queue_ms": 10.0,
        "encode_ms": 100.0,
        "prefill_ms": 20.0,
        "decode_ms": 200.0,
        "total_ms": 330.0,
        "unknown_ms": 1.0,
    })
    tracker.add({"queue_ms": 30.0, "total_ms": 770.0})

    payload = tracker.as_dict()

    assert payload["count"] == 2
    assert payload["queue_ms"] == {
        "count": 2, "total_ms": 40.0, "average_ms": 20.0, "max_ms": 30.0
    }
    assert payload["encode_ms"]["count"] == 1
    assert "unknown_ms" not in payload


def test_merge_shared_timing_summaries_weights_by_field_count():
    merged = merge_shared_timing_summaries([
        {"count": 2, "queue_ms": {
            "count": 2, "total_ms": 40.0, "average_ms": 20.0, "max_ms": 30.0
        }},
        {"count": 1, "queue_ms": {
            "count": 1, "total_ms": 50.0, "average_ms": 50.0, "max_ms": 50.0
        }},
    ])
    assert merged["count"] == 3
    assert merged["queue_ms"] == {
        "count": 3, "total_ms": 90.0, "average_ms": 30.0, "max_ms": 50.0
    }
```

Add a recorder test with two lane summaries and assert global and per-lane
`offline_shared_timings` are present while model-output keys remain unchanged.

- [ ] **Step 2: Run and verify RED**

```bash
.venv313/bin/pytest tests/test_buffers_metrics.py -q
```

Expected: imports fail for the new tracker/merge helper.

- [ ] **Step 3: Implement the thread-safe tracker and merge**

Add to `metrics.py`:

```python
SHARED_TIMING_FIELDS = (
    "queue_ms", "encode_ms", "prefill_ms", "decode_ms", "total_ms"
)


@dataclass
class SharedTimingsTracker:
    samples: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in SHARED_TIMING_FIELDS}
    )

    def __post_init__(self):
        self._lock = threading.Lock()

    def add(self, timings: dict[str, float]) -> None:
        accepted = {
            key: float(value)
            for key, value in timings.items()
            if key in SHARED_TIMING_FIELDS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        }
        if not accepted:
            return
        with self._lock:
            for key, value in accepted.items():
                self.samples[key].append(value)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            copied = {key: tuple(values) for key, values in self.samples.items()}
        payload = {"count": max((len(v) for v in copied.values()), default=0)}
        for key, values in copied.items():
            count = len(values)
            total = float(sum(values))
            payload[key] = {
                "count": count,
                "total_ms": total,
                "average_ms": total / count if count else 0.0,
                "max_ms": max(values, default=0.0),
            }
        return payload
```

Implement `merge_shared_timing_summaries()` by summing each field's `count` and `total_ms`,
taking max `max_ms`, and recomputing `average_ms`; set overall `count` to the sum of input
overall counts.

- [ ] **Step 4: Record timings in the lane pipeline**

Add `shared_timings_tracker: SharedTimingsTracker | None = None` to `LanePipeline.__init__`,
default it to a new tracker, and in `run_offline()` add:

```python
if result.timings is not None:
    self.shared_timings_tracker.add(result.timings)
```

Add a pipeline test whose fake offline result contains all five timings, run one endpoint, and assert
the tracker count is 1 and `total_ms.average_ms` matches the result.

- [ ] **Step 5: Merge timings in `save_process_run`**

Add to schema v3 payload:

```python
"offline_shared_timings": merge_shared_timing_summaries([
    dict(lane.get("offline_shared_timings", {})) for lane in lanes
]),
```

Do not remove the timing block from each lane dict. For legacy schema 2 `save()`, write an empty
`SharedTimingsTracker().as_dict()` so existing non-process callers have a stable key.

- [ ] **Step 6: Run focused and full tests**

```bash
.venv313/bin/pytest tests/test_buffers_metrics.py tests/test_pipeline.py -q
.venv313/bin/pytest -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/multi_stream_asr/log/metrics.py src/multi_stream_asr/log/recorder.py src/multi_stream_asr/pipeline/pipeline.py tests/test_buffers_metrics.py tests/test_pipeline.py
git commit -m "feat: aggregate shared qwen timings"
```

---

### Task 5: Cut Lane Runtime and Process Manager Over to One Shared Service

**Files:**
- Modify: `src/multi_stream_asr/config.py`
- Modify: `src/multi_stream_asr/asr/offline/qwen3asr.py`
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py`
- Modify: `src/multi_stream_asr/pipeline/process_manager.py`
- Delete: `src/multi_stream_asr/asr/offline/llama_server.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_lane_runtime.py`
- Modify: `tests/test_process_manager.py`
- Modify: `tests/test_manager_headless.py`
- Modify: `tests/test_web_server.py`
- Delete: `tests/test_llama_server.py`

**Interfaces:**
- Final config: `AppConfig.shared_qwen`; no `AppConfig.offline`.
- Final lane config: `LaneConfig(stream_id: int)`.
- Process manager owns `SharedQwenServerManager` and `asyncio.Task` watchdog.
- Lane metrics payload includes `offline_shared_timings`.

- [ ] **Step 1: Write failing lane ownership tests**

Replace the old server-owning test in `tests/test_lane_runtime.py` with:

```python
def test_real_lane_loads_zipformer_and_shared_client_without_server(tmp_path, monkeypatch):
    order = []
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    lane = config.lane_configs[0]

    class FakeOnlineRecognizer:
        def __init__(self, model_dir, endpoint):
            order.append("online_init")

    class FakeSharedClient:
        def __init__(self, *, shared):
            assert shared is config.shared_qwen
            order.append("offline_init")

        async def close(self):
            order.append("offline_close")

    monkeypatch.setattr(lane_runtime, "ZipformerRecognizer", FakeOnlineRecognizer)
    monkeypatch.setattr(lane_runtime, "Qwen3AsrClient", FakeSharedClient)
    commands.put({"type": "stop"})
    runtime = LaneRuntime(
        process_config=LaneProcessConfig(app_config=config, lane=lane),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0
    assert order == ["online_init", "offline_init", "offline_close"]
    assert not hasattr(runtime, "_server_manager")
```

Update the compact snapshot assertion to require:

```python
assert set(snapshot.payload) == {
    "lane_metrics",
    "online_frame_latency",
    "offline_rtf",
    "offline_shared_timings",
    "event_loop_lag",
}
```

- [ ] **Step 2: Write failing process manager lifecycle tests**

Add a fake shared manager with ordered calls and these tests to `tests/test_process_manager.py`:

```python
class FakeSharedManager:
    instances = []

    def __init__(self, config):
        self.config = config
        self.alive = False
        self.code = None
        self.calls = []
        type(self).instances.append(self)

    @property
    def exit_code(self):
        return self.code

    def start(self):
        self.calls.append("start")
        self.alive = True

    def stop(self):
        self.calls.append("stop")
        self.alive = False
        if self.code is None:
            self.code = 0

    def is_alive(self):
        return self.alive


@pytest.mark.asyncio
async def test_nonfake_manager_starts_shared_qwen_before_lanes_and_stops_after_them(
    tmp_path, monkeypatch
):
    actions = []
    monkeypatch.setattr(process_manager, "SharedQwenServerManager", FakeSharedManager)
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    original_start = _ScriptedController.start
    def recording_start(controller):
        actions.append(f"lane:{controller.lane.stream_id}")
        original_start(controller)
    monkeypatch.setattr(_ScriptedController, "start", recording_start)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
    )

    await manager.load_models()
    shared = FakeSharedManager.instances[0]
    assert shared.calls == ["start"]
    assert actions == ["lane:0", "lane:1"]
    await manager.unload_models()
    assert shared.calls == ["start", "stop"]


@pytest.mark.asyncio
async def test_shared_qwen_exit_marks_all_lanes_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(process_manager, "SharedQwenServerManager", FakeSharedManager)
    monkeypatch.setattr(process_manager, "LaneProcessController", _ScriptedController)
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2, output_dir=tmp_path),
        event_bus=FrontendEventBus(),
        shared_watch_interval_seconds=0.001,
    )
    await manager.load_models()
    shared = FakeSharedManager.instances[0]
    shared.alive = False
    shared.code = 137
    await asyncio.sleep(0.01)
    assert manager.state is PipelineState.FATAL
    assert "shared qwen" in str(manager._failure)
    assert "137" in str(manager._failure)
    await manager.unload_models()
    assert manager.state is PipelineState.UNLOADED
```

Also add tests for shared start failure (no lane controller starts), fake mode (no shared manager), and
watchdog cancellation before intentional stop (normal unload must not become fatal).

- [ ] **Step 3: Run selected tests and verify RED**

```bash
.venv313/bin/pytest \
  tests/test_lane_runtime.py::test_real_lane_loads_zipformer_and_shared_client_without_server \
  tests/test_process_manager.py::test_nonfake_manager_starts_shared_qwen_before_lanes_and_stops_after_them \
  tests/test_process_manager.py::test_shared_qwen_exit_marks_all_lanes_fatal -v
```

Expected: tests fail because lane runtime still owns `LlamaServerManager` and process manager does not own shared Qwen.

- [ ] **Step 4: Simplify final config and client**

In `config.py`:

```python
@dataclass(frozen=True)
class LaneConfig:
    stream_id: int


def build_lane_configs(lanes: int) -> list[LaneConfig]:
    if lanes not in SUPPORTED_LANES:
        raise ValueError(f"lanes must be one of {SUPPORTED_LANES}, got {lanes}")
    return [LaneConfig(stream_id=stream_id) for stream_id in range(lanes)]
```

Remove `OfflineServerConfig`, `CoreAllocation`, `core_allocations_for_lanes`, `AppConfig.offline`, and the
temporary legacy client constructor. `AppConfig.lane_configs` calls `build_lane_configs(self.lanes)`.

- [ ] **Step 5: Remove lane-owned Qwen server lifecycle and add timing payload**

In `lane_runtime.py`:

- Remove `LlamaServerManager`, `_server_manager`, `_server_stopped`, `start_lane`, and `stop_all`.
- Keep `OnlineConsumerWorker` unchanged so Zipformer is loaded and executed on its owner thread.
- Construct `Qwen3AsrClient(shared=self.app_config.shared_qwen)` after Zipformer load.
- Add `self._shared_timings_tracker = SharedTimingsTracker()`.
- Pass it to `LanePipeline(shared_timings_tracker=self._shared_timings_tracker)`.
- Add to `_metrics_payload()`:

```python
"offline_shared_timings": self._shared_timings_tracker.as_dict(),
```

Use `self.app_config.shared_qwen.request_timeout_ms` for offline-tail timeout.

- [ ] **Step 6: Implement shared ownership and watchdog in process manager**

Add constructor argument and state:

```python
shared_watch_interval_seconds: float = 0.1
self._shared_manager: SharedQwenServerManager | None = None
self._shared_watchdog_task: asyncio.Task[None] | None = None
self._shared_stop_expected = False
```

Change the default lifecycle timeout source to:

```python
self._lifecycle_timeout_seconds = (
    config.shared_qwen.request_timeout_ms / 1000.0 + cleanup_margin_seconds
    if lifecycle_timeout_seconds is None
    else lifecycle_timeout_seconds
)
```

Add an empty shared summary to the fallback built by `lane_summaries`:

```python
"offline_shared_timings": SharedTimingsTracker().as_dict(),
```

At the beginning of real `load_models()`:

```python
self._shared_manager = SharedQwenServerManager(self._config)
await asyncio.to_thread(self._shared_manager.start)
self._shared_stop_expected = False
self._shared_watchdog_task = asyncio.create_task(
    self._watch_shared_qwen(), name="asr-shared-qwen-watchdog"
)
```

Only then create/start lane controllers. Implement watchdog:

```python
async def _watch_shared_qwen(self) -> None:
    assert self._shared_manager is not None
    while True:
        await asyncio.sleep(self._shared_watch_interval_seconds)
        if self._shared_stop_expected:
            return
        if not self._shared_manager.is_alive():
            code = self._shared_manager.exit_code
            await self._record_fatal(0, f"shared qwen exited unexpectedly (exit code {code})")
            return
```

Implement `_stop_shared_qwen()` to set expected stop, cancel and await watchdog, call manager.stop via
`asyncio.to_thread`, then clear references. Call it only after lane join/close completes. On shared start
failure, no lane may be spawned. On lane load/run fatal, retain the existing sibling-stop logic and stop
shared Qwen after every lane process is dead.

Remove `_cleanup_forced_servers_if_ready`, `_forced_cleanup_done`, and the import of
`LlamaServerManager`. Forced lane termination no longer needs orphan server discovery because the shared
service remains a direct child owned by the main manager; `_stop_shared_qwen()` is the single cleanup path.

- [ ] **Step 7: Delete the dedicated server implementation and update regressions**

Delete `llama_server.py` and `test_llama_server.py`. Replace all monkeypatch references to
`LlamaServerManager` with the shared manager at main-process tests or remove them from lane tests.
Update `tests/test_manager_headless.py` and `tests/test_web_server.py` fakes so fake mode asserts no real
shared server is constructed. Preserve all existing process artifact/path/fatal tests.

- [ ] **Step 8: Run focused and full tests**

```bash
.venv313/bin/pytest tests/test_lane_runtime.py tests/test_process_manager.py -q
.venv313/bin/pytest tests/test_manager_headless.py tests/test_web_server.py -q
.venv313/bin/pytest -q
```

Expected: all tests pass and the suite has no import of `llama_server.py`.

- [ ] **Step 9: Commit**

```bash
git add src/multi_stream_asr/config.py src/multi_stream_asr/asr/offline/qwen3asr.py src/multi_stream_asr/pipeline/lane_runtime.py src/multi_stream_asr/pipeline/process_manager.py tests/test_config.py tests/test_lane_runtime.py tests/test_process_manager.py tests/test_manager_headless.py tests/test_web_server.py
git rm src/multi_stream_asr/asr/offline/llama_server.py tests/test_llama_server.py
git commit -m "feat: share one qwen service across lane processes"
```

---

### Task 6: Add Reproducible Smoke and Three-Run Evaluation Tools

**Files:**
- Create: `scripts/smoke_shared_qwen.py`
- Create: `scripts/evaluate_shared_qwen_runs.py`
- Create: `tests/test_evaluate_shared_qwen_runs.py`

**Interfaces:**
- Produces: `evaluate_run(path)`, `evaluate_runs(paths)`, CLI exit 0 only when all functional/performance gates pass, and Markdown report output.

- [ ] **Step 1: Write failing evaluator tests**

Create fixtures for three minimal run directories and tests:

```python
import json
from pathlib import Path

import pytest

from scripts.evaluate_shared_qwen_runs import evaluate_runs


TIMING_FIELDS = ("queue_ms", "encode_ms", "prefill_ms", "decode_ms", "total_ms")


def timing_summary(value: float = 1.0) -> dict[str, object]:
    return {
        "count": 1,
        **{
            field: {
                "count": 1,
                "total_ms": value,
                "average_ms": value,
                "max_ms": value,
            }
            for field in TIMING_FIELDS
        },
    }


def make_run(
    path: Path,
    *,
    drops: list[int],
    rtf_total: float,
    rtf_count: int,
) -> Path:
    (path / "results").mkdir(parents=True)
    (path / "model_outputs").mkdir()
    lanes = []
    for stream_id, dropped in enumerate(drops):
        lanes.append({
            "stream_id": stream_id,
            "processed_frames": 10,
            "dropped_frames": dropped,
            "offline_requests": 1,
            "final_state": "finished",
            "online_frame_latency": {
                "count": 10, "total_ms": 400.0,
                "average_ms": 40.0, "max_ms": 60.0,
            },
            "offline_rtf": {
                "count": 1, "total": rtf_total / rtf_count,
                "average": rtf_total / rtf_count, "max": rtf_total / rtf_count,
            },
            "event_loop_lag": {
                "count": 10, "total_ms": 10.0,
                "average_ms": 1.0, "max_ms": 2.0,
            },
            "offline_shared_timings": timing_summary(),
            "results_path": f"results/lane{stream_id}.txt",
            "model_outputs_path": f"model_outputs/lane{stream_id}.jsonl",
        })
        (path / "results" / f"lane{stream_id}.txt").write_text("final text\n")
        records = [
            {
                "model": "zipformer", "audio_seconds": None, "rtf": None,
                "stream_id": stream_id, "sentence_id": 0, "text": "endpoint",
            },
            {
                "model": "qwen3-asr", "audio_seconds": 1.0,
                "rtf": rtf_total / rtf_count, "stream_id": stream_id,
                "sentence_id": 0, "text": "final text",
            },
        ]
        output = path / "model_outputs" / f"lane{stream_id}.jsonl"
        output.write_text("".join(json.dumps(row) + "\n" for row in records))
    metrics = {
        "metrics_schema_version": 3,
        "online_frame_latency": {
            "count": 40, "total_ms": 1600.0,
            "average_ms": 40.0, "max_ms": 60.0,
        },
        "offline_rtf": {
            "count": rtf_count, "total": rtf_total,
            "average": rtf_total / rtf_count, "max": rtf_total / rtf_count,
        },
        "event_loop_lag": {
            "count": 40, "total_ms": 40.0,
            "average_ms": 1.0, "max_ms": 2.0,
        },
        "offline_shared_timings": timing_summary(),
        "lanes": lanes,
    }
    (path / "metrics.json").write_text(json.dumps(metrics))
    return path


def test_three_run_evaluator_accepts_strict_thresholds(tmp_path):
    runs = [make_run(tmp_path / f"run{i}", drops=[1, 2, 3, 4], rtf_total=4.70, rtf_count=10)
            for i in range(3)]
    result = evaluate_runs(runs)
    assert result["passed"] is True
    assert result["combined_offline_rtf"] == pytest.approx(0.47)


def test_evaluator_rejects_any_lane_with_ten_drops(tmp_path):
    runs = [make_run(tmp_path / f"run{i}", drops=[1, 2, 3, 4], rtf_total=4.70, rtf_count=10)
            for i in range(3)]
    metrics = json.loads((runs[1] / "metrics.json").read_text())
    metrics["lanes"][2]["dropped_frames"] = 10
    (runs[1] / "metrics.json").write_text(json.dumps(metrics))
    result = evaluate_runs(runs)
    assert result["passed"] is False
    assert "run1 lane2 dropped_frames=10" in result["failures"]


def test_evaluator_rejects_combined_rtf_above_point_471(tmp_path):
    runs = [make_run(tmp_path / f"run{i}", drops=[1, 2, 3, 4], rtf_total=4.72, rtf_count=10)
            for i in range(3)]
    assert evaluate_runs(runs)["passed"] is False
```

- [ ] **Step 2: Run and verify RED**

```bash
.venv313/bin/pytest tests/test_evaluate_shared_qwen_runs.py -v
```

Expected: evaluator module is missing.

- [ ] **Step 3: Implement strict evaluator**

`evaluate_run(path)` must reject:

- schema version other than 3;
- lane count other than 4;
- any lane not `finished`;
- missing `offline_shared_timings` or any of five timing fields;
- empty result file;
- model-output keys other than exactly `{model,audio_seconds,rtf,stream_id,sentence_id,text}`;
- Qwen sentence IDs not exactly equal to Zipformer endpoint sentence IDs.

Implement its core with these constants and checks:

```python
MODEL_OUTPUT_KEYS = {
    "model", "audio_seconds", "rtf", "stream_id", "sentence_id", "text"
}
TIMING_FIELDS = ("queue_ms", "encode_ms", "prefill_ms", "decode_ms", "total_ms")


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def evaluate_run(path: Path) -> dict[str, object]:
    metrics = _load_json(path / "metrics.json")
    failures: list[str] = []
    if metrics.get("metrics_schema_version") != 3:
        failures.append(f"{path.name} metrics_schema_version is not 3")
    lanes = metrics.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 4:
        failures.append(f"{path.name} must contain exactly four lanes")
        lanes = []
    shared = metrics.get("offline_shared_timings")
    if not isinstance(shared, dict) or any(field not in shared for field in TIMING_FIELDS):
        failures.append(f"{path.name} missing offline_shared_timings")
    drops: list[int] = []
    for stream_id, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            failures.append(f"{path.name} lane{stream_id} is not an object")
            continue
        dropped = lane.get("dropped_frames")
        drops.append(int(dropped) if isinstance(dropped, int) else -1)
        if lane.get("final_state") != "finished":
            failures.append(f"{path.name} lane{stream_id} is not finished")
        lane_shared = lane.get("offline_shared_timings")
        if not isinstance(lane_shared, dict) or any(
            field not in lane_shared for field in TIMING_FIELDS
        ):
            failures.append(f"{path.name} lane{stream_id} missing shared timings")
        result_path = path / "results" / f"lane{stream_id}.txt"
        if not result_path.is_file() or not result_path.read_text().strip():
            failures.append(f"{path.name} lane{stream_id} result is empty")
        output_path = path / "model_outputs" / f"lane{stream_id}.jsonl"
        rows = [json.loads(line) for line in output_path.read_text().splitlines()]
        if any(set(row) != MODEL_OUTPUT_KEYS for row in rows):
            failures.append(f"{path.name} lane{stream_id} model-output keys differ")
        zip_ids = {row["sentence_id"] for row in rows if row["model"] == "zipformer"}
        qwen_ids = {row["sentence_id"] for row in rows if row["model"] == "qwen3-asr"}
        if zip_ids != qwen_ids:
            failures.append(f"{path.name} lane{stream_id} sentence IDs do not align")
    return {"path": str(path), "metrics": metrics, "drops": drops, "failures": failures}
```

`evaluate_runs(paths)` must require exactly three runs, retain each run's lane drops, sum every global
`offline_rtf.total` and `count`, calculate combined RTF, and pass only when every drop is below 10 and
combined RTF is at most 0.471. Implement the aggregation as:

```python
def evaluate_runs(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 3:
        raise ValueError("exactly three run directories are required")
    runs = [evaluate_run(path) for path in paths]
    failures = [failure for run in runs for failure in run["failures"]]
    total = 0.0
    count = 0
    for run in runs:
        metrics = run["metrics"]
        rtf = metrics.get("offline_rtf", {})
        total += float(rtf.get("total", 0.0))
        count += int(rtf.get("count", 0))
        for stream_id, dropped in enumerate(run["drops"]):
            if dropped >= 10:
                failures.append(
                    f"{Path(run['path']).name} lane{stream_id} dropped_frames={dropped}"
                )
    combined = total / count if count else float("inf")
    if combined > 0.471:
        failures.append(f"combined offline_rtf={combined:.6f} exceeds 0.471")
    return {
        "passed": not failures,
        "combined_offline_rtf": combined,
        "offline_rtf_total": total,
        "offline_rtf_count": count,
        "runs": runs,
        "failures": failures,
    }
```

Render and persist results with:

```python
def render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# 多进程 Zipformer + 单共享 Qwen3-ASR 验证报告",
        "",
        "## 拓扑",
        "",
        "1 个主 Python 进程、4 个 lane Python 子进程、1 个 llama-multi-asr。",
        "",
        "## 性能结论",
        "",
        f"- 验收：{'PASS' if result['passed'] else 'FAIL'}",
        f"- 三次合并 offline RTF：{result['combined_offline_rtf']:.6f}",
        f"- offline 请求数：{result['offline_rtf_count']}",
        "",
        "| Run | lane0 drops | lane1 drops | lane2 drops | lane3 drops | online avg/max ms | RTF avg/max | loop lag avg/max ms |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for run in result["runs"]:
        metrics = run["metrics"]
        online = metrics["online_frame_latency"]
        rtf = metrics["offline_rtf"]
        lag = metrics["event_loop_lag"]
        drops = run["drops"]
        lines.append(
            f"| {Path(run['path']).name} | {drops[0]} | {drops[1]} | {drops[2]} | {drops[3]} "
            f"| {online['average_ms']:.3f}/{online['max_ms']:.3f} "
            f"| {rtf['average']:.6f}/{rtf['max']:.6f} "
            f"| {lag['average_ms']:.3f}/{lag['max_ms']:.3f} |"
        )
    lines.extend(["", "## 共享服务 timings", ""])
    for run in result["runs"]:
        shared = run["metrics"]["offline_shared_timings"]
        values = ", ".join(
            f"{field}={shared[field]['average_ms']:.3f}/{shared[field]['max_ms']:.3f}ms"
            for field in TIMING_FIELDS
        )
        lines.append(f"- `{Path(run['path']).name}`: {values}")
    lines.extend(["", "## 功能检查", ""])
    lines.append("四路结果非空、model_outputs 六字段、Zipformer/Qwen 句 ID 对齐。")
    if result["failures"]:
        lines.extend(["", "## 失败项", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs=3, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_runs([path.resolve() for path in args.runs])
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    return 0 if result["passed"] else 1
```

Implement CLI:

```text
python scripts/evaluate_shared_qwen_runs.py RUN1 RUN2 RUN3 \
  --json-output runs/shared-qwen-evaluation.json \
  --markdown-output docs/superpowers/reports/2026-07-11-multi-process-single-qwen-report.md
```

The Markdown renderer must include topology, run paths, a per-run/per-lane drop table, combined RTF,
online latency, RTF max, five shared timings, event-loop lag, result validation, and a final PASS/FAIL.
Exit 1 on FAIL after still writing both outputs.

- [ ] **Step 4: Implement one-request real smoke CLI**

Create `scripts/smoke_shared_qwen.py` with this control flow:

```python
from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import numpy as np

from multi_stream_asr.asr.offline.qwen3asr import Qwen3AsrClient
from multi_stream_asr.asr.offline.shared_server import SharedQwenServerManager
from multi_stream_asr.audio.local_file import validate_wav_16k_mono_pcm
from multi_stream_asr.config import AppConfig


def read_prefix(path: Path, seconds: float) -> np.ndarray:
    validate_wav_16k_mono_pcm(path)
    with wave.open(str(path), "rb") as wav_file:
        raw = wav_file.readframes(min(wav_file.getnframes(), int(16000 * seconds)))
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


async def run(path: Path, seconds: float) -> int:
    config = AppConfig()
    manager = SharedQwenServerManager(config)
    spec = None
    try:
        spec = await asyncio.to_thread(manager.start)
        result = await Qwen3AsrClient(shared=config.shared_qwen).recognize(
            read_prefix(path, seconds), prompt=""
        )
        payload = {
            "text": result.text,
            "audio_seconds": result.audio_seconds,
            "rtf": result.rtf,
            "timings": result.timings,
            "command": spec.command,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if result.text and result.timings else 1
    finally:
        await asyncio.to_thread(manager.stop)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    return asyncio.run(run(args.wav.expanduser().resolve(), args.seconds))


if __name__ == "__main__":
    raise SystemExit(main())
```

This script must:

1. Parse `--wav` and optional `--seconds` (default 10.0).
2. Validate 16 kHz mono PCM WAV and read at most 10 seconds.
3. Construct `AppConfig`, `SharedQwenServerManager`, and `Qwen3AsrClient`.
4. Start service, submit one request, print JSON containing text, audio_seconds, rtf, timings, and command.
5. Always stop service in `finally`.
6. Return nonzero for empty text, missing timings, or any exception.

- [ ] **Step 5: Run tests and fake smoke**

```bash
.venv313/bin/pytest tests/test_evaluate_shared_qwen_runs.py -q
.venv313/bin/python -m multi_stream_asr.main --headless --fake --lanes 4 --verbose
latest=$(find runs -maxdepth 1 -type d -name 'process-*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
.venv313/bin/python -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); m=json.loads((p/"metrics.json").read_text()); assert m["metrics_schema_version"] == 3; assert len(m["lanes"]) == 4; assert "offline_shared_timings" in m' "$latest"
```

Expected: tests and fake smoke pass; fake shared timing count is zero.

- [ ] **Step 6: Commit**

```bash
git add scripts/smoke_shared_qwen.py scripts/evaluate_shared_qwen_runs.py tests/test_evaluate_shared_qwen_runs.py
git commit -m "test: add shared qwen validation tools"
```

---

### Task 7: Documentation, Local Verification, Remote Deployment, and Three Real Runs

**Files:**
- Modify: `docs/thread-model.md`
- Create from evaluator: `docs/superpowers/reports/2026-07-11-multi-process-single-qwen-report.md`
- Validate: local and `bianbu@10.0.91.51`

**Interfaces:**
- Consumes: all implementation and evaluation commands.
- Produces: final Chinese process/thread documentation, three synchronized real run directories, machine-readable evaluation, and committed report.

- [ ] **Step 1: Rewrite relevant thread-model sections**

Document exactly:

```text
4-lane stable topology:
  1 main Python process
  4 lane Python child processes
  1 llama-multi-asr process

Each lane:
  1 audio producer thread
  1 persistent Zipformer owner/consumer thread
  1 lane event loop

Main:
  shared Qwen lifecycle + watchdog
  IPC aggregation + WebSocket + schema v3 recorder
```

Replace all statements that say each lane owns or starts `llama-server`. Explain direct lane HTTP to port
8090, per-lane offline tail, cross-lane service concurrency, shared timings, and stop order. Preserve the
warning that total `/proc` task count includes runtime threads and is larger than project-owned threads.

- [ ] **Step 2: Run final local verification**

```bash
.venv313/bin/python -m compileall -q src scripts
.venv313/bin/pytest -q
.venv313/bin/python -m multi_stream_asr.main --headless --fake --lanes 4 --verbose
git diff --check
```

Expected: compile succeeds, all tests pass, fake exits 0, and diff check is clean.

- [ ] **Step 3: Commit documentation before board-generated report**

```bash
git add docs/thread-model.md
git commit -m "docs: describe shared qwen process topology"
```

- [ ] **Step 4: Sync code and create the remote Python 3.13 environment**

From the local feature worktree:

```bash
rsync -av --delete \
  --exclude '.git/' --exclude '.venv313/' --exclude 'runs/' \
  --exclude 'models/' --exclude '__pycache__/' --exclude '*.pyc' \
  ./ bianbu@10.0.91.51:~/Workspace/projects/multi-process-single-qwen/

ssh bianbu@10.0.91.51 '
  cd ~/Workspace/projects/multi-process-single-qwen &&
  mkdir -p models &&
  ln -sfn ~/Workspace/projects/multi-stream-asr/models/asr models/asr &&
  test -d .venv313 || virtualenv -p /usr/bin/python3.13 .venv313 &&
  . .venv313/bin/activate &&
  export UV_EXTRA_INDEX_URL="https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple" &&
  export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple &&
  export PIP_INDEX_URL=https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple &&
  export PIP_EXTRA_INDEX_URL="https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple https://mirrors.aliyun.com/pypi/simple" &&
  python -m pip install -e ".[dev,hardware]" &&
  python --version
'
```

Expected: Python 3.13.x; no model/runtime files copied into the new project.

- [ ] **Step 5: Run remote full tests and fake smoke**

```bash
ssh bianbu@10.0.91.51 '
  cd ~/Workspace/projects/multi-process-single-qwen &&
  . .venv313/bin/activate &&
  python -m compileall -q src scripts &&
  pytest -q &&
  python -m multi_stream_asr.main --headless --fake --lanes 4 --verbose
'
```

Expected: all tests pass, fake exits 0, schema v3 run has four lane artifacts and empty shared timing count.

- [ ] **Step 6: Run one real q40/8192 smoke request**

```bash
ssh bianbu@10.0.91.51 '
  cd ~/Workspace/projects/multi-process-single-qwen &&
  . .venv313/bin/activate &&
  python scripts/smoke_shared_qwen.py \
    --wav ~/Music/wav16k/00_simple_news_16000hz_1ch.wav \
    --seconds 10
'
```

Expected: exit 0, non-empty Chinese text, all five timing values, command includes q40 and
`--n-ctx 8192`, no `llama-multi-asr` remains. Check `dmesg` for no OOM/exit 137.

- [ ] **Step 7: Run three real 4-lane tests and capture topology**

For each index 1, 2, and 3, run remotely:

```bash
cd ~/Workspace/projects/multi-process-single-qwen
. .venv313/bin/activate
before=$(find runs -maxdepth 1 -type d -name 'process-*' | sort)
python -m multi_stream_asr.main --headless --lanes 4 --verbose \
  > "/tmp/multi-process-single-qwen-run-${index}.log" 2>&1 &
app_pid=$!
for attempt in $(seq 1 120); do
  lane_count=$(pgrep -P "$app_pid" -f 'multiprocessing.spawn' | wc -l)
  qwen_count=$(pgrep -f 'multi-asr-runtime/bin/llama-multi-asr.*--port 8090' | wc -l)
  if [ "$lane_count" -eq 4 ] && [ "$qwen_count" -eq 1 ]; then
    pstree -ap "$app_pid" > "/tmp/multi-process-single-qwen-run-${index}.pstree"
    break
  fi
  sleep 1
done
wait "$app_pid"
status=$?
test "$status" -eq 0
after=$(find runs -maxdepth 1 -type d -name 'process-*' | sort)
run_dir=$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | tail -1)
test -n "$run_dir"
printf '%s\n' "$run_dir" >> /tmp/multi-process-single-qwen-run-dirs.txt
test -z "$(pgrep -f 'multi-asr-runtime/bin/llama-multi-asr.*--port 8090' || true)"
```

Do not alter configuration between runs. Send progress updates at least every 30 seconds and do not leave
the remote session running when the task ends.

- [ ] **Step 8: Evaluate remotely and sync all artifacts**

```bash
ssh bianbu@10.0.91.51 '
  cd ~/Workspace/projects/multi-process-single-qwen &&
  . .venv313/bin/activate &&
  set -- $(cat /tmp/multi-process-single-qwen-run-dirs.txt) &&
  python scripts/evaluate_shared_qwen_runs.py "$1" "$2" "$3" \
    --json-output runs/shared-qwen-evaluation.json \
    --markdown-output docs/superpowers/reports/2026-07-11-multi-process-single-qwen-report.md
'
```

Record the evaluator exit code even when it is 1; a performance FAIL is a valid measured result and must
not trigger parameter tuning.

Sync the three paths listed in the remote run-dir file, evaluation JSON, report, logs, and process trees to
the local worktree. Verify local/remote SHA-256 checksums match.

- [ ] **Step 9: Final verification and report commit**

```bash
.venv313/bin/pytest -q
.venv313/bin/python -m compileall -q src scripts
git diff --check
git status --short
git add docs/superpowers/reports/2026-07-11-multi-process-single-qwen-report.md
git commit -m "docs: report multi-process shared qwen validation"
```

Do not commit ignored `runs/`. Report the three local run paths, exact drop counts, combined RTF, PASS/FAIL,
test counts, process cleanup status, and any residual risk.

---

## Plan Self-Review

- **Spec coverage:** Tasks cover main-owned service lifecycle, lane direct clients, q40/8192, no PCM IPC,
  timing metrics, schema v3, fake/Web/Headless regressions, model/runtime reuse, three-run strict thresholds,
  report generation, and no-tuning behavior on failure.
- **Task boundaries:** Tasks 1-4 add independently testable primitives without removing the old runtime;
  Task 5 is the single atomic ownership cutover; Tasks 6-7 make validation reproducible and run it.
- **Type consistency:** Final code uses `AppConfig.shared_qwen`, `LaneConfig(stream_id)`,
  `Qwen3AsrClient(shared=config.shared_qwen)`, `SharedTimingsTracker`, and
  `offline_shared_timings` consistently.
- **No placeholders:** Every test and implementation step contains concrete names, values, commands,
  expected behavior, and complete code for the non-obvious cases.
