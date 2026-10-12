# Single-Process Share-Qwen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-lane dedicated llama-server offline backend with the single shared `llama-multi-asr` service (one weight copy, server-side orchestrator queue), verified by a real 4-lane headless run on the K3 board with a function/performance report.

**Architecture:** The Python app (single process, per-lane producer/consumer threads from commit 8522bf4) launches ONE `llama-multi-asr` subprocess instead of N `llama-server` subprocesses. All lanes POST to `http://127.0.0.1:8090/v1/chat/completions`; queuing is server-side (`--n-parallel 4`). Client drops `id_slot`/slot-erase, parses the new `multi_asr_timings` response field into a new metrics section `offline_shared_timings`.

**Tech Stack:** Python 3.13 (board `.venv313`), httpx, pytest + pytest-asyncio, prebuilt RISC-V binary `llama-multi-asr` (from local `/home/liangff/Workspace/projects/multi-asr-test`), q40 gguf + dynq ONNX already on board.

**Spec:** `docs/superpowers/specs/2026-07-10-single-process-share-qwen-design.md`

## Global Constraints

- Base commit: `8522bf4240f5af9f3fb42d5240e2bb738a51d4b5`; branch: `single-process-share-qwen`; local worktree `~/Workspace/projects/multi-stream-asr-share-qwen`.
- Board: `bianbu@10.0.91.51`. Do NOT touch `~/Workspace/projects/multi-stream-asr` on the board (that is main's rsync target). Board work dir: `~/Workspace/projects/multi-stream-asr-share-qwen`; runtime dir: `~/Workspace/projects/multi-asr-runtime`.
- Board Python: 3.13 only, project-local `.venv313` (per `~/.codex/AGENTS.md`).
- pip on board: `PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/`; sherpa-onnx as prebuilt wheel: `pip install sherpa-onnx --index-url https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple` (never compile).
- Models: reuse board `~/Workspace/sm-sdk/models/asr/qwen3asr/` (q40 gguf + dynq onnx). No large-file uploads unless the q40 smoke test (Task 2) fails.
- Shared service defaults: port 8090, encoder affinity `"8;9"` (2 threads, set via config.json flat lowercase `spacemit_ep_*` fields), decoder cores `10,11,12,13,14,15` (6 threads), `--n-parallel 4`, `--warmup` (~25 s; health = `GET /health` → `{"status":"ok"}`).
- All local test commands run inside the worktree with its venv: `cd ~/Workspace/projects/multi-stream-asr-share-qwen && python -m pytest ...` (bootstrap a `.venv` in Task 1).
- Commit after every green test cycle. Commit messages in English, conventional style (`feat:`, `test:`, `docs:`, `chore:`).

---

### Task 1: Create worktree, branch, and carry design docs

**Files:**
- Create: git worktree `~/Workspace/projects/multi-stream-asr-share-qwen` on new branch `single-process-share-qwen`
- Create (in worktree): `docs/superpowers/specs/2026-07-10-single-process-share-qwen-design.md`, `docs/superpowers/plans/2026-07-10-single-process-share-qwen.md` (copied from main checkout)

**Interfaces:**
- Consumes: main repo at `/home/liangff/Workspace/projects/multi-stream-asr`, commit `8522bf4`.
- Produces: a working directory all later tasks run in. All later local paths are relative to this worktree.

- [ ] **Step 1: Create the worktree from the base commit**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr
git worktree add ../multi-stream-asr-share-qwen -b single-process-share-qwen 8522bf4240f5af9f3fb42d5240e2bb738a51d4b5
```

Expected: `Preparing worktree (new branch 'single-process-share-qwen')`, `HEAD is now at 8522bf4 ...`

- [ ] **Step 2: Copy spec and plan into the branch and commit**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr-share-qwen
cp /home/liangff/Workspace/projects/multi-stream-asr/docs/superpowers/specs/2026-07-10-single-process-share-qwen-design.md docs/superpowers/specs/
cp /home/liangff/Workspace/projects/multi-stream-asr/docs/superpowers/plans/2026-07-10-single-process-share-qwen.md docs/superpowers/plans/
git add docs/superpowers
git commit -m "docs: carry share-qwen spec and plan onto dev branch"
```

- [ ] **Step 3: Bootstrap a local venv for running tests**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr-share-qwen
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Expected: install succeeds (sherpa-onnx is NOT required locally — unit tests that need it are skipped or mocked; check with next step).

- [ ] **Step 4: Verify the base test suite is green**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr-share-qwen && . .venv/bin/activate
python -m pytest -q
```

Expected: all collected tests pass (188 collected on main; similar count at 8522bf4). If sherpa-onnx import errors appear, install the wheel: `pip install sherpa-onnx --index-url https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple` (x86 wheel exists on that index; if not available locally, note which tests fail for this reason and confirm they are only zipformer-import ones — those also cannot run locally today and are validated on the board in Task 8).

---

### Task 2: Deploy runtime to board and q40 smoke test (risk gate)

This task validates the spec's #1 risk — q40 gguf compatibility with `llama-multi-asr` — BEFORE any code is written.

**Files:**
- Create (board): `~/Workspace/projects/multi-asr-runtime/{bin,lib,ort_lib}` (~56 MB rsync from local `multi-asr-test`)
- Create (board): `/tmp/multi-stream-asr/shared-qwen/config.json` (hand-generated for smoke only; Task 4 generates it from code)

**Interfaces:**
- Consumes: local `/home/liangff/Workspace/projects/multi-asr-test/{bin,lib,ort_lib,qwen3-asr-0.6B/config.json}`; board models at `~/Workspace/sm-sdk/models/asr/qwen3asr/`.
- Produces: board runtime dir used by `SharedQwenServerManager` (Task 4) and all board runs; GO/NO-GO decision on q40.

- [ ] **Step 1: Rsync the runtime (binary + libs only, no models/audio)**

```bash
ssh bianbu@10.0.91.51 'mkdir -p ~/Workspace/projects/multi-asr-runtime'
rsync -a /home/liangff/Workspace/projects/multi-asr-test/bin /home/liangff/Workspace/projects/multi-asr-test/lib /home/liangff/Workspace/projects/multi-asr-test/ort_lib bianbu@10.0.91.51:~/Workspace/projects/multi-asr-runtime/
ssh bianbu@10.0.91.51 'ls ~/Workspace/projects/multi-asr-runtime && du -sh ~/Workspace/projects/multi-asr-runtime'
```

Expected: `bin lib ort_lib`, ~56M total.

- [ ] **Step 2: Generate the smoke config.json on the board (reference config, paths patched, q40)**

```bash
scp /home/liangff/Workspace/projects/multi-asr-test/qwen3-asr-0.6B/config.json bianbu@10.0.91.51:/tmp/shared-qwen-ref.json
ssh bianbu@10.0.91.51 'mkdir -p /tmp/multi-stream-asr/shared-qwen && python3 - <<PY
import json, pathlib
ref = json.loads(pathlib.Path("/tmp/shared-qwen-ref.json").read_text())
mdir = pathlib.Path("~/Workspace/sm-sdk/models/asr/qwen3asr").expanduser()
ref["audio_model"]["frontend_model_path"] = str(mdir / "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx")
ref["audio_model"]["backend_model_path"] = str(mdir / "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx")
ref["text_model"]["model_path"] = str(mdir / "Qwen3-ASR-0.6B-text-q40.gguf")
pathlib.Path("/tmp/multi-stream-asr/shared-qwen/config.json").write_text(json.dumps(ref, indent=2))
print("wrote", ref["text_model"]["model_path"])
PY'
```

Expected: `wrote /home/bianbu/Workspace/sm-sdk/models/asr/qwen3asr/Qwen3-ASR-0.6B-text-q40.gguf`

- [ ] **Step 3: Launch the shared server with q40**

```bash
ssh bianbu@10.0.91.51 'pkill -f llama-multi-asr 2>/dev/null; pkill -x llama-server 2>/dev/null; sleep 1; cd ~/Workspace/projects/multi-asr-runtime && LD_LIBRARY_PATH=./lib:./ort_lib nohup ./bin/llama-multi-asr -m ~/Workspace/sm-sdk/models/asr/qwen3asr/Qwen3-ASR-0.6B-text-q40.gguf --smt-config-dir /tmp/multi-stream-asr/shared-qwen --host 127.0.0.1 --port 8090 --decoder-cores 10,11,12,13,14,15 --decoder-n-threads 6 --n-parallel 4 --warmup > /tmp/llama-multi-asr-smoke.log 2>&1 & echo started'
ssh bianbu@10.0.91.51 'for i in $(seq 1 45); do out=$(curl -s http://127.0.0.1:8090/health); [ -n "$out" ] && echo "$out" && exit 0; sleep 2; done; echo TIMEOUT; tail -30 /tmp/llama-multi-asr-smoke.log; exit 1'
```

Expected: `{"status":"ok"}` within ~90 s. On TIMEOUT: read the log; if it shows a gguf load/quantization error → **fallback**: rsync `qwen3-asr-0.6B/Qwen3-ASR-0.6B-text-q41.gguf` (506 MB) to the same board model dir, patch config.json/`-m` to q41, retry, and record the decision (Task 4's `_TEXT_MODEL_NAME` must then be `Qwen3-ASR-0.6B-text-q41.gguf`).

- [ ] **Step 4: One real recognition request through the OpenAI-compatible endpoint**

```bash
ssh bianbu@10.0.91.51 'python3 - <<PY
import base64, json, urllib.request
wav = open("/home/bianbu/Music/wav16k/00_simple_news_16000hz_1ch.wav", "rb").read()
payload = {"model": "qwen3-asr", "messages": [{"role": "user", "content": [
    {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
    {"type": "text", "text": "language Chinese<asr_text>"}]}],
    "max_tokens": 256, "temperature": 0}
req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
out = json.loads(urllib.request.urlopen(req, timeout=300).read())
print("TEXT:", out["choices"][0]["message"]["content"][:120])
print("TIMINGS:", out.get("multi_asr_timings"))
PY'
```

Expected: `TEXT:` non-empty Chinese transcription; `TIMINGS:` a dict with `queue_ms, encode_ms, prefill_ms, decode_ms, total_ms`. Both MUST hold to proceed.

- [ ] **Step 5: Stop the smoke server and record the gate result**

```bash
ssh bianbu@10.0.91.51 'pkill -f llama-multi-asr; sleep 1; pgrep -f llama-multi-asr || echo stopped'
```

Expected: `stopped`. Note in the task report: which gguf passed (q40 or fallback q41) and one sample of the timings dict.

---

### Task 3: Config — `SharedQwenConfig`, simplified `LaneConfig`

**Files:**
- Modify: `src/multi_stream_asr/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Tasks 4–7):
  - `SharedQwenConfig` frozen dataclass, fields: `host: str = "127.0.0.1"`, `port: int = 8090`, `runtime_dir: Path = Path("~/Workspace/projects/multi-asr-runtime").expanduser()`, `encoder_thread_num: int = 2`, `encoder_affinity: str = "8;9"`, `decoder_cores: str = "10,11,12,13,14,15"`, `decoder_threads: int = 6`, `n_parallel: int = 4`, `request_timeout_ms: int = 120000`
  - `AppConfig.offline: SharedQwenConfig` (replaces `OfflineServerConfig`)
  - `LaneConfig` frozen dataclass with the single field `stream_id: int`
  - `build_lane_configs(lanes: int) -> list[LaneConfig]`
  - REMOVED: `OfflineServerConfig`, `CoreAllocation`, `core_allocations_for_lanes`, `LaneConfig.{offline_port,gguf_cores,onnx_cores,thread_count}`, `SharedQwenConfig` has no per-lane ports.

- [ ] **Step 1: Rewrite `tests/test_config.py` for the new shape**

Replace the file's lane/port/core tests with:

```python
from pathlib import Path

import pytest

from multi_stream_asr.config import (
    AppConfig,
    LaneConfig,
    SharedQwenConfig,
    build_lane_configs,
)


def test_shared_qwen_config_defaults_match_board_best_split():
    shared = SharedQwenConfig()
    assert shared.host == "127.0.0.1"
    assert shared.port == 8090
    assert shared.runtime_dir == Path("~/Workspace/projects/multi-asr-runtime").expanduser()
    assert shared.encoder_thread_num == 2
    assert shared.encoder_affinity == "8;9"
    assert shared.decoder_cores == "10,11,12,13,14,15"
    assert shared.decoder_threads == 6
    assert shared.n_parallel == 4
    assert shared.request_timeout_ms == 120000


def test_app_config_offline_is_shared_qwen_config():
    config = AppConfig()
    assert isinstance(config.offline, SharedQwenConfig)


@pytest.mark.parametrize("lanes", [2, 4, 6, 8])
def test_build_lane_configs_yields_sequential_stream_ids(lanes: int):
    lane_configs = build_lane_configs(lanes)
    assert [lane.stream_id for lane in lane_configs] == list(range(lanes))
    assert all(isinstance(lane, LaneConfig) for lane in lane_configs)


def test_app_config_rejects_unsupported_lane_counts():
    with pytest.raises(ValueError, match="lanes must be one of"):
        AppConfig(lanes=3)
```

Keep any existing tests in that file that do not reference `offline_port` / `gguf_cores` / `onnx_cores` / `thread_count` / `core_allocations_for_lanes` / `base_port` (e.g. `segment_cache_seconds` tests) unchanged.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'SharedQwenConfig'`

- [ ] **Step 3: Implement in `src/multi_stream_asr/config.py`**

Delete `OfflineServerConfig`, `CoreAllocation`, `core_allocations_for_lanes`; replace `LaneConfig` and add `SharedQwenConfig`:

```python
@dataclass(frozen=True)
class SharedQwenConfig:
    host: str = "127.0.0.1"
    port: int = 8090
    runtime_dir: Path = Path("~/Workspace/projects/multi-asr-runtime").expanduser()
    encoder_thread_num: int = 2
    encoder_affinity: str = "8;9"
    decoder_cores: str = "10,11,12,13,14,15"
    decoder_threads: int = 6
    n_parallel: int = 4
    request_timeout_ms: int = 120000


@dataclass(frozen=True)
class LaneConfig:
    stream_id: int
```

In `AppConfig`: `offline: SharedQwenConfig = field(default_factory=SharedQwenConfig)`. Replace `lane_configs` property body and `build_lane_configs`:

```python
    @property
    def lane_configs(self) -> list[LaneConfig]:
        return build_lane_configs(self.lanes)


def build_lane_configs(lanes: int) -> list[LaneConfig]:
    if lanes not in SUPPORTED_LANES:
        raise ValueError(f"lanes must be one of {SUPPORTED_LANES}, got {lanes}")
    return [LaneConfig(stream_id=stream_id) for stream_id in range(lanes)]
```

- [ ] **Step 4: Run config tests**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS. (Full-suite is still red — `llama_server.py`/`qwen3asr.py` reference deleted names; they are fixed in Tasks 4–5. Do NOT run the full suite yet.)

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/config.py tests/test_config.py
git commit -m "feat: replace per-lane offline config with SharedQwenConfig"
```

---

### Task 4: `shared_server.py` — single shared-service manager

**Files:**
- Create: `src/multi_stream_asr/asr/offline/shared_server.py`
- Delete: `src/multi_stream_asr/asr/offline/llama_server.py`
- Create: `tests/test_shared_server.py`
- Delete: `tests/test_llama_server.py`

**Interfaces:**
- Consumes: `AppConfig`, `SharedQwenConfig` (Task 3); `run_best_effort` from `multi_stream_asr.utils.process`.
- Produces (used by Task 7):
  - `SharedQwenServerManager(config: AppConfig, *, health_timeout_seconds: float = 120.0, poll_interval_seconds: float = 1.0, stop_timeout_seconds: float = 5.0)` with sync methods `start_all() -> list[SharedQwenServerSpec]`, `stop_all() -> None`, `cleanup_before_start() -> None` (same duck-typed surface `manager.py` already calls — `pipeline/manager.py` needs NO changes).
  - `build_shared_config(config: AppConfig) -> dict[str, object]`
  - `build_launch_command(config: AppConfig, config_dir: Path) -> list[str]`
  - `SHARED_CONFIG_DIR = Path("/tmp/multi-stream-asr/shared-qwen")`
  - `_TEXT_MODEL_NAME = "Qwen3-ASR-0.6B-text-q40.gguf"` (or q41 if Task 2 fell back — use whichever Task 2 recorded).

- [ ] **Step 1: Write `tests/test_shared_server.py`**

Port the process-teardown fakes from the old test file and add config-alignment assertions (the three flat lowercase `spacemit_ep_*` fields are the silent-failure risk called out in the spec):

```python
import json
import subprocess
from pathlib import Path

import pytest

from multi_stream_asr.config import AppConfig, ModelConfig, SharedQwenConfig
from multi_stream_asr.asr.offline.shared_server import (
    SHARED_CONFIG_DIR,
    SharedQwenServerManager,
    build_launch_command,
    build_shared_config,
)


class FakeProcess:
    def __init__(self, pid: int, *, wait_timeout_until_kill: bool = False):
        self.pid = pid
        self.wait_timeout_until_kill = wait_timeout_until_kill
        self.terminate_calls = 0
        self.kill_calls = 0
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout=None):
        if self.wait_timeout_until_kill and self.kill_calls == 0:
            raise subprocess.TimeoutExpired(cmd="llama-multi-asr", timeout=timeout or 0)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="llama-multi-asr", timeout=timeout or 0)
        return self.returncode


def test_build_shared_config_aligns_with_multi_asr_test_reference():
    config = AppConfig()
    rendered = build_shared_config(config)

    assert rendered["architectures"] == ["Qwen3ASRForConditionalGeneration"]
    assert isinstance(rendered["support_languages"], list)
    assert "Chinese" in rendered["support_languages"]

    audio = rendered["audio_model"]
    # flat lowercase fields — llama-multi-asr silently falls back to 4 threads without them
    assert audio["spacemit_ep_intra_thread_num"] == 2
    assert audio["spacemit_ep_inter_thread_num"] == 1
    assert audio["spacemit_ep_intra_thread_affinity"] == "8;9"
    assert "ep_config" not in audio
    assert audio["frontend_model_path"].endswith("Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx")
    assert audio["backend_model_path"].endswith("Qwen3-ASR-0.6B-encoder-backend.dynq.onnx")
    assert audio["num_mel_bins"] == 128
    assert audio["special_tokens"]["audio_token_id"] == 151676

    assert rendered["text_model"]["model_path"].endswith(".gguf")
    assert rendered["text_model"]["context_size"] == 65536
    assert rendered["prompt"]["asr_text_tag"] == "<asr_text>"


def test_build_shared_config_uses_configured_encoder_split():
    config = AppConfig(offline=SharedQwenConfig(encoder_thread_num=4, encoder_affinity="8;9;10;11"))
    audio = build_shared_config(config)["audio_model"]
    assert audio["spacemit_ep_intra_thread_num"] == 4
    assert audio["spacemit_ep_intra_thread_affinity"] == "8;9;10;11"


def test_build_launch_command_matches_run_server_invocation(tmp_path: Path):
    config = AppConfig()
    command = build_launch_command(config, tmp_path)

    binary = config.offline.runtime_dir / "bin" / "llama-multi-asr"
    assert command[0] == str(binary)
    assert command[command.index("-m") + 1].endswith(".gguf")
    assert command[command.index("--smt-config-dir") + 1] == str(tmp_path)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8090"
    assert command[command.index("--decoder-cores") + 1] == "10,11,12,13,14,15"
    assert command[command.index("--decoder-n-threads") + 1] == "6"
    assert command[command.index("--n-parallel") + 1] == "4"
    assert "--warmup" in command
    # dedicated-scheme flags must be gone
    for flag in ("-t", "-tb", "--media-backend", "--slot-save-path", "-c"):
        assert flag not in command


def test_write_config_does_not_touch_model_directory(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"preserve": true}')
    config = AppConfig(models=ModelConfig(offline_model_dir=model_dir))
    manager = SharedQwenServerManager(config)
    config_dir = tmp_path / "generated"

    manager._write_config(config_dir)

    rendered = json.loads((config_dir / "config.json").read_text())
    assert rendered["text_model"]["model_path"].startswith(str(model_dir))
    assert (model_dir / "config.json").read_text() == '{"preserve": true}'


def test_cleanup_before_start_kills_leftover_shared_and_dedicated_servers(monkeypatch: pytest.MonkeyPatch):
    commands: list[list[str]] = []

    def fake_run_best_effort(command):
        commands.append(command)
        if command[0] == "pgrep":
            return subprocess.CompletedProcess(command, 1, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("multi_stream_asr.asr.offline.shared_server.run_best_effort", fake_run_best_effort)
    monkeypatch.setattr("multi_stream_asr.asr.offline.shared_server.time.sleep", lambda s: None)

    SharedQwenServerManager(AppConfig()).cleanup_before_start()

    assert ["pkill", "-f", "llama-multi-asr"] in commands
    assert ["pkill", "-x", "llama-server"] in commands
    assert ["spacemit-tcm-smi", "-c"] in commands


def test_start_all_launches_single_process_and_waits_health(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = AppConfig(models=ModelConfig(offline_model_dir=tmp_path))
    manager = SharedQwenServerManager(config, health_timeout_seconds=2.0, poll_interval_seconds=0.01)
    launched: list[dict] = []
    process = FakeProcess(pid=4321)

    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.SHARED_CONFIG_DIR", tmp_path / "cfg"
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.subprocess.Popen",
        lambda command, **kwargs: launched.append({"command": command, **kwargs}) or process,
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.is_server_healthy", lambda url, timeout: True
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.TMP_ROOT", tmp_path
    )

    specs = manager.start_all()

    assert len(specs) == 1
    assert len(launched) == 1
    assert specs[0].port == 8090
    env = launched[0]["env"]
    runtime = config.offline.runtime_dir
    assert str(runtime / "lib") in env["LD_LIBRARY_PATH"]
    assert str(runtime / "ort_lib") in env["LD_LIBRARY_PATH"]
    assert (tmp_path / "cfg" / "config.json").exists()

    manager.stop_all()
    assert process.terminate_calls == 1


def test_stop_all_escalates_to_kill_on_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = AppConfig(models=ModelConfig(offline_model_dir=tmp_path))
    manager = SharedQwenServerManager(config, health_timeout_seconds=2.0, poll_interval_seconds=0.01, stop_timeout_seconds=0.01)
    process = FakeProcess(pid=4321, wait_timeout_until_kill=True)

    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.SHARED_CONFIG_DIR", tmp_path / "cfg"
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.subprocess.Popen",
        lambda command, **kwargs: process,
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.is_server_healthy", lambda url, timeout: True
    )
    monkeypatch.setattr("multi_stream_asr.asr.offline.shared_server.TMP_ROOT", tmp_path)

    manager.start_all()
    manager.stop_all()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1


def test_start_all_raises_and_stops_on_health_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = AppConfig(models=ModelConfig(offline_model_dir=tmp_path))
    manager = SharedQwenServerManager(config, health_timeout_seconds=0.05, poll_interval_seconds=0.01, stop_timeout_seconds=0.01)
    process = FakeProcess(pid=4321)

    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.SHARED_CONFIG_DIR", tmp_path / "cfg"
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.subprocess.Popen",
        lambda command, **kwargs: process,
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.is_server_healthy", lambda url, timeout: False
    )
    monkeypatch.setattr("multi_stream_asr.asr.offline.shared_server.TMP_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="health"):
        manager.start_all()
    assert process.terminate_calls == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_shared_server.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'multi_stream_asr.asr.offline.shared_server'`

- [ ] **Step 3: Implement `src/multi_stream_asr/asr/offline/shared_server.py` and delete the old module**

```python
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from multi_stream_asr.config import AppConfig
from multi_stream_asr.utils.process import run_best_effort


TMP_ROOT = Path("/tmp")
SHARED_CONFIG_DIR = Path("/tmp/multi-stream-asr/shared-qwen")
_TEXT_MODEL_NAME = "Qwen3-ASR-0.6B-text-q40.gguf"
_FRONTEND_MODEL_NAME = "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx"
_BACKEND_MODEL_NAME = "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx"
_SUPPORT_LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]


@dataclass(frozen=True)
class SharedQwenServerSpec:
    port: int
    config_dir: Path
    log_path: Path
    pid_path: Path
    env: dict[str, str]
    command: list[str]


def build_shared_config(config: AppConfig) -> dict[str, object]:
    model_dir = config.models.offline_model_dir
    shared = config.offline
    return {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "support_languages": list(_SUPPORT_LANGUAGES),
        "audio_model": {
            "frontend_model_path": str(model_dir / _FRONTEND_MODEL_NAME),
            "backend_model_path": str(model_dir / _BACKEND_MODEL_NAME),
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
            "model_path": str(model_dir / _TEXT_MODEL_NAME),
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


def build_launch_command(config: AppConfig, config_dir: Path) -> list[str]:
    shared = config.offline
    binary = shared.runtime_dir / "bin" / "llama-multi-asr"
    return [
        str(binary),
        "-m",
        str(config.models.offline_model_dir / _TEXT_MODEL_NAME),
        "--smt-config-dir",
        str(config_dir),
        "--host",
        shared.host,
        "--port",
        str(shared.port),
        "--decoder-cores",
        shared.decoder_cores,
        "--decoder-n-threads",
        str(shared.decoder_threads),
        "--n-parallel",
        str(shared.n_parallel),
        "--warmup",
    ]


def is_server_healthy(url: str, timeout: float) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False
    return payload.get("status") == "ok"


class SharedQwenServerManager:
    def __init__(
        self,
        config: AppConfig,
        *,
        health_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 1.0,
        stop_timeout_seconds: float = 5.0,
    ) -> None:
        self._config = config
        self._health_timeout_seconds = health_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_timeout_seconds = stop_timeout_seconds
        self._started: list[tuple[SharedQwenServerSpec, subprocess.Popen[str], object]] = []

    def cleanup_before_start(self) -> None:
        run_best_effort(["pkill", "-f", "llama-multi-asr"])
        run_best_effort(["pkill", "-x", "llama-server"])
        time.sleep(1)
        if run_best_effort(["pgrep", "-f", "llama-multi-asr"]).returncode == 0:
            run_best_effort(["pkill", "-9", "-f", "llama-multi-asr"])
        run_best_effort(["spacemit-tcm-smi", "-c"])

    def start_all(self) -> list[SharedQwenServerSpec]:
        self.cleanup_before_start()
        spec = self._build_spec()
        try:
            self._write_config(spec.config_dir)
            log_handle = spec.log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                spec.command,
                env=spec.env,
                stdout=log_handle,
                stderr=log_handle,
                text=True,
            )
            spec.pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
            self._started.append((spec, process, log_handle))
            self._wait_for_health(spec.port)
            return [spec]
        except Exception:
            self.stop_all()
            raise

    def stop_all(self) -> None:
        while self._started:
            spec, process, log_handle = self._started.pop()
            try:
                if process.poll() is not None:
                    continue
                try:
                    process.terminate()
                except ProcessLookupError:
                    continue
                try:
                    process.wait(timeout=self._stop_timeout_seconds)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        continue
                    try:
                        process.wait(timeout=self._stop_timeout_seconds)
                    except (subprocess.TimeoutExpired, ProcessLookupError):
                        continue
            finally:
                log_handle.close()

    def _build_spec(self) -> SharedQwenServerSpec:
        shared = self._config.offline
        env = os.environ.copy()
        lib_paths = f"{shared.runtime_dir / 'lib'}:{shared.runtime_dir / 'ort_lib'}"
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib_paths}:{existing}" if existing else lib_paths
        return SharedQwenServerSpec(
            port=shared.port,
            config_dir=SHARED_CONFIG_DIR,
            log_path=TMP_ROOT / "llama-multi-asr.log",
            pid_path=TMP_ROOT / "llama-multi-asr.pid",
            env=env,
            command=build_launch_command(self._config, SHARED_CONFIG_DIR),
        )

    def _write_config(self, config_dir: Path) -> None:
        config_dir.mkdir(parents=True, exist_ok=True)
        payload = build_shared_config(self._config)
        (config_dir / "config.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _wait_for_health(self, port: int) -> None:
        url = f"http://{self._config.offline.host}:{port}/health"
        deadline = time.monotonic() + self._health_timeout_seconds
        while time.monotonic() < deadline:
            if is_server_healthy(url, timeout=min(self._poll_interval_seconds, 2.0)):
                return
            time.sleep(self._poll_interval_seconds)
        raise RuntimeError(f"llama-multi-asr health timed out on port {port}")
```

Note the `start_all` config-dir monkeypatch in the tests patches the module-level `SHARED_CONFIG_DIR` — `_build_spec` must read the module attribute at call time (as written above, it does: it references the global by name).

Then delete the old files:

```bash
git rm src/multi_stream_asr/asr/offline/llama_server.py tests/test_llama_server.py
```

If Task 2 fell back to q41: set `_TEXT_MODEL_NAME = "Qwen3-ASR-0.6B-text-q41.gguf"`.

- [ ] **Step 4: Run the new tests**

Run: `python -m pytest tests/test_shared_server.py -q`
Expected: PASS (full suite still red until Tasks 5–7).

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/asr/offline/shared_server.py tests/test_shared_server.py
git commit -m "feat: add SharedQwenServerManager for single llama-multi-asr service"
```

---

### Task 5: Client — shared endpoint, no slots, timings parsing

**Files:**
- Modify: `src/multi_stream_asr/asr/offline/qwen3asr.py`
- Modify: `src/multi_stream_asr/asr/offline/base.py`
- Test: `tests/test_qwen3asr.py`

**Interfaces:**
- Consumes: `SharedQwenConfig` (Task 3).
- Produces (used by Tasks 6–7):
  - `Qwen3AsrClient(shared: SharedQwenConfig)` — no lane arg; all lanes may share one instance or create one each (stateless).
  - `Qwen3AsrClient.recognize(pcm: np.ndarray, prompt: str) -> OfflineRecognitionResult` — signature unchanged (Protocol `OfflineRecognizer` untouched); `prompt` arg is accepted but the request always sends the fixed text `QWEN_USER_PROMPT`.
  - `QWEN_USER_PROMPT = "language Chinese<asr_text>"`
  - `build_chat_request(pcm: np.ndarray, max_tokens: int = 256) -> dict` — payload aligned to the proven `rtf_test.py` format: `model: "qwen3-asr"`, single user message `[input_audio, text]`, NO system message, NO `id_slot`.
  - `OfflineRecognitionResult` gains field `timings: dict[str, float] | None = None` (parsed from response `multi_asr_timings`).
  - REMOVED: `build_slot_erase_url`, `QWEN_SLOT_ID`, `QWEN_SYSTEM_PROMPT`, slot-erase POST, `slot_cleanup_timeout_ms` usage.

- [ ] **Step 1: Rewrite `tests/test_qwen3asr.py`**

Keep `FakeResponse`/`FakeAsyncClient` and the `encode_wav_base64` / `clean_qwen_text` tests exactly as they are (they don't touch removed APIs). Replace the payload/recognize/slot tests with:

```python
from multi_stream_asr.config import SharedQwenConfig
from multi_stream_asr.asr.offline.qwen3asr import (
    QWEN_USER_PROMPT,
    Qwen3AsrClient,
    build_chat_request,
    clean_qwen_text,
    encode_wav_base64,
)


def test_build_chat_request_matches_multi_asr_test_format():
    pcm = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)

    payload = build_chat_request(pcm)

    assert payload["model"] == "qwen3-asr"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 256
    assert "id_slot" not in payload
    assert len(payload["messages"]) == 1
    message = payload["messages"][0]
    assert message["role"] == "user"
    assert message["content"][0] == {
        "type": "input_audio",
        "input_audio": {"data": encode_wav_base64(pcm), "format": "wav"},
    }
    assert message["content"][1] == {"type": "text", "text": "language Chinese<asr_text>"}


@pytest.mark.asyncio
async def test_recognize_posts_once_and_parses_timings(monkeypatch: pytest.MonkeyPatch):
    client = FakeAsyncClient(
        post_responses=[
            FakeResponse(
                200,
                {
                    "choices": [{"message": {"content": "<asr_text> hello world </asr_text>"}}],
                    "multi_asr_timings": {
                        "queue_ms": 12.5,
                        "encode_ms": 900.0,
                        "prefill_ms": 300.0,
                        "decode_ms": 800.0,
                        "total_ms": 2012.5,
                    },
                },
            ),
        ]
    )
    monkeypatch.setattr("multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient", lambda: client)
    recognizer = Qwen3AsrClient(shared=SharedQwenConfig(port=9090, request_timeout_ms=2000))
    pcm = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)

    result = await recognizer.recognize(pcm, prompt="ignored partial text")

    assert result.text == "hello world"
    assert result.timings == {
        "queue_ms": 12.5,
        "encode_ms": 900.0,
        "prefill_ms": 300.0,
        "decode_ms": 800.0,
        "total_ms": 2012.5,
    }
    # exactly one HTTP call: no slot-erase follow-up
    assert len(client.calls) == 1
    method, url, body, timeout = client.calls[0]
    assert url == "http://127.0.0.1:9090/v1/chat/completions"
    assert timeout == 2.0
    assert body == build_chat_request(pcm)


@pytest.mark.asyncio
async def test_recognize_tolerates_missing_timings(monkeypatch: pytest.MonkeyPatch):
    client = FakeAsyncClient(
        post_responses=[FakeResponse(200, {"text": "<asr_text>ok</asr_text>"})]
    )
    monkeypatch.setattr("multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient", lambda: client)
    recognizer = Qwen3AsrClient(shared=SharedQwenConfig())

    result = await recognizer.recognize(np.asarray([0.25], dtype=np.float32), prompt="")

    assert result.text == "ok"
    assert result.timings is None
    assert client.calls[0][3] == 120.0


@pytest.mark.asyncio
async def test_recognize_raises_on_server_busy(monkeypatch: pytest.MonkeyPatch):
    client = FakeAsyncClient(
        post_responses=[FakeResponse(503, {"error": "server busy: queue full"})]
    )
    monkeypatch.setattr("multi_stream_asr.asr.offline.qwen3asr.httpx.AsyncClient", lambda: client)
    recognizer = Qwen3AsrClient(shared=SharedQwenConfig())

    with pytest.raises(RuntimeError):
        await recognizer.recognize(np.asarray([0.25], dtype=np.float32), prompt="")
```

Keep imports (`asyncio, base64, io, wave, dataclass, numpy, pytest`) at top as before; drop `LaneConfig, OfflineServerConfig, build_slot_erase_url` imports.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_qwen3asr.py -q`
Expected: FAIL — `ImportError: cannot import name 'QWEN_USER_PROMPT'`

- [ ] **Step 3: Implement**

In `src/multi_stream_asr/asr/offline/base.py` add the field:

```python
@dataclass(frozen=True)
class OfflineRecognitionResult:
    text: str
    elapsed_ms: float
    audio_seconds: float
    rtf: float
    raw_response: dict[str, Any] | None = None
    timings: dict[str, float] | None = None
```

Rewrite `qwen3asr.py`'s request/client parts (keep `encode_wav_bytes`, `encode_wav_base64`, `clean_qwen_text` unchanged; remove `QWEN_SLOT_ID`, `QWEN_SYSTEM_PROMPT`, `build_slot_erase_url`):

```python
from multi_stream_asr.config import SharedQwenConfig

QWEN_USER_PROMPT = "language Chinese<asr_text>"
QWEN_MAX_TOKENS = 256


def build_chat_request(pcm: np.ndarray, max_tokens: int = QWEN_MAX_TOKENS) -> dict[str, Any]:
    return {
        "model": "qwen3-asr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encode_wav_base64(pcm),
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": QWEN_USER_PROMPT},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def parse_timings(payload: dict[str, Any]) -> dict[str, float] | None:
    raw = payload.get("multi_asr_timings")
    if not isinstance(raw, dict):
        return None
    timings = {
        key: float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return timings or None


class Qwen3AsrClient:
    def __init__(self, *, shared: SharedQwenConfig) -> None:
        self._base_url = f"http://{shared.host}:{shared.port}"
        self._request_timeout_ms = shared.request_timeout_ms

    async def recognize(self, pcm: np.ndarray, prompt: str) -> OfflineRecognitionResult:
        started = time.monotonic()
        audio_seconds = float(np.asarray(pcm).size) / 16000.0
        payload = build_chat_request(pcm)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=self._request_timeout_ms / 1000.0,
            )
            response.raise_for_status()
            raw_response = response.json()
            text = clean_qwen_text(raw_response)

        elapsed_ms = (time.monotonic() - started) * 1000.0
        rtf = elapsed_ms / (audio_seconds * 1000.0) if audio_seconds > 0 else 0.0
        return OfflineRecognitionResult(
            text=text,
            elapsed_ms=elapsed_ms,
            audio_seconds=audio_seconds,
            rtf=rtf,
            raw_response=raw_response,
            timings=parse_timings(raw_response),
        )

    async def close(self) -> None:
        return None
```

Note on `test_recognize_raises_on_server_busy`: `FakeResponse.raise_for_status` raises `RuntimeError` on ≥400, so no extra client code is needed — a busy reply propagates as an exception into the existing pipeline offline-error path (per spec: no retry, no new handling at 4 lanes).

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_qwen3asr.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/multi_stream_asr/asr/offline/qwen3asr.py src/multi_stream_asr/asr/offline/base.py tests/test_qwen3asr.py
git commit -m "feat: point qwen3asr client at shared service and parse multi_asr_timings"
```

---

### Task 6: Metrics — `SharedTimingsTracker` through pipeline, manager, recorder

**Files:**
- Modify: `src/multi_stream_asr/log/metrics.py`
- Modify: `src/multi_stream_asr/pipeline/pipeline.py` (constructor + `_schedule_offline.run_offline`)
- Modify: `src/multi_stream_asr/pipeline/manager.py` (create/expose tracker, pass to pipelines)
- Modify: `src/multi_stream_asr/log/recorder.py` (`save` kwarg + metrics.json section)
- Test: `tests/test_buffers_metrics.py` (tracker), `tests/test_pipeline.py` (feed-through), `tests/test_manager_headless.py` (only if constructor changes break existing tests — no new assertions required there)

**Interfaces:**
- Consumes: `OfflineRecognitionResult.timings` (Task 5).
- Produces (used by Task 7):
  - `SharedTimingsTracker` in `multi_stream_asr.log.metrics` with `add(timings: dict[str, float]) -> None` and `as_dict() -> dict`; `as_dict()` returns `{"count": int, "queue_ms": {"average": float, "max": float}, "encode_ms": {...}, "prefill_ms": {...}, "decode_ms": {...}, "total_ms": {...}}`.
  - `LanePipeline.__init__(..., shared_timings_tracker: SharedTimingsTracker | None = None)`.
  - `PipelineManager.shared_timings_tracker` property.
  - `RunRecorder.save(..., shared_timings_tracker: SharedTimingsTracker | None = None)` writing metrics.json key `offline_shared_timings` and bumping `metrics_schema_version` to 3.

- [ ] **Step 1: Write the tracker test in `tests/test_buffers_metrics.py`** (append)

```python
from multi_stream_asr.log.metrics import SharedTimingsTracker


def test_shared_timings_tracker_aggregates_per_field():
    tracker = SharedTimingsTracker()
    tracker.add({"queue_ms": 10.0, "encode_ms": 100.0, "prefill_ms": 20.0, "decode_ms": 200.0, "total_ms": 330.0})
    tracker.add({"queue_ms": 30.0, "encode_ms": 300.0, "prefill_ms": 40.0, "decode_ms": 400.0, "total_ms": 770.0})

    payload = tracker.as_dict()

    assert payload["count"] == 2
    assert payload["queue_ms"] == {"average": 20.0, "max": 30.0}
    assert payload["encode_ms"] == {"average": 200.0, "max": 300.0}
    assert payload["total_ms"] == {"average": 550.0, "max": 770.0}


def test_shared_timings_tracker_ignores_unknown_and_missing_fields():
    tracker = SharedTimingsTracker()
    tracker.add({"queue_ms": 10.0, "bogus": 1.0})

    payload = tracker.as_dict()

    assert payload["count"] == 1
    assert payload["queue_ms"] == {"average": 10.0, "max": 10.0}
    assert payload["encode_ms"] == {"average": 0.0, "max": 0.0}
    assert "bogus" not in payload


def test_shared_timings_tracker_empty_as_dict():
    payload = SharedTimingsTracker().as_dict()
    assert payload["count"] == 0
    assert payload["total_ms"] == {"average": 0.0, "max": 0.0}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_buffers_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'SharedTimingsTracker'`

- [ ] **Step 3: Implement the tracker in `src/multi_stream_asr/log/metrics.py`** (append)

```python
_SHARED_TIMING_FIELDS = ("queue_ms", "encode_ms", "prefill_ms", "decode_ms", "total_ms")


@dataclass
class SharedTimingsTracker:
    samples: list[dict[str, float]] = field(default_factory=list)

    def add(self, timings: dict[str, float]) -> None:
        sample = {
            key: float(timings[key])
            for key in _SHARED_TIMING_FIELDS
            if key in timings
        }
        if sample:
            self.samples.append(sample)

    @property
    def count(self) -> int:
        return len(self.samples)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"count": self.count}
        for key in _SHARED_TIMING_FIELDS:
            values = [sample[key] for sample in self.samples if key in sample]
            payload[key] = {
                "average": sum(values) / len(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
        return payload
```

Run: `python -m pytest tests/test_buffers_metrics.py -q` → PASS.

- [ ] **Step 4: Feed-through test in `tests/test_pipeline.py`** (append; reuse that file's existing fake offline recognizer pattern — check its actual fixture names when editing)

```python
from multi_stream_asr.log.metrics import SharedTimingsTracker


@pytest.mark.asyncio
async def test_offline_result_timings_reach_shared_tracker():
    tracker = SharedTimingsTracker()

    class TimingsOfflineRecognizer:
        async def recognize(self, pcm, prompt):
            return OfflineRecognitionResult(
                text="final",
                elapsed_ms=10.0,
                audio_seconds=1.0,
                rtf=0.01,
                timings={"queue_ms": 5.0, "encode_ms": 1.0, "prefill_ms": 1.0, "decode_ms": 2.0, "total_ms": 9.0},
            )

        async def close(self):
            return None

    pipeline = make_pipeline(offline_recognizer=TimingsOfflineRecognizer(), shared_timings_tracker=tracker)
    # drive one endpointed sentence through the pipeline exactly like the
    # existing offline-final test in this file does, then:
    assert tracker.count == 1
    assert tracker.as_dict()["queue_ms"] == {"average": 5.0, "max": 5.0}
```

Adaptation note: `make_pipeline(...)` stands for however `tests/test_pipeline.py` constructs a `LanePipeline` in its existing offline-final test (a helper or inline construction). Copy that exact construction, add the two kwargs shown, and reuse its exact drive sequence (frames → endpoint → await completion). Do not invent a new driving mechanism.

- [ ] **Step 5: Run to verify failure**

Run: `python -m pytest tests/test_pipeline.py -q`
Expected: new test FAILS — `TypeError: ... unexpected keyword argument 'shared_timings_tracker'`

- [ ] **Step 6: Wire through pipeline, manager, recorder**

`pipeline.py` — constructor: add parameter `shared_timings_tracker: SharedTimingsTracker | None = None`, store `self.shared_timings_tracker = shared_timings_tracker or SharedTimingsTracker()` (import from `multi_stream_asr.log.metrics`). In `_schedule_offline.run_offline`, right after `self.rtf_tracker.add(...)`:

```python
            if result.timings:
                self.shared_timings_tracker.add(result.timings)
```

`manager.py` — in `__init__`: `self._shared_timings_tracker = SharedTimingsTracker()`. Add property:

```python
    @property
    def shared_timings_tracker(self) -> SharedTimingsTracker:
        return self._shared_timings_tracker
```

In `_start_recognition_locked`: create `shared_timings_tracker = SharedTimingsTracker()` beside `rtf_tracker = RtfTracker()`, pass `shared_timings_tracker=shared_timings_tracker` in the `self._pipeline_factory(...)` call, assign `self._shared_timings_tracker = shared_timings_tracker` in the same block that assigns `self._rtf_tracker`, and snapshot/restore it in the same places `previous_rtf_tracker` is handled (add `previous_shared_timings_tracker`).

`recorder.py` — `save(...)` gains kwarg `shared_timings_tracker: SharedTimingsTracker | None = None`; in `metrics_payload` set `"metrics_schema_version": 3` and add:

```python
            "offline_shared_timings": (
                shared_timings_tracker.as_dict()
                if shared_timings_tracker is not None
                else SharedTimingsTracker().as_dict()
            ),
```

(Import `SharedTimingsTracker` in recorder.py.)

- [ ] **Step 7: Run pipeline + metrics + any recorder tests**

Run: `python -m pytest tests/test_pipeline.py tests/test_buffers_metrics.py tests/test_manager_headless.py -q`
Expected: PASS (manager tests use duck-typed fakes and keyword-defaulted factories, so they should pass unmodified; if a `pipeline_factory` fake in `test_manager_headless.py` rejects the new kwarg, add `shared_timings_tracker=None` acceptance to that fake).

- [ ] **Step 8: Commit**

```bash
git add src/multi_stream_asr/log/metrics.py src/multi_stream_asr/pipeline/pipeline.py src/multi_stream_asr/pipeline/manager.py src/multi_stream_asr/log/recorder.py tests/test_buffers_metrics.py tests/test_pipeline.py tests/test_manager_headless.py
git commit -m "feat: record shared-qwen multi_asr_timings in metrics"
```

---

### Task 7: Runner wiring and full local suite

**Files:**
- Modify: `src/multi_stream_asr/headless/runner.py`
- Modify: `tests/test_manager_headless.py` / `tests/conftest.py` (only if red)

**Interfaces:**
- Consumes: `SharedQwenServerManager` (Task 4), `Qwen3AsrClient(shared=...)` (Task 5), `shared_timings_tracker` (Task 6).
- Produces: `python -m multi_stream_asr.main --headless [--fake] --lanes N` fully working against the shared backend; the whole test suite green.

- [ ] **Step 1: Rewire `headless/runner.py`**

- Replace import: `from multi_stream_asr.asr.offline.llama_server import LlamaServerManager` → `from multi_stream_asr.asr.offline.shared_server import SharedQwenServerManager`.
- Rename `_NoopLlamaServerManager` → `_NoopServerManager` (fake mode keeps not launching any server).
- In `_build_manager` real branch: `llama_server_manager=SharedQwenServerManager(self._config)` and

```python
            offline_recognizer_factory=lambda lane: Qwen3AsrClient(
                shared=self._config.offline,
            ),
```

- In `_save_results`, pass the new tracker through:

```python
            shared_timings_tracker=self.manager.shared_timings_tracker,
```

(as a new kwarg to `self._recorder.save(...)`).

Also grep for stragglers — every hit must be fixed or already renamed:

```bash
grep -rn "llama_server\|LlamaServerManager\|OfflineServerConfig\|offline_port\|gguf_cores\|onnx_cores\|id_slot\|slot_cleanup" src tests
```

Expected after fixes: only historical references inside `docs/` (fine) and the `_NoopServerManager` definition.

- [ ] **Step 2: Run the entire suite**

Run: `python -m pytest -q`
Expected: ALL PASS. Fix any missed call sites (most likely `tests/conftest.py` or `tests/test_manager_headless.py` constructing configs/lanes with removed fields).

- [ ] **Step 3: Local fake-mode smoke (no board, no server)**

```bash
python -m multi_stream_asr.main --headless --fake --lanes 4
echo "exit=$?"
```

Expected: `exit=0`, a new `runs/headless-*/metrics.json` containing keys `offline_rtf`, `offline_shared_timings` (count 0 in fake mode), `metrics_schema_version: 3`.

- [ ] **Step 4: Commit**

```bash
git add src/multi_stream_asr/headless/runner.py tests
git commit -m "feat: run headless pipeline against shared llama-multi-asr backend"
```

---

### Task 8: Board environment — rsync, venv, tests, fake smoke

**Files:**
- Create (board): `~/Workspace/projects/multi-stream-asr-share-qwen/` (rsync target + `.venv313`)

**Interfaces:**
- Consumes: green worktree (Task 7), board runtime dir (Task 2).
- Produces: board environment where Task 9's real run executes.

- [ ] **Step 1: Rsync the worktree to the NEW board directory (never the main one)**

```bash
rsync -a --delete \
  --exclude .git --exclude .venv --exclude .venv313 --exclude runs \
  --exclude .pytest_cache --exclude models --exclude '*.egg-info' \
  /home/liangff/Workspace/projects/multi-stream-asr-share-qwen/ \
  bianbu@10.0.91.51:~/Workspace/projects/multi-stream-asr-share-qwen/
```

- [ ] **Step 2: Create `.venv313` and install dependencies (per README + user instruction)**

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr-share-qwen && python3.13 -m venv .venv313 && . .venv313/bin/activate && export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ && python -m pip install -U pip && python -m pip install -e ".[dev]" && python -m pip install sounddevice && pip install sherpa-onnx --index-url https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple'
```

Expected: all installs succeed; sherpa-onnx resolves to a prebuilt riscv64 wheel (never compiles). If the spacemit index needs auth or the wheel name differs, fall back to the README's exact form: `python -m pip install --no-cache-dir --only-binary=:all: --no-deps sherpa-onnx --index-url https://git.spacemit.com/api/v4/projects/81/packages/pypi/simple`.

- [ ] **Step 3: Run the unit suite on the board**

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr-share-qwen && . .venv313/bin/activate && python -m pytest -q'
```

Expected: ALL PASS (including zipformer tests that may be skipped locally).

- [ ] **Step 4: Fake-mode smoke on the board**

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr-share-qwen && . .venv313/bin/activate && python -m multi_stream_asr.main --headless --fake --lanes 4; echo exit=$?'
```

Expected: `exit=0` and no `llama-multi-asr` process spawned (`ssh bianbu@10.0.91.51 'pgrep -f llama-multi-asr || echo none'` → `none`).

- [ ] **Step 5: Commit any fixes made during board bring-up**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr-share-qwen
git status --short   # if fixes were needed, commit them:
git add -A && git commit -m "fix: board environment adjustments for shared backend"
```

(Skip the commit if the tree is clean.)

---

### Task 9: Real 4-lane headless run on the board

**Files:**
- Create (board): `~/Workspace/projects/multi-stream-asr-share-qwen/runs/headless-*/` (metrics + results)

**Interfaces:**
- Consumes: board env (Task 8), runtime (Task 2), wavs at `~/Music/wav16k`.
- Produces: `metrics.json` + `results/lane*.txt` + memory numbers consumed by the Task 10 report.

- [ ] **Step 1: Capture pre-run memory baseline**

```bash
ssh bianbu@10.0.91.51 'free -m | head -2'
```

Record `used`.

- [ ] **Step 2: Run the real 4-lane headless test**

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr-share-qwen && . .venv313/bin/activate && python -m multi_stream_asr.main --headless --lanes 4 --input-dir ~/Music/wav16k --verbose; echo exit=$?'
```

Expected: `exit=0`. Warmup phase takes ~25–60 s before lanes start. While it runs (or in a second run), capture live measurements:

```bash
ssh bianbu@10.0.91.51 'pid=$(pgrep -f "bin/llama-multi-asr"); echo "llama-multi-asr pid=$pid"; grep VmRSS /proc/$pid/status; free -m | head -2'
ssh bianbu@10.0.91.51 'pid=$(pgrep -f "bin/llama-multi-asr"); for t in /proc/$pid/task/*; do awk "/Cpus_allowed_list/{print \$2}" $t/status; done | sort | uniq -c'
```

Record: server RSS, total used memory during run, and the thread-affinity distribution (expect clusters on `8-9` for encoder threads and `10-15` for decoder threads — this is the config.json silent-failure spot check from the spec).

- [ ] **Step 3: Verify run outputs**

```bash
ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr-share-qwen && latest=$(ls -td runs/headless-* | head -1) && echo "$latest" && python3 - <<PY
import json, pathlib, os
run = pathlib.Path(sorted([p for p in pathlib.Path("runs").glob("headless-*")], key=os.path.getmtime)[-1])
m = json.loads((run / "metrics.json").read_text())
print("schema:", m["metrics_schema_version"])
print("offline_rtf:", m["offline_rtf"])
print("offline_shared_timings:", json.dumps(m["offline_shared_timings"], indent=1))
print("online_frame_latency:", m["online_frame_latency"])
print("event_loop_lag:", m["event_loop_lag"])
for lane in range(4):
    text = (run / "results" / f"lane{lane}.txt").read_text().strip()
    print(f"lane{lane}: {len(text)} chars, sample: {text[:40]!r}")
PY'
```

Acceptance (all must hold):
- `metrics_schema_version` == 3
- `offline_shared_timings.count` > 0 and equals `offline_rtf.count`
- every `lane{0..3}.txt` non-empty (real Chinese transcriptions)
- `offline_rtf.average` recorded (target: ≤ the dedicated baseline 0.50; if it exceeds, still record — the report analyzes why)
- no leftover `llama-multi-asr` process after exit: `ssh bianbu@10.0.91.51 'pgrep -f llama-multi-asr || echo clean'` → `clean`

- [ ] **Step 4: Pull the run artifacts back to the worktree**

```bash
latest=$(ssh bianbu@10.0.91.51 'cd ~/Workspace/projects/multi-stream-asr-share-qwen && ls -td runs/headless-* | head -1')
rsync -a bianbu@10.0.91.51:~/Workspace/projects/multi-stream-asr-share-qwen/"$latest"/ \
  /home/liangff/Workspace/projects/multi-stream-asr-share-qwen/"$latest"/
```

- [ ] **Step 5: Commit the run artifacts**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr-share-qwen
git add runs/
git commit -m "test: record 4-lane real headless run on shared-qwen backend"
```

---

### Task 10: Function and performance report

**Files:**
- Create: `docs/superpowers/reports/2026-07-10-single-process-share-qwen-report.md`

**Interfaces:**
- Consumes: Task 9's metrics.json + memory/affinity numbers; dedicated-scheme baselines from the MAIN checkout: `/home/liangff/Workspace/projects/multi-stream-asr/runs/headless-1783590515514/metrics.json` (offline_rtf avg 0.521) and `/home/liangff/Workspace/projects/multi-stream-asr/runs/headless-1783600625335/metrics.json` (offline_rtf avg 0.499, max 0.855).
- Produces: the final deliverable of this project.

- [ ] **Step 1: Write the report**

Structure (fill every number from the actual runs — no estimates):

```markdown
# 单进程共享 Qwen：4 路无头测试报告（2026-07-10）

## 功能验证
- 分支/提交、板上目录、启动方式（1× llama-multi-asr 替代 4× llama-server）
- 单元测试：本机 N passed / 板上 N passed
- 4 路真实无头跑通：exit=0，lane0-3 识别文本非空（各贴 1 句样例）
- 分核验证：/proc 线程亲和分布（encoder 8-9 / decoder 10-15），config.json 平铺字段生效
- 进程回收：运行结束后无残留 llama-multi-asr

## 性能对比（4 路，input ~/Music/wav16k）
| 指标 | 独占基线 A (1783590515514) | 独占基线 B (1783600625335) | 共享方案（本次） |
| offline_rtf avg / max / count | 0.521 / ~1.06 / … | 0.499 / 0.855 / 149 | … |
| online_frame_latency avg / max ms | … | 45.6 / … | … |
| event_loop_lag avg / max ms | … | 1.22 / 84.3 | … |
| 内存（模型权重侧） | ≈4×660MB | 同左 | 服务 RSS 实测 … |

## 共享方案专有指标（offline_shared_timings）
queue/encode/prefill/decode/total 的 avg/max —— 排队 vs 计算分解

## 结论与已知差异
- q40（本方案）vs q41（multi-asr-test 基线 rtf_infer≈0.16）量化版本差异注记
- 4 路 n_parallel=4 下 queue_ms 表现；是否达到 offline_rtf ≤ 0.50 / README 目标 ≤ 0.4
- 后续建议（如 6/8 路扩展测试）
```

Baseline values not listed above are read from the two baseline metrics.json files with:

```bash
python3 - <<'PY'
import json
for run in ("headless-1783590515514", "headless-1783600625335"):
    m = json.load(open(f"/home/liangff/Workspace/projects/multi-stream-asr/runs/{run}/metrics.json"))
    print(run, {k: m[k] for k in ("offline_rtf", "online_frame_latency", "event_loop_lag")})
PY
```

- [ ] **Step 2: Commit the report**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr-share-qwen
git add docs/superpowers/reports/2026-07-10-single-process-share-qwen-report.md
git commit -m "docs: add 4-lane shared-qwen function and performance report"
```

- [ ] **Step 3: Final check — branch is self-contained, main untouched**

```bash
cd /home/liangff/Workspace/projects/multi-stream-asr-share-qwen && git log --oneline 8522bf4..HEAD
cd /home/liangff/Workspace/projects/multi-stream-asr && git status --short && git log --oneline -1
ssh bianbu@10.0.91.51 'ls -dt ~/Workspace/projects/multi-stream-asr/runs/headless-* | head -1'
```

Expected: dev branch shows the full commit series; main checkout clean; the board's ORIGINAL `multi-stream-asr` dir has no new runs (untouched).
