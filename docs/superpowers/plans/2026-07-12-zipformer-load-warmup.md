# Zipformer Load Warmup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warm every real Zipformer recognizer with one fixed packaged WAV during model loading, expose a global Web warmup phase, and verify that three real four-lane runs no longer contain the existing 610–630 ms cold-start spike.

**Architecture:** Add a two-phase lane load barrier: every lane first reports `lane_model_loaded`, then the main process broadcasts one `warmup` command and waits for the existing `lane_loaded`. Extend the persistent online worker with short actions so model construction, warmup, and formal consumption all run serially on the same OS thread. Keep Pipeline state as `loading`; expose `load_phase` through `/health` and poll it from the browser while the load request is pending.

**Tech Stack:** Python 3.13, asyncio, multiprocessing spawn, concurrent futures, sherpa-onnx, SpaceMiT EP, NumPy, FastAPI, vanilla JavaScript, pytest, setuptools package data.

## Global Constraints

- Implement Zipformer-only manual warmup; do not add a Qwen HTTP warmup request.
- Preserve the existing shared Qwen `--warmup`, `--n-parallel 4`, `--n-ctx 8192`, encoder affinity, decoder cores, and model selection behavior.
- Copy `/home/liangff/Music/wavs/junzhao_nihao.wav` byte-for-byte to `src/multi_stream_asr/assets/zipformer_warmup.wav`.
- The packaged WAV SHA-256 must be `d29d16456d9c7662135fbd696605391745ed56ce47eee3385a37c4b120a9691f`.
- The packaged WAV must remain 16 kHz, mono, 16-bit signed PCM with 22080 samples.
- Warmup must be independent of `input_dir`, `default_files`, `local_wav`, microphone selection, and every user-selected input path.
- All lanes warm concurrently only after all lane recognizers have loaded.
- Factory, warmup, and formal Zipformer consumption for one lane must run on the same persistent OS thread.
- Warmup must not create a run directory, publish ASR events, advance sentence IDs, write artifacts, or contribute to formal metrics.
- Keep `OnlineFrameQueue(maxsize=1)`, endpoint defaults, result formats, metrics schema, and process topology unchanged.
- Frontend displays one global `Warming up Zipformer...` phase; it must not display per-lane progress.
- Any warmup failure fails the complete model load and runs existing bounded cleanup.
- Fake mode follows the same IPC barrier but does not open the WAV or call a real recognizer.
- Board work uses project-local Python 3.13 `.venv313` and the package-index environment from `~/.codex/docs/k3/remote-device-python.md` if installation is needed.
- Do not commit model files, `.venv313`, `runs/`, board-local files, or the user-owned untracked `TODO`.

---

## File Structure

Create:

- `src/multi_stream_asr/assets/zipformer_warmup.wav`: immutable packaged warmup input.

Modify:

- `pyproject.toml`: include `assets/*.wav` in `multi_stream_asr` package data.
- `src/multi_stream_asr/config.py`: expose `ZIPFORMER_WARMUP_AUDIO_PATH`.
- `src/multi_stream_asr/pipeline/online_worker.py`: support repeatable short actions before the one terminal consumer.
- `src/multi_stream_asr/pipeline/lane_runtime.py`: implement fixed-input Zipformer warmup and two-phase lane load protocol.
- `src/multi_stream_asr/pipeline/process_manager.py`: implement the all-loaded/all-warmed barriers and `load_phase`.
- `src/multi_stream_asr/web/server.py`: expose `load_phase` in health/status payloads.
- `src/multi_stream_asr/web/static/index.html`: poll and display the real loading subphase.
- `README.md`: document automatic Zipformer warmup.
- `docs/thread-model.md`: document the load/warmup barrier and same-thread requirement.
- `tests/test_config.py`, `tests/test_online_worker.py`, `tests/test_lane_runtime.py`, `tests/test_process_manager.py`, `tests/test_web_server.py`: focused and regression coverage.

---

### Task 1: Package the Fixed Warmup WAV

**Files:**
- Create: `src/multi_stream_asr/assets/zipformer_warmup.wav`
- Modify: `src/multi_stream_asr/config.py`
- Modify: `pyproject.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ZIPFORMER_WARMUP_AUDIO_PATH: Path`.
- Produces: a wheel-installed `multi_stream_asr/assets/zipformer_warmup.wav` with the specified digest and WAV metadata.
- Consumes: no runtime interfaces from later tasks.

- [ ] **Step 1: Write the failing resource test**

Add imports and this test to `tests/test_config.py`:

```python
import hashlib
import tomllib
import wave

from multi_stream_asr.config import ZIPFORMER_WARMUP_AUDIO_PATH


def test_zipformer_warmup_audio_is_fixed_valid_package_data() -> None:
    audio = ZIPFORMER_WARMUP_AUDIO_PATH

    assert audio == Path(__file__).resolve().parents[1] / (
        "src/multi_stream_asr/assets/zipformer_warmup.wav"
    )
    assert hashlib.sha256(audio.read_bytes()).hexdigest() == (
        "d29d16456d9c7662135fbd696605391745ed56ce47eee3385a37c4b120a9691f"
    )
    with wave.open(str(audio), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 22080

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    assert pyproject["tool"]["setuptools"]["package-data"]["multi_stream_asr"] == [
        "assets/*.wav"
    ]
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
.venv313/bin/python -m pytest tests/test_config.py::test_zipformer_warmup_audio_is_fixed_valid_package_data -v
```

Expected: collection fails because `ZIPFORMER_WARMUP_AUDIO_PATH` does not exist.

- [ ] **Step 3: Copy the binary resource and add the path constant**

Create the asset directory and copy the file byte-for-byte:

```bash
mkdir -p src/multi_stream_asr/assets
cp /home/liangff/Music/wavs/junzhao_nihao.wav \
  src/multi_stream_asr/assets/zipformer_warmup.wav
sha256sum src/multi_stream_asr/assets/zipformer_warmup.wav
```

Expected digest:

```text
d29d16456d9c7662135fbd696605391745ed56ce47eee3385a37c4b120a9691f
```

Add beside the other project paths in `src/multi_stream_asr/config.py`:

```python
ZIPFORMER_WARMUP_AUDIO_PATH = (
    Path(__file__).resolve().parent / "assets" / "zipformer_warmup.wav"
)
```

Update package data in `pyproject.toml`:

```toml
[tool.setuptools.package-data]
"multi_stream_asr" = ["assets/*.wav"]
"multi_stream_asr.web" = ["static/*.html"]
```

- [ ] **Step 4: Run the resource test and verify GREEN**

Run:

```bash
.venv313/bin/python -m pytest tests/test_config.py -q
```

Expected: all configuration tests pass.

- [ ] **Step 5: Verify the built wheel contains the WAV**

Run:

```bash
rm -rf /tmp/multi-stream-asr-wheel
mkdir -p /tmp/multi-stream-asr-wheel
.venv313/bin/python -m pip wheel . --no-deps -w /tmp/multi-stream-asr-wheel
unzip -l /tmp/multi-stream-asr-wheel/multi_stream_asr-0.1.0-py3-none-any.whl \
  | rg 'multi_stream_asr/assets/zipformer_warmup.wav'
```

Expected: exactly one matching wheel entry.

- [ ] **Step 6: Commit the resource boundary**

```bash
git add pyproject.toml src/multi_stream_asr/config.py \
  src/multi_stream_asr/assets/zipformer_warmup.wav tests/test_config.py
git commit -m "feat: package zipformer warmup audio"
```

---

### Task 2: Add Short Actions to the Persistent Online Worker

**Files:**
- Modify: `src/multi_stream_asr/pipeline/online_worker.py`
- Test: `tests/test_online_worker.py`

**Interfaces:**
- Produces: `OnlineConsumerWorker.run_action(target: Callable[[], None]) -> concurrent.futures.Future[None]`.
- Preserves: `run_consumer(target)` as the one terminal consumer command.
- Guarantees: factory, every action, and terminal consumer execute serially on the worker thread.

- [ ] **Step 1: Write same-thread and action-failure tests**

Add to `tests/test_online_worker.py`:

```python
import pytest


def test_online_worker_runs_actions_before_consumer_on_owner_thread() -> None:
    calls: list[tuple[str, int]] = []

    def factory() -> object:
        calls.append(("factory", threading.get_ident()))
        return object()

    async def exercise() -> None:
        worker = OnlineConsumerWorker(name="test-online-actions", factory=factory)
        await worker.wait_until_loaded()
        await asyncio.wrap_future(
            worker.run_action(
                lambda: calls.append(("warmup", threading.get_ident()))
            )
        )
        await asyncio.wrap_future(
            worker.run_consumer(
                lambda: calls.append(("consumer", threading.get_ident()))
            )
        )
        await worker.close()

    asyncio.run(exercise())

    assert [name for name, _ in calls] == ["factory", "warmup", "consumer"]
    assert len({thread_id for _, thread_id in calls}) == 1


def test_online_worker_propagates_action_failure_and_can_close() -> None:
    async def exercise() -> None:
        worker = OnlineConsumerWorker(name="test-online-action-error", factory=object)
        await worker.wait_until_loaded()

        def fail() -> None:
            raise RuntimeError("warmup failed")

        with pytest.raises(RuntimeError, match="warmup failed"):
            await asyncio.wrap_future(worker.run_action(fail))
        await worker.close()
        assert not worker.is_alive()

    asyncio.run(exercise())
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv313/bin/python -m pytest tests/test_online_worker.py -v
```

Expected: failures report that `OnlineConsumerWorker` has no `run_action`.

- [ ] **Step 3: Implement reusable actions and one terminal consumer**

Add a focused command type and replace the one-command `_run` body in `online_worker.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class _WorkerCommand:
    target: Callable[[], None]
    future: concurrent.futures.Future[None]
    terminal: bool
```

Use this queue and submission API inside `OnlineConsumerWorker`:

```python
self._commands: queue.Queue[_WorkerCommand | None] = queue.Queue(maxsize=1)
self._consumer_submitted = False

def run_action(
    self,
    target: Callable[[], None],
) -> concurrent.futures.Future[None]:
    return self._submit(target, terminal=False)

def run_consumer(
    self,
    target: Callable[[], None],
) -> concurrent.futures.Future[None]:
    return self._submit(target, terminal=True)

def _submit(
    self,
    target: Callable[[], None],
    *,
    terminal: bool,
) -> concurrent.futures.Future[None]:
    with self._command_lock:
        if self._closed:
            raise RuntimeError("online consumer worker is closed")
        if self._consumer_submitted:
            raise RuntimeError("online consumer worker already has a consumer")
        if terminal:
            self._consumer_submitted = True
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._commands.put(
            _WorkerCommand(target=target, future=future, terminal=terminal)
        )
        return future
```

After the factory succeeds, process actions until close or the terminal consumer:

```python
self._loaded.set_result(recognizer)
while True:
    command = self._commands.get()
    if command is None:
        return
    if not command.future.set_running_or_notify_cancel():
        if command.terminal:
            return
        continue
    try:
        command.target()
    except BaseException as exc:
        command.future.set_exception(exc)
    else:
        command.future.set_result(None)
    if command.terminal:
        return
```

Keep `close()` behavior: enqueue `None` only when no terminal consumer has been submitted, then await `_completed`.

```python
async def close(self) -> None:
    with self._command_lock:
        if not self._closed:
            self._closed = True
            if not self._consumer_submitted:
                self._commands.put(None)
    await asyncio.wrap_future(self._completed)
```

- [ ] **Step 4: Run worker and pipeline regressions**

Run:

```bash
.venv313/bin/python -m pytest tests/test_online_worker.py tests/test_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the worker interface**

```bash
git add src/multi_stream_asr/pipeline/online_worker.py tests/test_online_worker.py
git commit -m "feat: run pre-consumer actions on online worker"
```

---

### Task 3: Implement Lane-Local Zipformer Warmup

**Files:**
- Modify: `src/multi_stream_asr/pipeline/lane_runtime.py`
- Test: `tests/test_lane_runtime.py`

**Interfaces:**
- Consumes: `OnlineConsumerWorker.run_action(...)` and fixed `warmup.input_path` IPC string.
- Produces: lifecycle `lane_model_loaded` before waiting for warmup.
- Preserves: lifecycle `lane_loaded`, now emitted only after warmup and reset.
- Produces: `warmup_zipformer(recognizer, *, stream_id, input_path, frame_duration_ms) -> None` async helper.

- [ ] **Step 1: Write a focused warmup helper test**

Add to `tests/test_lane_runtime.py`:

```python
@pytest.mark.asyncio
async def test_warmup_zipformer_feeds_fixed_wav_and_resets(tmp_path: Path) -> None:
    from multi_stream_asr.pipeline.lane_runtime import warmup_zipformer

    wav_path = tmp_path / "warmup.wav"
    samples = np.arange(22080, dtype="<i2")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    class RecordingRecognizer:
        def __init__(self) -> None:
            self.frames: list[AudioFrame] = []
            self.reset_calls = 0

        async def accept(self, frame: AudioFrame) -> None:
            self.frames.append(frame)

        async def reset(self) -> None:
            self.reset_calls += 1

    recognizer = RecordingRecognizer()
    await warmup_zipformer(
        recognizer,
        stream_id=2,
        input_path=wav_path,
        frame_duration_ms=100,
    )

    assert len(recognizer.frames) == 14
    assert all(not frame.is_finish for frame in recognizer.frames)
    assert [frame.frame_index for frame in recognizer.frames] == list(range(14))
    assert recognizer.reset_calls == 1
```

Add imports `threading`, `wave`, and `ZIPFORMER_WARMUP_AUDIO_PATH`, then add this integration test:

```python
def test_real_lane_warms_on_owner_thread_before_lane_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import lane_runtime

    owner_threads: list[int] = []
    accepted_frames: list[AudioFrame] = []
    reset_threads: list[int] = []
    events = MemoryQueue()
    commands = MemoryQueue()
    config = AppConfig(lanes=2, output_dir=tmp_path)
    lane = config.lane_configs[0]

    class FakeOnlineRecognizer:
        def __init__(self, model_dir: Path, endpoint: object) -> None:
            owner_threads.append(threading.get_ident())

        async def accept(self, frame: AudioFrame) -> None:
            owner_threads.append(threading.get_ident())
            accepted_frames.append(frame)

        async def reset(self) -> None:
            owner_threads.append(threading.get_ident())
            reset_threads.append(threading.get_ident())

    class FakeOfflineRecognizer:
        def __init__(self, *, shared: object) -> None:
            pass

        async def close(self) -> None:
            pass

    monkeypatch.setattr(lane_runtime, "ZipformerRecognizer", FakeOnlineRecognizer)
    monkeypatch.setattr(lane_runtime, "Qwen3AsrClient", FakeOfflineRecognizer)
    commands.put(
        {
            "type": "warmup",
            "input_path": str(ZIPFORMER_WARMUP_AUDIO_PATH),
        }
    )
    commands.put({"type": "stop"})

    runtime = LaneRuntime(
        process_config=LaneProcessConfig(app_config=config, lane=lane),
        event_queue=events,
        command_queue=commands,
    )

    assert asyncio.run(runtime.run()) == 0
    emitted = messages(events)
    lifecycle = [
        message.payload["state"]
        for message in emitted
        if message.type == "lifecycle"
    ]
    assert lifecycle == [
        "lane_loading",
        "lane_model_loaded",
        "lane_loaded",
        "lane_stopped",
    ]
    assert len(accepted_frames) == 14
    assert len(set(owner_threads)) == 1
    assert len(reset_threads) == 1
    assert not [
        message for message in emitted if message.type == "asr_event"
    ]
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "model_outputs").exists()
```

- [ ] **Step 2: Run the new lane tests and verify RED**

Run:

```bash
.venv313/bin/python -m pytest tests/test_lane_runtime.py \
  -k 'warmup_zipformer or model_loaded' -v
```

Expected: import or lifecycle assertions fail because warmup is not implemented.

- [ ] **Step 3: Implement the reusable warmup helper**

Add to `lane_runtime.py` after fake recognizers:

```python
async def warmup_zipformer(
    recognizer: Any,
    *,
    stream_id: int,
    input_path: Path,
    frame_duration_ms: int,
) -> None:
    source = LocalWavSource(
        stream_id=stream_id,
        path=input_path,
        frame_duration_ms=frame_duration_ms,
        realtime=False,
    )
    await source.prepare()
    try:
        async for frame in source.frames():
            if frame.is_finish:
                break
            await recognizer.accept(frame)
        await recognizer.reset()
    finally:
        await source.stop()
```

- [ ] **Step 4: Add the two-phase lane load protocol**

Change `_load()` to return an early stop command or `None`:

```python
async def _load(self) -> dict[str, Any] | None:
    self.send("lifecycle", {"state": "lane_loading", "message": ""})
    if self.process_config.fake:
        self._online_recognizer = _FakeOnlineRecognizer(self.lane)
        self._offline_recognizer = _FakeOfflineRecognizer(self.lane)
    else:
        self._online_worker = OnlineConsumerWorker(
            name=f"asr-lane-{self.lane.stream_id}-zipformer-consumer",
            factory=lambda: ZipformerRecognizer(
                self.app_config.models.online_model_dir,
                self.app_config.endpoint,
            ),
        )
        self._online_recognizer = await self._online_worker.wait_until_loaded()
        self._offline_recognizer = Qwen3AsrClient(
            shared=self.app_config.shared_qwen,
        )

    self.send("lifecycle", {"state": "lane_model_loaded", "message": ""})
    command = await self._wait_for_command({"warmup", "stop", "shutdown"})
    if command["type"] != "warmup":
        return command

    if not self.process_config.fake:
        assert self._online_worker is not None
        assert self._online_recognizer is not None
        input_path = Path(str(command["input_path"]))
        future = self._online_worker.run_action(
            lambda: asyncio.run(
                warmup_zipformer(
                    self._online_recognizer,
                    stream_id=self.lane.stream_id,
                    input_path=input_path,
                    frame_duration_ms=self.app_config.frame_duration_ms,
                )
            )
        )
        await asyncio.wrap_future(future)

    self.send("lifecycle", {"state": "lane_loaded", "message": ""})
    return None
```

Update `run()` so it consumes an early stop exactly once:

```python
command = await self._load()
self._fatal_phase = "run"
if command is None:
    command = await self._wait_for_command({"start", "stop", "shutdown"})
```

Do not instantiate `LanePipeline`, writers, cache, trackers, or a source selected from user configuration during `_load()`.

- [ ] **Step 5: Update existing lane tests to send warmup explicitly**

For real/fake lane tests that previously queued `start` or `stop` immediately after model creation, queue this first:

```python
commands.put(
    {
        "type": "warmup",
        "input_path": str(ZIPFORMER_WARMUP_AUDIO_PATH),
    }
)
```

Update lifecycle expectations to include `lane_model_loaded`. Keep tests that intentionally stop during load without a warmup command; those must not emit `lane_loaded`.

- [ ] **Step 6: Run all lane and worker tests**

Run:

```bash
.venv313/bin/python -m pytest tests/test_lane_runtime.py tests/test_online_worker.py -q
```

Expected: all tests pass, including warmup failure propagation and same-thread ownership.

- [ ] **Step 7: Commit lane warmup**

```bash
git add src/multi_stream_asr/pipeline/lane_runtime.py tests/test_lane_runtime.py
git commit -m "feat: warm zipformer inside lane owner thread"
```

---

### Task 4: Add the Main-Process Warmup Barrier

**Files:**
- Modify: `src/multi_stream_asr/pipeline/process_manager.py`
- Test: `tests/test_process_manager.py`

**Interfaces:**
- Consumes: `ZIPFORMER_WARMUP_AUDIO_PATH`, `lane_model_loaded`, and final `lane_loaded`.
- Produces: `ProcessPipelineManager.load_phase -> str | None`.
- Produces: one JSON-safe warmup command per active lane after the all-model-loaded barrier.

- [ ] **Step 1: Make the scripted controller model both barriers**

Change the default `_ScriptedController.start()` lifecycle to:

```python
[("lifecycle", {"state": "lane_model_loaded", "message": ""})]
```

Add warmup handling before start handling in `_ScriptedController.send()`:

```python
if command["type"] == "warmup":
    self.emit("lifecycle", {"state": "lane_loaded", "message": ""})
    return
```

This preserves existing manager tests while exercising the new command barrier.

- [ ] **Step 2: Write the barrier and phase test**

Add imports for `ZIPFORMER_WARMUP_AUDIO_PATH`, then add:

```python
@pytest.mark.asyncio
async def test_manager_waits_for_all_models_then_all_warmups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class ManualWarmupController(_ScriptedController):
        def start(self) -> None:
            self._alive = True
            type(self).actions.append(f"start:{self.lane.stream_id}")

    async def wait_until(predicate: Any) -> None:
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("condition was not reached")

    monkeypatch.setattr(process_manager, "LaneProcessController", ManualWarmupController)
    config = AppConfig(
        lanes=2,
        lane_inputs=(
            LaneInputConfig(kind="microphone", microphone_device=3),
            LaneInputConfig(
                kind="local_wav",
                local_wav_path=Path("/must-not-be-used-for-warmup.wav"),
            ),
        ),
    )
    manager = ProcessPipelineManager(
        config=config,
        event_bus=FrontendEventBus(),
        fake=True,
        lifecycle_timeout_seconds=1.0,
    )
    load_task = asyncio.create_task(manager.load_models())
    await wait_until(lambda: len(ManualWarmupController.instances) == 2)

    controllers = sorted(
        ManualWarmupController.instances,
        key=lambda controller: controller.lane.stream_id,
    )
    controllers[0].emit("lifecycle", {"state": "lane_model_loaded", "message": ""})
    await asyncio.sleep(0)
    assert all(not controller.commands for controller in controllers)

    controllers[1].emit("lifecycle", {"state": "lane_model_loaded", "message": ""})
    await wait_until(
        lambda: all(
            any(command["type"] == "warmup" for command in controller.commands)
            for controller in controllers
        )
    )
    assert manager.state is PipelineState.LOADING
    assert manager.load_phase == "warming_zipformer"
    assert all(
        controller.commands[-1] == {
            "type": "warmup",
            "input_path": str(ZIPFORMER_WARMUP_AUDIO_PATH),
        }
        for controller in controllers
    )

    controllers[0].emit("lifecycle", {"state": "lane_loaded", "message": ""})
    await asyncio.sleep(0)
    assert not load_task.done()
    controllers[1].emit("lifecycle", {"state": "lane_loaded", "message": ""})
    await load_task
    assert manager.state is PipelineState.LOADED
    assert manager.load_phase is None
```

Add this warmup-fatal test:

```python
@pytest.mark.asyncio
async def test_warmup_fatal_cleans_all_lanes_and_shared_qwen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_stream_asr.pipeline import process_manager

    class WarmupFailingController(_ScriptedController):
        def send(self, command: dict[str, Any]) -> None:
            self.commands.append(command)
            if command["type"] == "warmup":
                self.emit("fatal", {"message": "zipformer warmup failed"})
                self._alive = False
                self._exitcode = 1
            elif command["type"] in {"stop", "shutdown"}:
                self.emit("lifecycle", {"state": "lane_stopped", "message": ""})
                self._alive = False
                self._exitcode = 0

    monkeypatch.setattr(
        process_manager, "LaneProcessController", WarmupFailingController
    )
    monkeypatch.setattr(
        process_manager,
        "SharedQwenServerManager",
        _NoopSharedQwenServerManager,
    )
    manager = ProcessPipelineManager(
        config=AppConfig(lanes=2),
        event_bus=FrontendEventBus(),
        fake=False,
        lifecycle_timeout_seconds=1.0,
    )

    with pytest.raises(RuntimeError, match="zipformer warmup failed"):
        await manager.load_models()

    assert manager.state is PipelineState.FATAL
    assert manager.load_phase is None
    assert _NoopSharedQwenServerManager.instances
    assert not _NoopSharedQwenServerManager.instances[0].is_alive()
```

- [ ] **Step 3: Run the focused manager tests and verify RED**

Run:

```bash
.venv313/bin/python -m pytest tests/test_process_manager.py \
  -k 'all_models_then_all_warmups or warmup_fatal' -v
```

Expected: failures because `load_phase`, `lane_model_loaded`, and warmup broadcast are absent.

- [ ] **Step 4: Implement phase state and model-loaded tracking**

Add imports:

```python
from multi_stream_asr.audio.local_file import select_input_files, validate_wav_16k_mono_pcm
from multi_stream_asr.config import AppConfig, LaneConfig, ZIPFORMER_WARMUP_AUDIO_PATH
```

Add `lane_model_loaded` to `_LIFECYCLE_STATES`. Add fields and property:

```python
self._model_loaded: set[int] = set()
self._load_phase: str | None = None

@property
def load_phase(self) -> str | None:
    return self._load_phase
```

Clear `_model_loaded` in `_reset_for_load()` and final unload cleanup. Handle lifecycle:

```python
if state == "lane_model_loaded":
    self._model_loaded.add(stream_id)
elif state == "lane_loaded":
    self._loaded.add(stream_id)
```

- [ ] **Step 5: Implement the two waits and broadcast**

Validate the fixed asset before starting Qwen, then replace `load_models()` with:

```python
async def load_models(self) -> None:
    warmup_path = ZIPFORMER_WARMUP_AUDIO_PATH.resolve()
    validate_wav_16k_mono_pcm(warmup_path)

    async with self._lifecycle_lock:
        assert_transition(self._state, PipelineState.LOADING)
        self._state = PipelineState.LOADING
        self._load_phase = "loading_models"
        self._reset_for_load()
        try:
            if not self._fake:
                self._shared_manager = SharedQwenServerManager(self._config)
                await asyncio.to_thread(self._shared_manager.start)
                self._shared_stop_expected = False
                self._shared_watchdog_task = asyncio.create_task(
                    self._watch_shared_qwen(),
                    name="asr-shared-qwen-watchdog",
                )
            self._controllers = {
                lane.stream_id: LaneProcessController(
                    app_config=self._config,
                    lane=lane,
                    fake=self._fake,
                )
                for lane in self._config.lane_configs
            }
            for stream_id, controller in self._controllers.items():
                controller.start()
                self._reader_tasks[stream_id] = asyncio.create_task(
                    self._read_lane_events(controller),
                    name=f"asr-lane-reader-{stream_id}",
                )
            await self._wait_for(
                lambda: len(self._model_loaded) == self._config.lanes,
                timeout=self._lifecycle_timeout_seconds,
            )
            self._load_phase = "warming_zipformer"
            self._send_to_active(
                {"type": "warmup", "input_path": str(warmup_path)}
            )
            await self._wait_for(
                lambda: len(self._loaded) == self._config.lanes,
                timeout=self._lifecycle_timeout_seconds,
            )
            assert_transition(self._state, PipelineState.LOADED)
            self._state = PipelineState.LOADED
        except Exception as exc:
            if self._failure is None:
                await self._record_fatal(0, f"lane load failed: {exc}")
            self._send_to_active({"type": "stop", "reason": "load_failed"})
            await self._join_or_escalate()
            await self._finish_readers_and_close()
            await self._stop_shared_qwen()
            raise self._failure from exc
        finally:
            self._load_phase = None
```

Do not merge the two waits: the frontend phase must change only after all model-loaded events have arrived.

- [ ] **Step 6: Run process-manager regressions**

Run:

```bash
.venv313/bin/python -m pytest tests/test_process_manager.py tests/test_manager_headless.py -q
```

Expected: all tests pass; existing controller scripts have been updated to the new barrier.

- [ ] **Step 7: Commit main-process orchestration**

```bash
git add src/multi_stream_asr/pipeline/process_manager.py \
  tests/test_process_manager.py tests/test_manager_headless.py
git commit -m "feat: coordinate parallel zipformer warmup"
```

---

### Task 5: Expose and Render the Warmup Phase

**Files:**
- Modify: `src/multi_stream_asr/web/server.py`
- Modify: `src/multi_stream_asr/web/static/index.html`
- Test: `tests/test_web_server.py`

**Interfaces:**
- Consumes: manager `load_phase` property.
- Produces: every status payload contains `load_phase: str | None`.
- Produces: browser polling every 250 ms only while model-load POST is pending.

- [ ] **Step 1: Write API and static frontend tests**

Give `RecordingManager` a default `load_phase = None`, then add:

```python
def test_health_exposes_manager_load_phase(tmp_path: Path) -> None:
    manager = RecordingManager(state=PipelineState.LOADING)
    manager.load_phase = "warming_zipformer"

    with TestClient(
        create_app(config=AppConfig(), manager=manager)
    ) as client:
        response = client.get("/health")

    assert response.json()["load_phase"] == "warming_zipformer"
```

Add static assertions:

```python
def test_index_html_polls_and_displays_zipformer_warmup(tmp_path: Path) -> None:
    with TestClient(
        create_app(config=AppConfig(), manager=RecordingManager())
    ) as client:
        html = client.get("/").text

    assert "async function pollLoadPhase()" in html
    assert "window.setInterval(pollLoadPhase, 250)" in html
    assert 'payload.load_phase === "warming_zipformer"' in html
    assert 'setOperation("Warming up Zipformer...")' in html
    assert "stopLoadPhasePolling();" in html
```

- [ ] **Step 2: Run Web tests and verify RED**

Run:

```bash
.venv313/bin/python -m pytest tests/test_web_server.py \
  -k 'load_phase or zipformer_warmup' -v
```

Expected: API field and JavaScript assertions fail.

- [ ] **Step 3: Expose `load_phase` from the server**

Add to `_status_payload()` in `web/server.py`:

```python
"load_phase": getattr(app.state.manager, "load_phase", None),
```

Using `getattr` preserves compatibility with focused fake managers.

- [ ] **Step 4: Add bounded browser polling**

Add state and helpers near existing health functions:

```javascript
let loadPhaseTimer = null;

async function pollLoadPhase() {
  const response = await fetch("/health");
  if (!response.ok) return;
  const payload = await response.json();
  if (payload.load_phase === "warming_zipformer") {
    setOperation("Warming up Zipformer...");
  } else if (payload.load_phase === "loading_models") {
    setOperation("Loading models...");
  }
}

function startLoadPhasePolling() {
  stopLoadPhasePolling();
  loadPhaseTimer = window.setInterval(pollLoadPhase, 250);
}

function stopLoadPhasePolling() {
  if (loadPhaseTimer !== null) {
    window.clearInterval(loadPhaseTimer);
    loadPhaseTimer = null;
  }
}
```

Wrap the load request with polling:

```javascript
appState.state = "loading";
setOperation("Loading models...");
renderStatus();
startLoadPhasePolling();
try {
  await postJson("/api/models/load", {
    lanes: appState.lanes,
    offline_model_id: appState.draft.offlineModelId,
    offline_model_dir: appState.draft.offlineModelDir,
    lane_inputs: appState.draft.laneInputs,
  });
} finally {
  stopLoadPhasePolling();
}
setOperation("Models loaded.");
```

Do not render lane counters or lifecycle messages in result panels.

- [ ] **Step 5: Run all Web tests**

Run:

```bash
.venv313/bin/python -m pytest tests/test_web_server.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Web progress reporting**

```bash
git add src/multi_stream_asr/web/server.py \
  src/multi_stream_asr/web/static/index.html tests/test_web_server.py
git commit -m "feat: show zipformer warmup load phase"
```

---

### Task 6: Document and Run Local Regression

**Files:**
- Modify: `README.md`
- Modify: `docs/thread-model.md`

**Interfaces:**
- Consumes: final lifecycle and same-thread behavior from Tasks 1–5.
- Produces: operator-facing documentation; no runtime API.

- [ ] **Step 1: Update operator documentation**

Add to the Web/headless model-loading description in `README.md`:

```markdown
After all lane Zipformer models load, the service warms every recognizer in parallel
with the packaged `zipformer_warmup.wav`. Warmup is independent of the selected lane
inputs and completes before the service enters `loaded` or starts audio producers.
The Web status displays `Warming up Zipformer...` during this barrier.
```

Update the load section of `docs/thread-model.md` with this exact sequence:

```markdown
每路 recognizer 创建完成后先发送 `lane_model_loaded`。主进程等所有 lane 完成加载，
再同时发送固定 WAV 的 `warmup` 命令。预热 action 与正式 consumer 在同一长期
Zipformer 线程串行执行；每路 reset 成功后发送 `lane_loaded`。四路全部预热完成前，
主进程保持 `loading`，不会启动任何正式音频 producer。
```

- [ ] **Step 2: Run focused tests**

```bash
.venv313/bin/python -m pytest \
  tests/test_config.py \
  tests/test_online_worker.py \
  tests/test_lane_runtime.py \
  tests/test_process_manager.py \
  tests/test_manager_headless.py \
  tests/test_web_server.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run full tests and fake headless**

```bash
.venv313/bin/python -m pytest -q
.venv313/bin/python -m multi_stream_asr.main --headless --fake --lanes 4
git diff --check
```

Expected: full suite and fake headless exit 0; no whitespace errors.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/thread-model.md
git commit -m "docs: describe zipformer load warmup"
```

---

### Task 7: Run Three Real Board Validations

**Files:**
- No source modifications required.
- Inspect: three generated `metrics.json`, lane result files, and lane model-output files.

**Interfaces:**
- Consumes: completed branch, packaged WAV, board-local models/runtime, `scripts.evaluate_shared_qwen_runs.evaluate_run`.
- Produces: evidence for the fixed acceptance thresholds; does not change formal metrics schema.

- [ ] **Step 1: Synchronize source without external model or runtime copies**

From the local project root:

```bash
rsync -az \
  --exclude '.git/' --exclude '.venv313/' --exclude '.venv/' \
  --exclude 'models/' --exclude 'runs/' --exclude '__pycache__/' \
  --exclude '*.pyc' --exclude 'TODO' \
  ./ bianbu@10.0.91.51:~/Workspace/projects/multi-stream-asr/
```

Expected: source and packaged warmup WAV update; board-local model directories remain untouched.

- [ ] **Step 2: Verify board environment and packaged audio**

```bash
ssh bianbu@10.0.91.51 '
  cd ~/Workspace/projects/multi-stream-asr &&
  .venv313/bin/python --version &&
  sha256sum src/multi_stream_asr/assets/zipformer_warmup.wav &&
  .venv313/bin/python -m pytest \
    tests/test_config.py tests/test_online_worker.py \
    tests/test_lane_runtime.py tests/test_process_manager.py \
    tests/test_web_server.py -q
'
```

Expected: Python 3.13.x, exact digest from Global Constraints, and focused tests pass.

- [ ] **Step 3: Run three real four-lane headless tests**

On the board, record only newly created process directories:

```bash
ssh bianbu@10.0.91.51 '
  set -eu
  cd ~/Workspace/projects/multi-stream-asr
  : > /tmp/zipformer-warmup-run-dirs.txt
  for index in 1 2 3; do
    before=$(find runs -maxdepth 1 -type d -name "process-*" -printf "%f\n" | sort)
    .venv313/bin/python -m multi_stream_asr.main \
      --headless --lanes 4 --input-dir ~/Music/wav16k
    after=$(find runs -maxdepth 1 -type d -name "process-*" -printf "%f\n" | sort)
    run=$(comm -13 <(printf "%s\n" "$before") <(printf "%s\n" "$after") | tail -n 1)
    test -n "$run"
    printf "%s/%s\n" "$PWD/runs" "$run" >> /tmp/zipformer-warmup-run-dirs.txt
  done
'
```

Expected: each command exits 0 and exactly three absolute run paths are recorded.

- [ ] **Step 4: Check functional and performance gates independently of offline RTF**

```bash
ssh bianbu@10.0.91.51 '
  cd ~/Workspace/projects/multi-stream-asr
  .venv313/bin/python - <<"PY"
from pathlib import Path

from scripts.evaluate_shared_qwen_runs import evaluate_run

paths = [
    Path(line)
    for line in Path("/tmp/zipformer-warmup-run-dirs.txt").read_text().splitlines()
    if line
]
assert len(paths) == 3
for path in paths:
    result = evaluate_run(path)
    assert not result["failures"], result["failures"]
    metrics = result["metrics"]
    assert all(lane["dropped_frames"] < 10 for lane in metrics["lanes"])
    assert metrics["online_frame_latency"]["max_ms"] < 300.0
    print(
        path.name,
        "drops=", [lane["dropped_frames"] for lane in metrics["lanes"]],
        "online_avg_ms=", metrics["online_frame_latency"]["average_ms"],
        "online_max_ms=", metrics["online_frame_latency"]["max_ms"],
        "offline_rtf=", metrics["offline_rtf"]["average"],
    )
PY
'
```

Expected for every run: functional checks pass, every lane has fewer than 10 drops, and global maximum online latency is below 300 ms. Report offline RTF but do not fail this task on a new RTF target.

- [ ] **Step 5: Verify process cleanup**

```bash
ssh bianbu@10.0.91.51 '
  pgrep -af "[m]ulti_stream_asr.main|[s]pawn_main|[l]lama-multi-asr.*8090" \
    && exit 1 || exit 0
'
```

Expected: no project or shared-Qwen process remains.

- [ ] **Step 6: Push commits and fast-forward the board repository**

After all local commits and board validation pass:

```bash
git push origin multi-process-single-qwen
```

Preserve board-local untracked reproduction scripts. Stage and stash exactly the rsynced project paths before the fast-forward:

```bash
ssh bianbu@10.0.91.51 '
  set -eu
  cd ~/Workspace/projects/multi-stream-asr
  git add \
    pyproject.toml README.md docs/thread-model.md \
    docs/superpowers/specs/2026-07-12-zipformer-load-warmup-design.md \
    docs/superpowers/plans/2026-07-12-zipformer-load-warmup.md \
    src/multi_stream_asr/assets/zipformer_warmup.wav \
    src/multi_stream_asr/config.py \
    src/multi_stream_asr/pipeline/online_worker.py \
    src/multi_stream_asr/pipeline/lane_runtime.py \
    src/multi_stream_asr/pipeline/process_manager.py \
    src/multi_stream_asr/web/server.py \
    src/multi_stream_asr/web/static/index.html \
    tests/test_config.py tests/test_online_worker.py \
    tests/test_lane_runtime.py tests/test_process_manager.py \
    tests/test_manager_headless.py tests/test_web_server.py
  git stash push -m zipformer-warmup-rsync
  git pull --ff-only origin multi-process-single-qwen
  git diff --quiet stash@{0} HEAD -- \
    pyproject.toml README.md docs/thread-model.md \
    docs/superpowers/specs/2026-07-12-zipformer-load-warmup-design.md \
    docs/superpowers/plans/2026-07-12-zipformer-load-warmup.md \
    src/multi_stream_asr tests
  git stash drop stash@{0}
  git log -1 --oneline &&
  git status --short
'
```

Expected: board and local branch point to the same final commit; only pre-existing board-local untracked reproduction scripts remain.
