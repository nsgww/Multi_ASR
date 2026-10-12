from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import threading
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from multi_stream_asr.config import AppConfig, LaneInputConfig, ModelConfig
from multi_stream_asr.log.metrics import LatencyTracker, RtfTracker
from multi_stream_asr.main import main
from multi_stream_asr.models import ASREvent, AudioFrame, EventStatus, ResultType
from multi_stream_asr.pipeline.state import PipelineState
from multi_stream_asr.web.event_bus import FrontendEventBus
from multi_stream_asr.web import server as web_server
from multi_stream_asr.web.server import create_app


def run_index_javascript_scenario(html: str, scenario: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available on PATH; skipping browser behavior test")
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    harness = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeClassList {
  constructor() {
    this.names = new Set();
  }

  toggle(name, force) {
    const enabled = force === undefined ? !this.names.has(name) : Boolean(force);
    if (enabled) this.names.add(name);
    else this.names.delete(name);
    return enabled;
  }
}

class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.classList = new FakeClassList();
    this.listeners = new Map();
    this.queries = new Map();
    this.className = "";
    this.textContent = "";
    this.value = "";
    this.disabled = false;
    this.checked = false;
    this.open = false;
    this.parentPath = null;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 100;
    this.selectedOptions = [];
    this._innerHTML = "";
  }

  set innerHTML(value) {
    this._innerHTML = value;
    this.children = [];
    this.scrollHeight = 0;
  }

  get innerHTML() {
    return this._innerHTML;
  }

  appendChild(child) {
    this.children.push(child);
    child.parentElement = this;
    this.lastElementChild = child;
    this.scrollHeight = this.children.length * 20;
    return child;
  }

  remove() {
    if (!this.parentElement) return;
    const index = this.parentElement.children.indexOf(this);
    if (index >= 0) this.parentElement.children.splice(index, 1);
    this.parentElement.scrollHeight = this.parentElement.children.length * 20;
    this.parentElement.lastElementChild = this.parentElement.children.at(-1);
    this.parentElement = null;
  }

  querySelector(selector) {
    if (!this.queries.has(selector)) {
      this.queries.set(selector, new FakeElement(selector));
    }
    return this.queries.get(selector);
  }

  querySelectorAll(_selector) {
    return [];
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  setAttribute(name, value) {
    this[name] = value;
  }

  showModal() {
    this.open = true;
  }

  close() {
    this.open = false;
  }
}

const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new FakeElement(id));
    return elements.get(id);
  },
  createElement(tagName) {
    return new FakeElement(tagName);
  },
};

class FakeWebSocket {
  static instance = null;

  constructor(url) {
    this.url = url;
    this.listeners = new Map();
    FakeWebSocket.instance = this;
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  dispatchMessage(event) {
    this.listeners.get("message")({ data: JSON.stringify(event) });
  }
}

const windowObject = {
  location: { protocol: "http:", host: "testserver" },
  requestAnimationFrame(callback) { callback(); return 1; },
  setTimeout() { return 1; },
  clearTimeout() {},
  setInterval() { return 1; },
  clearInterval() {},
};
const context = vm.createContext({
  assert,
  console,
  document,
  elements,
  fetch: async () => ({
    ok: true,
    status: 200,
    json: async () => ({ state: "running", lanes: 4 }),
  }),
  URLSearchParams,
  FakeWebSocket,
  WebSocket: FakeWebSocket,
  window: windowObject,
});

(async () => {
  const source = fs.readFileSync(0, "utf8");
  vm.runInContext(source, context, { filename: "index.html" });
  await new Promise((resolve) => setImmediate(resolve));
  const scenario = process.env.INDEX_SCENARIO;
  await vm.runInContext(
    "(async () => {\n" + scenario + "\n})()",
    context,
    { filename: "index-scenario.js" },
  );
})().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
"""
    completed = subprocess.run(
        [node, "-e", harness],
        input=script,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "INDEX_SCENARIO": scenario},
    )
    assert completed.returncode == 0, completed.stderr


def test_index_javascript_scenario_skips_when_node_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(pytest.skip.Exception, match="Node.js is not available"):
        run_index_javascript_scenario("<script></script>", "")


@dataclass
class RecordingRecorder:
    saved_runs: list[dict[str, object]] = field(default_factory=list)

    def save(
        self,
        run_dir: str | Path,
        *,
        config: AppConfig,
        lane_metrics=(),
        latency_tracker=None,
        rtf_tracker=None,
        event_loop_lag_tracker=None,
        results=(),
    ) -> Path:
        self.saved_runs.append(
            {
                "run_dir": Path(run_dir),
                "config": config,
                "lane_metrics": list(lane_metrics),
                "latency_tracker": latency_tracker,
                "rtf_tracker": rtf_tracker,
                "results": results,
            }
        )
        return Path(run_dir)

    def save_process_run(
        self,
        run_dir: str | Path,
        *,
        config: AppConfig,
        lane_summaries,
    ) -> Path:
        self.saved_runs.append(
            {
                "run_dir": Path(run_dir),
                "config": config,
                "lane_summaries": list(lane_summaries),
                "process": True,
            }
        )
        return Path(run_dir)


class RecordingManager:
    def __init__(
        self,
        *,
        state: PipelineState = PipelineState.UNLOADED,
        event_bus: FrontendEventBus | None = None,
    ) -> None:
        self.state = state
        self.event_bus = event_bus
        self.calls: list[str] = []
        self.load_phase: str | None = None
        self.lane_metrics = []
        self.latency_tracker = LatencyTracker()
        self.rtf_tracker = RtfTracker()

    async def load_models(self) -> None:
        self.calls.append("load_models")
        self.state = PipelineState.LOADED

    async def start_recognition(self) -> None:
        self.calls.append("start_recognition")
        if self.event_bus is not None:
            self.event_bus.clear_results()
        self.state = PipelineState.RUNNING

    async def stop_recognition(self) -> None:
        self.calls.append("stop_recognition")
        self.state = PipelineState.LOADED

    def set_lane_inputs(self, lane_inputs) -> None:
        if self.state is not PipelineState.LOADED:
            raise ValueError(f"cannot update lane inputs from {self.state.value}")
        self.calls.append("set_lane_inputs")
        self.lane_inputs = lane_inputs

    async def unload_models(self) -> None:
        self.calls.append("unload_models")
        self.state = PipelineState.UNLOADED


class FailingStartManager(RecordingManager):
    async def start_recognition(self) -> None:
        self.calls.append("start_recognition")
        raise RuntimeError("start failed")


class InvalidMicrophoneManager(RecordingManager):
    async def start_recognition(self) -> None:
        self.calls.append("start_recognition")
        raise ValueError("lane 1 microphone device 20 is unavailable")


class InvalidWavStartManager(RecordingManager):
    async def start_recognition(self) -> None:
        self.calls.append("start_recognition")
        raise wave.Error("file does not start with RIFF id")


class FailingLoadManager(RecordingManager):
    async def load_models(self) -> None:
        self.calls.append("load_models")
        self.state = PipelineState.FATAL
        raise RuntimeError("load failed")


def app_config_with_readable_model_files(tmp_path: Path, lanes: int) -> AppConfig:
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    models = ModelConfig(
        offline_model_id="qwen3-asr-0.6b-q40",
        offline_model_dir=model_directory,
    )
    for filename in models.offline_model.required_filenames:
        (model_directory / filename).write_bytes(b"model")
    return replace(AppConfig(lanes=lanes), models=models, output_dir=tmp_path)


def write_valid_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\0\0")


def test_lane_inputs_update_while_models_are_loaded(tmp_path: Path):
    config = app_config_with_readable_model_files(tmp_path, lanes=2)
    valid_wav = tmp_path / "input.wav"
    write_valid_wav(valid_wav)
    manager = RecordingManager(state=PipelineState.LOADED)
    app = create_app(config=config, manager=manager, recorder=RecordingRecorder())

    with TestClient(app) as client:
        response = client.post(
            "/api/lane-inputs",
            json={
                "lane_inputs": [
                    {"kind": "local_wav", "local_wav_path": str(valid_wav)},
                    {
                        "kind": "microphone",
                        "microphone_device": 2,
                        "microphone_device_id": "USB microphone, PulseAudio",
                    },
                ]
            },
        )

    assert response.status_code == 200
    assert manager.calls[0] == "set_lane_inputs"
    assert response.json()["lane_inputs"] == [
        {
            "kind": "local_wav",
            "local_wav_path": str(valid_wav),
            "microphone_device": None,
            "microphone_device_id": None,
            "loop_audio": False,
        },
        {
            "kind": "microphone",
            "local_wav_path": None,
            "microphone_device": 2,
            "microphone_device_id": "USB microphone, PulseAudio",
            "loop_audio": False,
        },
    ]


def test_lane_inputs_preserve_four_lane_microphone_then_default_order(tmp_path: Path):
    config = app_config_with_readable_model_files(tmp_path, lanes=4)
    manager = RecordingManager(state=PipelineState.LOADED)

    with TestClient(create_app(config=config, manager=manager)) as client:
        response = client.post(
            "/api/lane-inputs",
            json={
                "lane_inputs": [
                    {"kind": "microphone", "microphone_device": 11},
                    {"kind": "microphone", "microphone_device": 22},
                    {"kind": "default_files"},
                    {"kind": "default_files"},
                ]
            },
        )
        assert response.status_code == 200
        assert manager.calls == ["set_lane_inputs"]

    assert manager.lane_inputs == (
        LaneInputConfig(kind="microphone", microphone_device=11),
        LaneInputConfig(kind="microphone", microphone_device=22),
        LaneInputConfig(kind="default_files"),
        LaneInputConfig(kind="default_files"),
    )


def test_lane_inputs_reject_boolean_microphone_device(tmp_path: Path):
    config = app_config_with_readable_model_files(tmp_path, lanes=2)
    manager = RecordingManager(state=PipelineState.LOADED)

    with TestClient(create_app(config=config, manager=manager)) as client:
        response = client.post(
            "/api/lane-inputs",
            json={
                "lane_inputs": [
                    {"kind": "microphone", "microphone_device": True},
                    {"kind": "default_files"},
                ]
            },
        )

    assert response.status_code == 422
    assert "set_lane_inputs" not in manager.calls


def test_lane_inputs_accept_loop_audio_for_file_inputs_and_reject_microphones(tmp_path: Path):
    config = app_config_with_readable_model_files(tmp_path, lanes=2)
    valid_wav = tmp_path / "input.wav"
    write_valid_wav(valid_wav)
    manager = RecordingManager(state=PipelineState.LOADED)

    with TestClient(create_app(config=config, manager=manager)) as client:
        accepted = client.post(
            "/api/lane-inputs",
            json={
                "lane_inputs": [
                    {"kind": "local_wav", "local_wav_path": str(valid_wav), "loop_audio": True},
                    {"kind": "default_files", "loop_audio": True},
                ]
            },
        )
        rejected = client.post(
            "/api/lane-inputs",
            json={
                "lane_inputs": [
                    {"kind": "microphone", "loop_audio": True},
                    {"kind": "default_files"},
                ]
            },
        )

    assert accepted.status_code == 200
    assert [item["loop_audio"] for item in accepted.json()["lane_inputs"]] == [True, True]
    assert rejected.status_code == 422


def test_path_browser_lists_readable_directories_and_wav_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    child_directory = home / "Child"
    wav_file = home / "sample.wav"
    home.mkdir()
    child_directory.mkdir()
    wav_file.write_bytes(b"wav")
    (home / "ignored.txt").write_text("ignore")
    monkeypatch.setattr(web_server.Path, "home", lambda: home)

    with TestClient(create_app(manager=RecordingManager())) as client:
        response = client.get("/api/path-browser", params={"selection": "wav"})

    assert response.status_code == 200
    assert response.json() == {
        "current_path": str(home.resolve()),
        "parent_path": str(tmp_path.resolve()),
        "entries": [
            {
                "name": "Child",
                "path": str(child_directory.resolve()),
                "kind": "directory",
            },
            {
                "name": "sample.wav",
                "path": str(wav_file.resolve()),
                "kind": "wav",
            },
        ],
    }


def test_path_browser_reopens_selected_wav_in_its_parent_directory(tmp_path):
    wav_directory = tmp_path / "recordings"
    wav_file = wav_directory / "sample.wav"
    wav_directory.mkdir()
    wav_file.write_bytes(b"wav")

    with TestClient(create_app(manager=RecordingManager())) as client:
        response = client.get(
            "/api/path-browser",
            params={"path": str(wav_file), "selection": "wav"},
        )

    assert response.status_code == 200
    assert response.json()["current_path"] == str(wav_directory.resolve())


def test_path_browser_excludes_wav_symlink_resolving_to_non_wav_file(tmp_path):
    directory = tmp_path / "recordings"
    target = directory / "target.txt"
    alias = directory / "alias.wav"
    directory.mkdir()
    target.write_text("not a WAV file")
    alias.symlink_to(target)

    with TestClient(create_app(manager=RecordingManager())) as client:
        response = client.get(
            "/api/path-browser",
            params={"path": str(directory), "selection": "wav"},
        )

    assert response.status_code == 200
    assert response.json()["entries"] == []


def test_path_browser_rejects_nonexistent_path(tmp_path):
    with TestClient(create_app(manager=RecordingManager())) as client:
        response = client.get(
            "/api/path-browser",
            params={"path": str(tmp_path / "missing"), "selection": "directory"},
        )

    assert response.status_code == 422


def test_path_browser_rejects_file_requested_as_directory(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"wav")

    with TestClient(create_app(manager=RecordingManager())) as client:
        response = client.get(
            "/api/path-browser",
            params={"path": str(path), "selection": "directory"},
        )

    assert response.status_code == 422


def test_path_browser_rejects_access_denied_directory(tmp_path, monkeypatch):
    denied_directory = tmp_path / "denied"
    denied_directory.mkdir()
    original_access = os.access

    def access(path, mode):
        if Path(path) == denied_directory and mode == os.R_OK | os.X_OK:
            return False
        return original_access(path, mode)

    monkeypatch.setattr(web_server.os, "access", access)

    with TestClient(create_app(manager=RecordingManager())) as client:
        response = client.get(
            "/api/path-browser",
            params={"path": str(denied_directory), "selection": "directory"},
        )

    assert response.status_code == 422


def test_record_results_uses_process_run_summary_when_manager_exposes_run_dir(tmp_path):
    class ProcessStyleManager(RecordingManager):
        def __init__(self) -> None:
            super().__init__(state=PipelineState.STOPPED)
            self.run_dir = tmp_path / "process-run"
            self.lane_summaries = [
                {
                    "lane_metrics": {
                        "stream_id": 0,
                        "processed_frames": 1,
                        "dropped_frames": 0,
                        "offline_requests": 1,
                        "final_state": "finished",
                    },
                    "online_frame_latency": {
                        "count": 1,
                        "total_ms": 50.0,
                        "average_ms": 50.0,
                        "max_ms": 50.0,
                    },
                    "offline_rtf": {
                        "count": 1,
                        "total": 0.4,
                        "average": 0.4,
                        "max": 0.4,
                    },
                    "event_loop_lag": {
                        "count": 1,
                        "total_ms": 1.0,
                        "average_ms": 1.0,
                        "max_ms": 1.0,
                    },
                    "results_path": "results/lane0.txt",
                    "model_outputs_path": "model_outputs/lane0.jsonl",
                }
            ]

    recorder = RecordingRecorder()
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=ProcessStyleManager(),
        event_bus=FrontendEventBus(),
        recorder=recorder,
    )

    with TestClient(app):
        pass

    assert recorder.saved_runs[-1]["process"] is True
    assert recorder.saved_runs[-1]["run_dir"] == tmp_path / "process-run"


def test_fatal_run_records_summary_after_completion_raises(tmp_path: Path) -> None:
    class FatalCompletionManager(RecordingManager):
        def __init__(self) -> None:
            super().__init__(state=PipelineState.FATAL)
            self.run_dir = tmp_path / "fatal-process-run"
            self.lane_summaries = [{"lane_metrics": {"final_state": "fatal"}}]
            self.wait_calls = 0

        async def wait_for_completion(self) -> None:
            self.wait_calls += 1
            raise RuntimeError("fatal recognition completion")

    recorder = RecordingRecorder()
    manager = FatalCompletionManager()
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=manager,
        event_bus=FrontendEventBus(),
        recorder=recorder,
    )

    asyncio.run(web_server._record_after_completion(app, manager))

    assert manager.wait_calls == 1
    assert manager.state is PipelineState.FATAL
    assert len(recorder.saved_runs) == 1
    assert recorder.saved_runs[0]["process"] is True
    assert recorder.saved_runs[0]["run_dir"] == tmp_path / "fatal-process-run"


@pytest.mark.asyncio
async def test_cancelling_completion_coordinator_preserves_shared_recorder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    partial_summary = {
        "lane_metrics": {
            "stream_id": 0,
            "processed_frames": 0,
            "offline_requests": 0,
            "final_state": "fatal",
        },
        "results_path": "results/partial.txt",
        "model_outputs_path": "model_outputs/partial.jsonl",
    }
    final_summary = {
        "lane_metrics": {
            "stream_id": 0,
            "processed_frames": 23,
            "offline_requests": 4,
            "final_state": "fatal",
        },
        "results_path": "results/lane0.txt",
        "model_outputs_path": "model_outputs/lane0.jsonl",
    }

    class FatalCompletionManager(RecordingManager):
        def __init__(self) -> None:
            super().__init__(state=PipelineState.FATAL)
            self.run_dir = tmp_path / "cancelled-coordinator-process-run"
            self.lane_summaries = [partial_summary]
            self.completion_started = asyncio.Event()
            self.release_completion = asyncio.Event()
            self.wait_calls = 0

        async def wait_for_completion(self) -> None:
            self.wait_calls += 1
            self.completion_started.set()
            await self.release_completion.wait()
            self.lane_summaries = [final_summary]

    recorder = RecordingRecorder()
    manager = FatalCompletionManager()
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=manager,
        event_bus=FrontendEventBus(),
        recorder=recorder,
    )

    completion_task = web_server._start_completion_recording(app)
    assert completion_task is not None
    await manager.completion_started.wait()

    shield_entries: list[tuple[asyncio.Task[None], asyncio.Task[None], asyncio.Future[None]]] = []
    shields_entered = asyncio.Event()
    original_shield = web_server.asyncio.shield

    def recording_shield(awaitable):
        assert awaitable is completion_task
        shielded = original_shield(awaitable)
        coordinator = asyncio.current_task()
        assert coordinator is not None
        shield_entries.append((coordinator, awaitable, shielded))
        if len(shield_entries) == 2:
            shields_entered.set()
        return shielded

    monkeypatch.setattr(web_server.asyncio, "shield", recording_shield)

    cancelled_waiter = asyncio.create_task(
        web_server._coordinate_completion_recording(app, manager)
    )
    surviving_waiter = asyncio.create_task(
        web_server._coordinate_completion_recording(app, manager)
    )
    await shields_entered.wait()

    assert len(shield_entries) == 2
    assert {entry[0] for entry in shield_entries} == {
        cancelled_waiter,
        surviving_waiter,
    }
    assert all(entry[1] is completion_task for entry in shield_entries)
    # A coordinator is actively waiting on its shield future, not merely past
    # the shield call.  This proves both waiters share the same completion task.
    assert all(
        coordinator._fut_waiter is shielded
        for coordinator, _shared_task, shielded in shield_entries
    )

    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter

    assert not completion_task.done()
    assert not surviving_waiter.done()
    assert recorder.saved_runs == []
    assert app.state.recorded_process_run_dirs == set()

    manager.release_completion.set()
    await surviving_waiter

    assert completion_task.done()
    assert not completion_task.cancelled()
    assert manager.wait_calls == 1
    assert len(recorder.saved_runs) == 1
    assert recorder.saved_runs[0]["run_dir"] == manager.run_dir
    assert recorder.saved_runs[0]["lane_summaries"] == [final_summary]
    assert app.state.recorded_process_run_dirs == {manager.run_dir.resolve()}


def test_unload_after_fatal_waits_for_final_completion_summary(tmp_path: Path) -> None:
    final_summary = {
        "lane_metrics": {
            "stream_id": 0,
            "processed_frames": 17,
            "offline_requests": 2,
            "final_state": "fatal",
        },
        "results_path": "results/lane0.txt",
        "model_outputs_path": "model_outputs/lane0.jsonl",
    }

    class FatalCleanupManager(RecordingManager):
        def __init__(self) -> None:
            super().__init__(state=PipelineState.LOADED)
            self.run_dir = None
            self.lane_summaries = []
            self.completion_started = threading.Event()
            self.release_completion = threading.Event()
            self.unload_called = threading.Event()
            self.unload_before_cleanup = False
            self.cleanup_finished = False

        async def start_recognition(self) -> None:
            self.calls.append("start_recognition")
            self.run_dir = tmp_path / "fatal-process-run"
            self.lane_summaries = [{"lane_metrics": {"final_state": "running"}}]
            self.state = PipelineState.RUNNING

        async def wait_for_completion(self) -> None:
            self.state = PipelineState.FATAL
            self.completion_started.set()
            await asyncio.to_thread(self.release_completion.wait)
            self.lane_summaries = [final_summary]
            self.cleanup_finished = True
            raise RuntimeError("fatal recognition completion")

        async def unload_models(self) -> None:
            self.unload_called.set()
            self.unload_before_cleanup = not self.cleanup_finished
            self.calls.append("unload_models")
            self.state = PipelineState.UNLOADED

    recorder = RecordingRecorder()
    manager = FatalCleanupManager()
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=manager,
        event_bus=FrontendEventBus(),
        recorder=recorder,
    )
    unload_responses = []

    with TestClient(app) as client:
        assert client.post("/api/recognition/start").status_code == 200
        assert manager.completion_started.wait(timeout=1.0)
        worker = threading.Thread(
            target=lambda: unload_responses.append(client.post("/api/models/unload"))
        )
        worker.start()
        try:
            assert not manager.unload_called.wait(timeout=0.1)
        finally:
            manager.release_completion.set()
            worker.join(timeout=1.0)

        assert not worker.is_alive()
        assert unload_responses[0].status_code == 200

    assert len(recorder.saved_runs) == 1
    assert manager.unload_before_cleanup is False
    assert recorder.saved_runs[0]["lane_summaries"] == [final_summary]


def test_concurrent_unloads_during_starting_fatal_wait_for_final_run_once(
    tmp_path: Path,
) -> None:
    partial_summary = {
        "lane_metrics": {
            "stream_id": 0,
            "processed_frames": 0,
            "offline_requests": 0,
            "final_state": "fatal",
        },
    }
    final_summary = {
        "lane_metrics": {
            "stream_id": 0,
            "processed_frames": 31,
            "offline_requests": 3,
            "final_state": "fatal",
        },
        "results_path": "results/lane0.txt",
        "model_outputs_path": "model_outputs/lane0.jsonl",
    }

    class StartingThresholdFatalManager(RecordingManager):
        def __init__(self) -> None:
            super().__init__(state=PipelineState.LOADED)
            self.run_dir = None
            self.lane_summaries = []
            self.cleanup_finished = False
            self.starting_fatal = threading.Event()
            self.release_cleanup = threading.Event()
            self.completion_waited = threading.Event()
            self.unload_called = threading.Event()
            self.unload_before_cleanup = False
            self.wait_calls = 0

        async def start_recognition(self) -> None:
            self.calls.append("start_recognition")
            self.state = PipelineState.STARTING
            self.run_dir = tmp_path / "starting-fatal-process-run"
            self.lane_summaries = [partial_summary]
            self.state = PipelineState.FATAL
            self.starting_fatal.set()
            await asyncio.to_thread(self.release_cleanup.wait)
            self.lane_summaries = [final_summary]
            self.cleanup_finished = True
            raise RuntimeError("Qwen failure threshold reached during startup")

        async def wait_for_completion(self) -> None:
            self.wait_calls += 1
            self.completion_waited.set()
            raise RuntimeError("fatal recognition completion")

        async def unload_models(self) -> None:
            self.unload_called.set()
            self.unload_before_cleanup |= not self.cleanup_finished
            self.calls.append("unload_models")
            self.state = PipelineState.UNLOADED

    recorder = RecordingRecorder()
    manager = StartingThresholdFatalManager()
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=manager,
        event_bus=FrontendEventBus(),
        recorder=recorder,
    )
    start_responses = []
    unload_responses = []

    with TestClient(app, raise_server_exceptions=False) as client:
        start_worker = threading.Thread(
            target=lambda: start_responses.append(
                client.post("/api/recognition/start")
            )
        )
        start_worker.start()
        assert manager.starting_fatal.wait(timeout=1.0)

        unload_workers = [
            threading.Thread(
                target=lambda: unload_responses.append(
                    client.post("/api/models/unload")
                )
            )
            for _ in range(2)
        ]
        for worker in unload_workers:
            worker.start()
        try:
            assert not manager.completion_waited.wait(timeout=0.1)
            assert not manager.unload_called.is_set()
            assert recorder.saved_runs == []
        finally:
            manager.release_cleanup.set()
            start_worker.join(timeout=2.0)
            for worker in unload_workers:
                worker.join(timeout=2.0)

        assert not start_worker.is_alive()
        assert all(not worker.is_alive() for worker in unload_workers)
        assert [response.status_code for response in start_responses] == [500]
        assert sorted(response.status_code for response in unload_responses) == [200, 200]

    assert len(recorder.saved_runs) == 1
    assert manager.wait_calls == 1
    assert manager.unload_before_cleanup is False
    assert recorder.saved_runs[0]["run_dir"] == tmp_path / "starting-fatal-process-run"
    assert recorder.saved_runs[0]["lane_summaries"] == [final_summary]
    recorded_summary = recorder.saved_runs[0]["lane_summaries"][0]
    assert recorded_summary["lane_metrics"]["processed_frames"] == 31
    assert recorded_summary["results_path"] == "results/lane0.txt"
    assert recorded_summary["model_outputs_path"] == "model_outputs/lane0.jsonl"


def test_health_returns_state_and_lane_count(tmp_path):
    config = replace(AppConfig(lanes=4), output_dir=tmp_path)

    with TestClient(
        create_app(
            config=config,
            manager=RecordingManager(),
            event_bus=FrontendEventBus(),
            recorder=RecordingRecorder(),
        )
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["state"] == "unloaded"
    assert response.json()["lanes"] == 4


def test_health_exposes_absolute_default_directory_for_each_offline_model(tmp_path):
    app = create_app(
        config=app_config_with_readable_model_files(tmp_path, lanes=2),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        payload = client.get("/health").json()

    assert payload["offline_model_dirs"] == {
        model_id: str(spec.model_dir)
        for model_id, spec in web_server.OFFLINE_MODELS.items()
    }
    assert all(Path(path).is_absolute() for path in payload["offline_model_dirs"].values())


def test_health_returns_global_model_load_phase(tmp_path):
    manager = RecordingManager(state=PipelineState.LOADING)
    manager.load_phase = "warming_zipformer"

    with TestClient(
        create_app(
            config=replace(AppConfig(lanes=4), output_dir=tmp_path),
            manager=manager,
            event_bus=FrontendEventBus(),
            recorder=RecordingRecorder(),
        )
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["load_phase"] == "warming_zipformer"


def test_model_validation_reports_missing_files_without_creating_manager(tmp_path):
    built: list[RecordingManager] = []

    def manager_factory(config: AppConfig, event_bus: FrontendEventBus) -> RecordingManager:
        manager = RecordingManager()
        built.append(manager)
        return manager

    app = create_app(manager_factory=manager_factory)
    with TestClient(app) as client:
        response = client.post(
            "/api/models/validate",
            json={
                "offline_model_id": "qwen3-asr-1.7b-q4km",
                "offline_model_dir": str(tmp_path / "missing"),
            },
        )

    assert response.json() == {
        "valid": False,
        "model_directory": [str((tmp_path / "missing").resolve())],
        "missing_files": [str((tmp_path / "missing").resolve())],
    }
    assert len(built) == 1


def test_load_rejects_missing_selected_model_files_before_manager_load(tmp_path):
    manager = RecordingManager()
    app = create_app(config=AppConfig(), manager=manager)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/models/load",
            json={
                "offline_model_id": "qwen3-asr-0.6b-q40",
                "offline_model_dir": str(tmp_path / "missing"),
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["model_directory"] == [
        str((tmp_path / "missing").resolve())
    ]
    assert manager.calls == []


def test_load_rejects_unreadable_model_directory_before_manager_load(
    tmp_path,
    monkeypatch,
):
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    manager = RecordingManager()
    app = create_app(config=AppConfig(), manager=manager)

    real_access = os.access
    monkeypatch.setattr(
        web_server.os,
        "access",
        lambda path, mode: (
            False
            if Path(path) == model_directory.resolve() and mode == os.R_OK | os.X_OK
            else real_access(path, mode)
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/models/load",
            json={
                "offline_model_id": "qwen3-asr-0.6b-q40",
                "offline_model_dir": str(model_directory),
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["model_directory"] == [str(model_directory.resolve())]
    assert manager.calls == []


def test_invalid_path_load_does_not_construct_reconfigured_manager(tmp_path):
    built_configs: list[AppConfig] = []

    def manager_factory(config: AppConfig, event_bus: FrontendEventBus) -> RecordingManager:
        built_configs.append(config)
        return RecordingManager()

    initial_config = app_config_with_readable_model_files(tmp_path, lanes=4)
    app = create_app(config=initial_config, manager_factory=manager_factory)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/models/load",
            json={
                "offline_model_id": "qwen3-asr-1.7b-q4km",
                "offline_model_dir": str(tmp_path / "missing-models"),
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["model_directory"] == [
        str((tmp_path / "missing-models").resolve())
    ]
    assert built_configs == [initial_config]


def test_model_load_does_not_validate_existing_lane_inputs(tmp_path: Path):
    config = app_config_with_readable_model_files(tmp_path, lanes=2)
    config = replace(
        config,
        lane_inputs=(
            LaneInputConfig(kind="local_wav", local_wav_path=tmp_path / "missing.wav"),
            LaneInputConfig(),
        ),
    )
    manager = RecordingManager()

    with TestClient(create_app(config=config, manager=manager)) as client:
        response = client.post("/api/models/load")
        assert manager.calls == ["load_models"]

    assert response.status_code == 200


def test_lane_input_submit_rejects_invalid_wav_before_updating_manager(tmp_path: Path):
    config = app_config_with_readable_model_files(tmp_path, lanes=2)
    manager = RecordingManager(state=PipelineState.LOADED)
    invalid_wav = tmp_path / "invalid.wav"
    invalid_wav.write_bytes(b"not wav")

    with TestClient(create_app(config=config, manager=manager)) as client:
        response = client.post(
            "/api/lane-inputs",
            json={
                "lane_inputs": [
                    {"kind": "local_wav", "local_wav_path": str(invalid_wav)},
                    {"kind": "default_files"},
                ]
            },
        )
        assert manager.calls == []

    assert response.status_code == 422
    assert response.json()["detail"]["lane_inputs"] == [str(invalid_wav.resolve())]


def test_load_rejects_unreadable_required_model_file_before_manager_load(
    tmp_path,
    monkeypatch,
):
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    model_spec = web_server.ModelConfig(
        offline_model_id="qwen3-asr-0.6b-q40",
        offline_model_dir=model_directory,
    ).offline_model
    for filename in model_spec.required_filenames:
        (model_directory / filename).write_bytes(b"model")
    unreadable_path = model_directory / model_spec.frontend_filename
    manager = RecordingManager()
    app = create_app(config=AppConfig(), manager=manager)

    real_access = os.access
    monkeypatch.setattr(
        web_server.os,
        "access",
        lambda path, mode: (
            False
            if Path(path) == unreadable_path.resolve() and mode == os.R_OK
            else real_access(path, mode)
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/models/load",
            json={
                "offline_model_id": "qwen3-asr-0.6b-q40",
                "offline_model_dir": str(model_directory),
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["model_files"] == [str(unreadable_path.resolve())]
    assert manager.calls == []


def test_load_rejects_required_model_file_symlink_outside_directory_before_manager_load(
    tmp_path: Path,
):
    model_directory = tmp_path / "models"
    external_directory = tmp_path / "external"
    model_directory.mkdir()
    external_directory.mkdir()
    model_spec = web_server.ModelConfig(
        offline_model_id="qwen3-asr-0.6b-q40",
        offline_model_dir=model_directory,
    ).offline_model
    for filename in model_spec.required_filenames:
        (model_directory / filename).write_bytes(b"model")

    external_target = external_directory / model_spec.frontend_filename
    external_target.write_bytes(b"model")
    external_link = model_directory / model_spec.frontend_filename
    external_link.unlink()
    external_link.symlink_to(external_target)
    manager = RecordingManager()
    app = create_app(config=AppConfig(), manager=manager)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/models/load",
            json={
                "offline_model_id": "qwen3-asr-0.6b-q40",
                "offline_model_dir": str(model_directory),
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["model_files"] == [str(external_target.resolve())]
    assert manager.calls == []


def test_model_routes_reject_malformed_model_directory_without_constructing_manager():
    malformed_path = "models" + chr(0) + "invalid"
    manager = RecordingManager()
    app = create_app(config=AppConfig(), manager=manager)
    requests = (
        (
            "/api/models/validate",
            {
                "offline_model_id": "qwen3-asr-0.6b-q40",
                "offline_model_dir": malformed_path,
            },
        ),
        (
            "/api/models/load",
            {
                "offline_model_id": "qwen3-asr-0.6b-q40",
                "offline_model_dir": malformed_path,
            },
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        for url, payload in requests:
            response = client.post(url, json=payload)

            assert response.status_code == 422
            assert response.json() == {
                "detail": {"model_directory": [malformed_path]}
            }

    assert manager.calls == []


def test_failed_initial_load_does_not_record_run_artifacts(tmp_path):
    recorder = RecordingRecorder()
    manager = FailingLoadManager()
    app = create_app(
        config=app_config_with_readable_model_files(tmp_path, lanes=2),
        manager=manager,
        recorder=recorder,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/models/load")

    assert response.status_code == 500
    assert recorder.saved_runs == []


def test_audio_devices_returns_enumerated_input_devices(monkeypatch):
    monkeypatch.setattr(
        web_server,
        "list_microphone_devices",
        lambda: [{"index": 2, "name": "USB"}],
    )
    with TestClient(create_app(manager=RecordingManager())) as client:
        response = client.get("/api/audio-devices")

    assert response.json() == {"devices": [{"index": 2, "name": "USB"}]}
    assert response.headers["cache-control"] == "no-store"


def test_audio_devices_responds_while_models_are_loading(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    load_responses = []

    class PausingLoadManager(RecordingManager):
        async def load_models(self) -> None:
            started.set()
            await asyncio.to_thread(release.wait)
            self.state = PipelineState.LOADED

    monkeypatch.setattr(
        web_server,
        "list_microphone_devices",
        lambda: [{"index": 15, "name": "USB microphone"}],
    )
    app = create_app(
        config=app_config_with_readable_model_files(tmp_path, lanes=2),
        manager=PausingLoadManager(),
    )

    with TestClient(app) as client:
        worker = threading.Thread(
            target=lambda: load_responses.append(client.post("/api/models/load"))
        )
        worker.start()
        try:
            assert started.wait(timeout=1.0)
            devices = client.get("/api/audio-devices")
        finally:
            release.set()
            worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert devices.status_code == 200
    assert devices.json()["devices"] == [{"index": 15, "name": "USB microphone"}]
    assert load_responses[0].status_code == 200


def test_control_endpoints_call_manager_methods_and_allow_lane_reconfiguration(tmp_path):
    built_managers: list[RecordingManager] = []
    built_configs: list[AppConfig] = []

    def manager_factory(config: AppConfig, event_bus: FrontendEventBus) -> RecordingManager:
        built_configs.append(config)
        manager = RecordingManager()
        built_managers.append(manager)
        return manager

    app = create_app(
        config=app_config_with_readable_model_files(tmp_path, lanes=4),
        manager_factory=manager_factory,
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        load_response = client.post("/api/models/load", json={"lanes": 6})
        start_response = client.post("/api/recognition/start")
        stop_response = client.post("/api/recognition/stop")
        unload_response = client.post("/api/models/unload")

    assert [config.lanes for config in built_configs] == [4, 6]
    assert built_managers[0].calls == []
    assert built_managers[1].calls == [
        "load_models",
        "start_recognition",
        "stop_recognition",
        "unload_models",
    ]
    assert load_response.json()["lanes"] == 6
    assert start_response.json()["state"] == "running"
    assert stop_response.json()["state"] == "loaded"
    assert unload_response.json()["state"] == "unloaded"


def test_start_recognition_returns_validation_error_without_starting_run():
    manager = InvalidMicrophoneManager(state=PipelineState.LOADED)
    with TestClient(create_app(manager=manager), raise_server_exceptions=False) as client:
        response = client.post("/api/recognition/start")
        assert manager.state is PipelineState.LOADED

    assert response.status_code == 422
    assert response.json() == {"detail": "lane 1 microphone device 20 is unavailable"}


def test_start_recognition_returns_json_422_for_malformed_default_wav():
    manager = InvalidWavStartManager(state=PipelineState.LOADED)

    with TestClient(create_app(manager=manager), raise_server_exceptions=False) as client:
        response = client.post("/api/recognition/start")
        assert manager.state is PipelineState.LOADED
        assert manager.calls == ["start_recognition"]

    assert response.status_code == 422
    assert response.json() == {
        "detail": "audio input validation failed: file does not start with RIFF id"
    }


def test_websocket_receives_published_asr_event(tmp_path):
    bus = FrontendEventBus()
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=bus,
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            bus.publish(
                ASREvent.asr_result(
                    stream_id=1,
                    sentence_id=2,
                    revision=3,
                    status=EventStatus.FINAL,
                    text="final text",
                    timestamp_ms=400,
                )
            )

            message = websocket.receive_json()

    assert message == {
        "event_type": "asr_result",
        "stream_id": 1,
        "sentence_id": 2,
        "revision": 3,
        "status": "final",
        "result_type": "offline",
        "is_final": True,
        "text": "final text",
        "timestamp_ms": 400,
    }


def test_index_html_references_and_serves_local_spacemit_logo(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        index = client.get("/")
        logo = client.get("/spacemit-logo.svg")

    assert index.status_code == 200
    assert 'src="/spacemit-logo.svg"' in index.text
    assert 'class="brand"' in index.text
    assert 'class="brand-copy"' in index.text
    assert "--accent: #b0e237" in index.text.lower()
    assert ".lane::before" in index.text
    assert logo.status_code == 200
    assert logo.headers["content-type"].startswith("image/svg+xml")
    assert "#B0E237" in logo.text


def test_index_html_contains_lane_selector_and_controls(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    for lanes in ("2", "4", "6", "8"):
        assert f'value="{lanes}"' in response.text
    for control in ("Load Models", "Unload Models", "Start Recognition", "Stop Recognition"):
        assert control in response.text


def test_index_html_websocket_fatal_updates_controls_and_preserves_sentence_zero_rows(
    tmp_path,
):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    run_index_javascript_scenario(
        response.text,
        r"""
const socket = FakeWebSocket.instance;
assert.ok(socket);
const event = (status, revision, text) => ({
  event_type: "asr_result",
  stream_id: 0,
  sentence_id: 0,
  revision,
  status,
  result_type: status === "final" || status === "qwen_error" ? "offline" : "online",
  is_final: status === "final",
  text,
  timestamp_ms: revision,
});
const rows = () => elements
  .get("lane-grid")
  .children[0]
  .querySelector(".results")
  .children;

assert.equal(elements.get("server-state").textContent, "State: running");
assert.equal(elements.get("unload-models").disabled, true);
assert.equal(elements.get("stop-recognition").disabled, false);

socket.dispatchMessage(event("pending", 1, "pending sentence zero"));
socket.dispatchMessage(event("qwen_error", 2, "Qwen request failed"));
socket.dispatchMessage(event("fatal_error", 3, "Qwen failure threshold reached"));

assert.equal(elements.get("server-state").textContent, "State: fatal");
assert.equal(elements.get("unload-models").disabled, false);
assert.equal(elements.get("start-recognition").disabled, true);
assert.equal(elements.get("stop-recognition").disabled, true);
assert.equal(
  rows().map((row) => row.className).join(","),
  "result pending,result qwen_error,result fatal_error",
);
assert.equal(rows()[0].textContent, "pending sentence zero");

socket.dispatchMessage(event("final", 4, ""));

assert.equal(
  rows().map((row) => row.className).join(","),
  "result final,result qwen_error,result fatal_error",
);
assert.equal(rows()[0].textContent, "");
""",
    )


def test_index_html_failed_start_keeps_events_received_while_post_is_pending(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    run_index_javascript_scenario(
        response.text,
        r"""
const socket = FakeWebSocket.instance;
const event = (status, revision, text) => ({
  event_type: "asr_result",
  stream_id: 0,
  sentence_id: 0,
  revision,
  status,
  result_type: status === "qwen_error" ? "offline" : "online",
  is_final: false,
  text,
  timestamp_ms: revision,
});
const response = (ok, status, payload) => ({
  ok,
  status,
  json: async () => payload,
});
fetch = async (url) => {
  if (url === "/api/lane-inputs") {
    return response(true, 200, { state: "loaded", lanes: 4 });
  }
  if (url === "/api/recognition/start") {
    socket.dispatchMessage(event("pending", 1, "pending sentence zero"));
    socket.dispatchMessage(event("qwen_error", 2, "Qwen request failed"));
    socket.dispatchMessage(event("fatal_error", 3, "Qwen failure threshold reached"));
    return response(false, 500, { detail: "fatal recognition start" });
  }
  if (url === "/health") {
    return response(true, 200, { state: "fatal", lanes: 4 });
  }
  throw new Error(`unexpected request: ${url}`);
};
appState.state = "loaded";
renderStatus();

await elements.get("start-recognition").listeners.get("click")();

const rows = elements
  .get("lane-grid")
  .children[0]
  .querySelector(".results")
  .children;
assert.equal(elements.get("server-state").textContent, "State: fatal");
assert.equal(elements.get("unload-models").disabled, false);
assert.equal(elements.get("start-recognition").disabled, true);
assert.equal(elements.get("stop-recognition").disabled, true);
assert.equal(
  rows.map((row) => row.className).join(","),
  "result pending,result qwen_error,result fatal_error",
);
assert.equal(rows[0].textContent, "pending sentence zero");
""",
    )


def test_index_html_retains_exact_sentence_capacity_and_ignores_delayed_qwen_error(
    tmp_path,
):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    run_index_javascript_scenario(
        response.text,
        r"""
const socket = FakeWebSocket.instance;
const event = (sentenceId, status, revision, text) => ({
  event_type: "asr_result",
  stream_id: 0,
  sentence_id: sentenceId,
  revision,
  status,
  result_type: status === "qwen_error" ? "offline" : "online",
  is_final: false,
  text,
  timestamp_ms: revision,
});
const rows = () => elements
  .get("lane-grid")
  .children[0]
  .querySelector(".results")
  .children;

for (let sentenceId = 0; sentenceId < 100; sentenceId += 1) {
  socket.dispatchMessage(event(sentenceId, "pending", 1, `pending ${sentenceId}`));
}
socket.dispatchMessage(event(0, "qwen_error", 2, "Qwen request failed"));

assert.equal(rows().length, 101);
assert.equal(rows()[0].textContent, "pending 0");
assert.equal(rows()[100].className, "result qwen_error");

socket.dispatchMessage(event(100, "pending", 1, "pending 100"));
assert.equal(rows().length, 100);
assert.equal(rows()[0].textContent, "pending 1");
assert.equal(rows().some((row) => row.className === "result qwen_error"), false);

socket.dispatchMessage(event(0, "qwen_error", 3, "delayed old failure"));
assert.equal(rows().length, 100);
assert.equal(rows()[0].textContent, "pending 1");
assert.equal(rows().some((row) => row.textContent === "delayed old failure"), false);
""",
    )


def test_index_html_contains_native_path_picker_controls(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert 'id="offline-model"' in response.text
    assert 'id="path-picker-dialog"' in response.text
    assert 'id="offline-model-picker"' in response.text
    assert 'class="local-wav-picker icon-button hidden"' in response.text
    assert "function openPathDialog" in response.text
    assert "/api/path-browser" in response.text
    assert "showModal()" in response.text
    assert "close()" in response.text
    assert "pathPickerUp.disabled = !pathPickerEditable || !pathPicker.parentPath" in response.text
    assert "pathPickerSelect.disabled = !pathPickerEditable || (" in response.text
    assert 'for (const element of pathPickerEntries.querySelectorAll("button")) {' in response.text
    assert "element.disabled = !pathPickerEditable;" in response.text
    assert "pathPickerSelect.disabled = false;" not in response.text
    assert "pathPicker.pendingPath = entry.path;" in response.text
    assert "updateControlAvailability();" in response.text
    assert '<input id="offline-model-dir"' not in response.text
    assert '<output id="offline-model-dir"' in response.text


def test_index_html_preserves_draft_paths_when_model_request_fails(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    refresh_start = response.text.index("async function refreshOperationalState()")
    refresh_body = response.text[
        refresh_start : response.text.index("async function pollLoadPhase()", refresh_start)
    ]
    load_handler_start = response.text.index(
        'loadButton.addEventListener("click", async () => {'
    )
    load_handler = response.text[
        load_handler_start : response.text.index(
            'unloadButton.addEventListener("click", async () => {', load_handler_start
        )
    ]
    unload_handler_start = response.text.index(
        'unloadButton.addEventListener("click", async () => {'
    )
    unload_handler = response.text[
        unload_handler_start : response.text.index(
            'document.getElementById("start-recognition").addEventListener',
            unload_handler_start,
        )
    ]
    assert "await fetch(\"/health\")" in refresh_body
    assert "loadDraftFromPayload" not in refresh_body
    assert "appState.draft.offlineModelDir" not in refresh_body
    assert "appState.draft.laneInputs" not in refresh_body
    assert "await refreshOperationalState();" in load_handler
    assert "await loadHealth();" not in load_handler
    assert "await refreshOperationalState();" in unload_handler
    assert "await loadHealth();" not in unload_handler


def test_index_html_reopens_selected_wav_via_path_browser(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    dialog_start = response.text.index("async function openPathDialog")
    dialog_body = response.text[
        dialog_start : response.text.index("async function loadPathEntries", dialog_start)
    ]
    assert "await loadPathEntries(value || null);" in dialog_body
    assert 'selection: "wav",' in response.text
    assert "value: current.local_wav_path," in response.text


def test_index_html_contains_configuration_state_machine_and_scroll_support(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'id="offline-model-dir"' in response.text
    assert 'id="configuration-error"' in response.text
    assert 'id="lane-input-config"' in response.text
    assert "function renderLaneInputControls()" in response.text
    assert "function updateControlAvailability()" in response.text
    assert "function validateModelSelection()" in response.text
    assert "function scrollResultsIfPinnedToBottom" in response.text
    assert '.lane-grid[data-lanes="6"]' in response.text
    assert ".lane .results {" in response.text
    assert "overflow-y: auto;" in response.text
    assert "loadButton.disabled" in response.text
    assert "unloadButton.disabled" in response.text
    assert "startButton.disabled" in response.text
    assert "stopButton.disabled" in response.text


def test_index_html_keeps_microphone_device_selector_inside_lane_input_row(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert ".lane-input-row { flex-wrap: nowrap; }" in response.text
    assert ".input-kind { width: 128px; min-width: 0; }" in response.text
    assert ".microphone-device {" in response.text
    assert "flex: 1 1 0;" in response.text
    assert "text-overflow: ellipsis;" in response.text


def test_index_html_restores_recognition_view_only_before_start_is_submitted(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "function resetRecognitionView()" in response.text
    assert "appState.revisions = new Map();" in response.text
    assert "function snapshotRecognitionView()" in response.text
    assert "function restoreRecognitionView(snapshot)" in response.text
    start_marker = (
        'document.getElementById("start-recognition").addEventListener("click", async () => {'
    )
    start_start = response.text.index(start_marker)
    start_end = response.text.index('stopButton.addEventListener("click", async () => {', start_start)
    start_handler = response.text[start_start:start_end]
    snapshot_call = "const snapshot = snapshotRecognitionView();"
    reset_call = "resetRecognitionView();"
    submit_call = 'await postJson("/api/lane-inputs", {'
    start_call = 'await postJson("/api/recognition/start");'
    submitted_flag = "let recognitionStartSubmitted = false;"
    submitted_set = "recognitionStartSubmitted = true;"
    conditional_restore = "if (!recognitionStartSubmitted) restoreRecognitionView(snapshot);"
    assert snapshot_call in start_handler
    assert submitted_flag in start_handler
    assert submit_call in start_handler
    assert "lane_inputs: appState.draft.laneInputs" in start_handler
    assert submitted_set in start_handler
    assert start_call in start_handler
    assert reset_call in start_handler
    assert conditional_restore in start_handler
    assert "await refreshOperationalState();" in start_handler
    assert "await loadHealth();" not in start_handler
    assert (
        start_handler.index(snapshot_call)
        < start_handler.index(submit_call)
        < start_handler.index(submitted_set)
        < start_handler.index(start_call)
        < start_handler.index(reset_call)
        < start_handler.index(conditional_restore)
    )


def test_index_html_separates_model_and_lane_input_availability(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert "laneInputUpdatePromise" not in response.text
    assert "persistLaneInputs" not in response.text
    assert "microphoneDevices: []" in response.text
    assert "microphoneDeviceRefreshPending: false" in response.text
    assert "laneInputSubmissionPending: false" in response.text
    assert 'const modelBusy = ["loading", "unloading"].includes(appState.state);' in response.text
    assert (
        'const recognitionActive = ["starting", "running", "stopping"].includes(appState.state);'
        in response.text
    )
    assert 'const modelEditable = appState.state === "unloaded" && !modelBusy;' in response.text
    assert (
        "const laneInputsEditable = !recognitionActive "
        "&& !appState.laneInputSubmissionPending;"
    ) in response.text
    assert "element.disabled = !laneInputsEditable;" in response.text
    assert (
        "unloadButton.disabled = "
        '!["loaded", "fatal"].includes(appState.state) || modelBusy '
        "|| recognitionActive || appState.laneInputSubmissionPending;"
    ) in response.text
    assert (
        'startButton.disabled = appState.state !== "loaded"\n'
        "          || appState.laneInputSubmissionPending\n"
        "          || appState.microphoneDeviceRefreshPending\n"
        "          || appState.microphoneDeviceErrors.size > 0;"
    ) in response.text
    assert 'stopButton.disabled = !["starting", "running"].includes(appState.state);' in response.text
    assert "function updateLaneInputVisibility(row, input)" in response.text
    assert "function waitForRecognitionCompletion()" in response.text
    assert 'event.status === "lane_finished"' in response.text
    assert '![' + '"loaded", "stopped"' + '].includes(appState.state)' not in response.text
    load_start = response.text.index('loadButton.addEventListener("click", async () => {')
    load_end = response.text.index('unloadButton.addEventListener("click", async () => {', load_start)
    load_handler = response.text[load_start:load_end]
    assert "lane_inputs" not in load_handler
    microphone_start = response.text.index("async function loadMicrophoneDevices()")
    microphone_end = response.text.index("function populateMicrophoneDevices", microphone_start)
    microphone_loader = response.text[microphone_start:microphone_end]
    assert 'fetch("/api/audio-devices", { cache: "no-store" })' in microphone_loader
    controls_start = response.text.index("function renderLaneInputControls()")
    kind_start = response.text.index('kind.addEventListener("change", async () => {', controls_start)
    kind_end = response.text.index("pathPicker.addEventListener", kind_start)
    kind_handler = response.text[kind_start:kind_end]
    assert "const current = appState.draft.laneInputs[streamId];" in kind_handler
    assert "updateLaneInputVisibility(row, current);" in kind_handler
    assert 'postJson("/api/lane-inputs"' not in kind_handler

    start_marker = (
        'document.getElementById("start-recognition").addEventListener("click", async () => {'
    )
    start_start = response.text.index(start_marker)
    start_end = response.text.index(
        'stopButton.addEventListener("click", async () => {', start_start
    )
    start_handler = response.text[start_start:start_end]
    pending_start = "appState.laneInputSubmissionPending = true;"
    submit_call = 'await postJson("/api/lane-inputs", {'
    pending_end = "appState.laneInputSubmissionPending = false;"
    starting_state = 'appState.state = "starting";'
    start_call = 'await postJson("/api/recognition/start");'
    assert (
        start_handler.index(pending_start)
        < start_handler.index(submit_call)
        < start_handler.index(pending_end)
        < start_handler.index(starting_state)
        < start_handler.index(start_call)
    )


def test_index_html_prunes_removed_lane_microphone_errors(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    ensure_start = response.text.index("function ensureDraftLaneInputs()")
    ensure_end = response.text.index("function showError(message)", ensure_start)
    ensure_handler = response.text[ensure_start:ensure_end]
    assert "const previousLaneCount = appState.draft.laneInputs.length;" in ensure_handler
    assert "if (appState.lanes < previousLaneCount)" in ensure_handler
    assert (
        "for (const streamId of appState.microphoneDeviceErrors.keys())"
        in ensure_handler
    )
    assert "appState.microphoneDeviceErrors.delete(streamId);" in ensure_handler
    assert "renderConfigurationErrors();" in ensure_handler

    assert "microphoneDeviceRequestGeneration: 0" in response.text
    generation_start = response.text.index(
        "function nextMicrophoneDeviceRequestGeneration()"
    )
    generation_end = response.text.index(
        "function isCurrentMicrophoneDeviceRequest", generation_start
    )
    generation_handler = response.text[generation_start:generation_end]
    assert "appState.microphoneDeviceRequestGeneration += 1;" in generation_handler
    assert "return appState.microphoneDeviceRequestGeneration;" in generation_handler


def test_index_html_ignores_stale_microphone_device_responses(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert "microphoneDeviceRequestGenerations: new Map()" not in response.text
    assert "microphoneDeviceErrors: new Map()" in response.text
    assert "function setMicrophoneDeviceError(streamId, message)" in response.text
    controls_start = response.text.index("function renderLaneInputControls()")
    kind_start = response.text.index('kind.addEventListener("change", async () => {', controls_start)
    kind_end = response.text.index("pathPicker.addEventListener", kind_start)
    kind_handler = response.text[kind_start:kind_end]
    request_call = "requestMicrophoneDevices();"
    assert request_call in kind_handler
    assert kind_handler.index("current.kind = kind.value;") < kind_handler.index(request_call)

    request_start = response.text.index("function requestMicrophoneDevices(")
    request_end = response.text.index("function updateControlAvailability()", request_start)
    request_handler = response.text[request_start:request_end]
    current_request = "isCurrentMicrophoneDeviceRequest(generation)"
    assert request_handler.count(current_request) == 3
    assert request_handler.index(current_request) < request_handler.index("refreshAllMicrophoneDeviceControls(")
    error_handler = request_handler[request_handler.index(".catch((error) => {") :]
    assert error_handler.index(current_request) < error_handler.index(
        "setMicrophoneDeviceError(streamId, error.message);"
    )
    assert "refreshAllMicrophoneDeviceControls(devices);" in request_handler
    assert "showError(" not in request_handler

    current_start = response.text.index("function isCurrentMicrophoneDeviceRequest(")
    current_end = response.text.index("function requestMicrophoneDevices(", current_start)
    current_request_handler = response.text[current_start:current_end]
    assert "appState.microphoneDeviceRequestGeneration === generation" in current_request_handler


def test_index_html_reconciles_all_microphone_lanes_by_stable_id(tmp_path):
    with TestClient(create_app(manager=RecordingManager())) as client:
        html = client.get("/").text

    assert "microphone_device_id: null" in html
    assert 'option.value = device.id;' in html
    assert 'option.dataset.index = String(device.index);' in html
    assert "function refreshAllMicrophoneDeviceControls(devices)" in html
    assert "appState.microphoneDevices = devices;" in html
    assert "appState.microphoneDeviceRefreshPending = true;" in html
    assert "appState.microphoneDeviceRefreshPending = false;" in html
    assert "current.microphone_device_id = selectedId;" in html
    assert "Number(selectedOption.dataset.index)" in html
    assert 'available ? "" : "Selected microphone is unavailable"' in html


def test_index_html_has_loop_control_only_for_file_inputs(tmp_path):
    with TestClient(create_app(manager=RecordingManager())) as client:
        html = client.get("/").text

    assert 'loop.className = "loop-audio hidden";' in html
    assert 'appState.draft.laneInputs[streamId].loop_audio = loopCheckbox.checked;' in html
    assert 'input.kind !== "microphone"' in html
    assert "loop_audio: false" in html
    assert "MAX_DISPLAYED_SENTENCES_PER_LANE = 100" in html
    assert "while (sentenceKeys.length > MAX_DISPLAYED_SENTENCES_PER_LANE)" in html


def test_index_html_uses_server_model_directory_mapping(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert "offlineModelDirectories: {}" in response.text
    assert "payload.offline_model_dirs" in response.text
    assert (
        "appState.draft.offlineModelDir = "
        "appState.offlineModelDirectories[offlineModelSelect.value];"
    ) in response.text
    assert '"models/asr/qwen3asr"' not in response.text


def test_index_html_checks_post_response_ok_before_mutating_app_state(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    ok_check = "if (!response.ok) {"
    state_update = "appState.state = payload.state;"
    lane_update = "appState.lanes = payload.lanes;"
    assert ok_check in response.text
    assert state_update in response.text
    assert lane_update in response.text
    assert response.text.index(ok_check) < response.text.index(state_update)
    post_json = response.text[
        response.text.index("async function postJson") : response.text.index("async function validateModel")
    ]
    assert post_json.index(ok_check) < post_json.index(state_update)
    assert "try {" in post_json
    assert "payload = await response.json();" in post_json
    assert "catch (_error)" in post_json
    assert "Server returned a non-JSON response (HTTP ${response.status})" in post_json


def test_index_html_preserves_empty_final_transcript_text(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'const isTranscriptEvent = ["partial", "pending", "final"].includes(event.status);' in response.text
    assert "row.textContent = isTranscriptEvent ? event.text : (event.text || event.status);" in response.text


def test_index_html_revision_filter_does_not_drop_lifecycle_events(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'const tracksRevision = ["partial", "pending", "final"].includes(event.status);' in response.text
    assert "if (tracksRevision && event.revision <= currentRevision) return;" in response.text
    assert "if (tracksRevision) appState.revisions.set(key, event.revision);" in response.text


def test_index_html_only_rebuilds_lane_grid_when_lane_count_changes(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "const laneCountChanged = payload.lanes && payload.lanes !== appState.lanes;" in response.text
    assert "if (laneCountChanged) {" in response.text
    assert "renderLaneGrid();" in response.text


def test_index_html_health_only_rebuilds_lane_grid_when_lane_count_changes(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    health_start = response.text.index("async function loadHealth()")
    health_body = response.text[health_start : response.text.index("function connectWebSocket", health_start)]
    assert "const laneCountChanged = loadDraftFromPayload(payload);" in health_body
    assert "if (laneCountChanged || appState.laneEntries.size === 0) {" in health_body
    assert health_body.index("const laneCountChanged") < health_body.index("renderLaneGrid();")


def test_index_html_disables_controls_before_initial_health_response(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    bootstrap = response.text[response.text.index("pathPickerDialog.addEventListener"):]
    assert "renderStatus();\n      loadHealth().catch" in bootstrap


def test_index_html_polls_and_displays_global_zipformer_warmup_phase(tmp_path):
    app = create_app(
        config=replace(AppConfig(lanes=4), output_dir=tmp_path),
        manager=RecordingManager(),
        event_bus=FrontendEventBus(),
        recorder=RecordingRecorder(),
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "async function pollLoadPhase()" in response.text
    assert "window.setInterval(pollLoadPhase, 250)" in response.text
    assert 'payload.load_phase === "warming_zipformer"' in response.text
    assert 'setOperation("Warming up Zipformer...")' in response.text
    assert "stopLoadPhasePolling();" in response.text


def test_second_start_clears_prior_event_bus_results_before_recording_next_run(tmp_path):
    bus = FrontendEventBus()
    recorder = RecordingRecorder()
    manager = RecordingManager(state=PipelineState.LOADED, event_bus=bus)
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=manager,
        event_bus=bus,
        recorder=recorder,
    )

    with TestClient(app) as client:
        first_start = client.post("/api/recognition/start")
        assert first_start.status_code == 200

        bus.publish(
            ASREvent.asr_result(
                stream_id=0,
                sentence_id=0,
                revision=1,
                status=EventStatus.FINAL,
                text="run one",
                timestamp_ms=100,
            )
        )
        assert bus.snapshot_results()[(0, 0)]["text"] == "run one"

        stop_response = client.post("/api/recognition/stop")
        assert stop_response.status_code == 200

        second_start = client.post("/api/recognition/start")
        assert second_start.status_code == 200
        assert bus.snapshot_results() == {}

        bus.publish(
            ASREvent.asr_result(
                stream_id=0,
                sentence_id=0,
                revision=1,
                status=EventStatus.FINAL,
                text="run two",
                timestamp_ms=200,
            )
        )

    assert len(recorder.saved_runs) == 1
    assert recorder.saved_runs[0]["results"] == {
        (0, 0): {
            "revision": 1,
            "text": "run two",
            "status": EventStatus.FINAL,
            "result_type": ResultType.OFFLINE,
            "is_final": True,
        }
    }


def test_lane_reconfiguration_records_old_results_before_rebuilding_manager(tmp_path):
    bus = FrontendEventBus()
    recorder = RecordingRecorder()
    built_managers: list[RecordingManager] = []

    def manager_factory(config: AppConfig, event_bus: FrontendEventBus) -> RecordingManager:
        manager = RecordingManager(state=PipelineState.UNLOADED, event_bus=event_bus)
        built_managers.append(manager)
        return manager

    app = create_app(
        config=app_config_with_readable_model_files(tmp_path, lanes=2),
        manager_factory=manager_factory,
        event_bus=bus,
        recorder=recorder,
    )
    bus.publish(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=1,
            status=EventStatus.FINAL,
            text="old topology",
            timestamp_ms=100,
        )
    )

    with TestClient(app) as client:
        load_response = client.post("/api/models/load", json={"lanes": 4})
        assert load_response.status_code == 200
        assert len(recorder.saved_runs) == 1
        assert recorder.saved_runs[0]["config"].lanes == 2
        assert recorder.saved_runs[0]["results"][(0, 0)]["text"] == "old topology"
        assert bus.snapshot_results() == {}

    stale_new_config_saves = [
        run
        for run in recorder.saved_runs
        if run["config"].lanes == 4 and run["results"]
    ]
    assert stale_new_config_saves == []
    assert len(built_managers) == 2


def test_invalid_lane_reconfiguration_preserves_existing_results(tmp_path):
    bus = FrontendEventBus()
    recorder = RecordingRecorder()
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=RecordingManager(state=PipelineState.UNLOADED),
        event_bus=bus,
        recorder=recorder,
    )
    bus.publish(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=1,
            status=EventStatus.FINAL,
            text="still here",
            timestamp_ms=100,
        )
    )

    with TestClient(app) as client:
        response = client.post("/api/models/load", json={"lanes": 3})
        assert response.status_code == 422
        assert recorder.saved_runs == []
        assert bus.snapshot_results()[(0, 0)]["text"] == "still here"

    assert len(recorder.saved_runs) == 1
    assert recorder.saved_runs[0]["config"].lanes == 2
    assert recorder.saved_runs[0]["results"][(0, 0)]["text"] == "still here"


def test_failed_lane_reconfiguration_load_preserves_existing_state(tmp_path):
    bus = FrontendEventBus()
    recorder = RecordingRecorder()
    original_manager = RecordingManager(state=PipelineState.UNLOADED)
    built_configs: list[AppConfig] = []

    def manager_factory(config: AppConfig, event_bus: FrontendEventBus) -> RecordingManager:
        built_configs.append(config)
        return FailingLoadManager(state=PipelineState.UNLOADED, event_bus=event_bus)

    app = create_app(
        config=app_config_with_readable_model_files(tmp_path, lanes=2),
        manager=original_manager,
        manager_factory=manager_factory,
        event_bus=bus,
        recorder=recorder,
    )
    bus.publish(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=1,
            status=EventStatus.FINAL,
            text="previous lanes",
            timestamp_ms=100,
        )
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/models/load", json={"lanes": 4})
        health_response = client.get("/health")

        assert response.status_code == 500
        assert health_response.json()["lanes"] == 2
        assert health_response.json()["state"] == "unloaded"
        assert app.state.manager is original_manager
        assert recorder.saved_runs == []
        assert bus.snapshot_results()[(0, 0)]["text"] == "previous lanes"

    assert [config.lanes for config in built_configs] == [4]


def test_fake_web_app_reuses_loaded_models_between_runs_and_records_each_run(tmp_path):
    bus = FrontendEventBus()
    recorder = RecordingRecorder()
    app = create_app(
        config=app_config_with_readable_model_files(tmp_path, lanes=2),
        event_bus=bus,
        recorder=recorder,
        fake=True,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        load_response = client.post("/api/models/load")
        first_start_response = client.post("/api/recognition/start")
        stop_response = client.post("/api/recognition/stop")
        bus.publish(
            ASREvent.asr_result(
                stream_id=0,
                sentence_id=0,
                revision=11,
                status=EventStatus.FINAL,
                text="stale result",
                timestamp_ms=100,
            )
        )
        assert bus.snapshot_results()[(0, 0)]["text"] == "stale result"
        second_start_response = client.post("/api/recognition/start")
        reset_snapshot = bus.snapshot_results()
        second_stop_response = client.post("/api/recognition/stop")

    assert load_response.status_code == 200
    assert first_start_response.status_code == 200
    assert first_start_response.json()["state"] == "running"
    assert stop_response.status_code == 200
    assert stop_response.json()["state"] == "loaded"
    assert second_start_response.status_code == 200
    assert second_start_response.json()["state"] == "running"
    assert all(
        result["text"] != "stale result"
        for result in reset_snapshot.values()
    )
    assert second_stop_response.status_code == 200
    assert second_stop_response.json()["state"] == "loaded"
    process_runs = [run for run in recorder.saved_runs if run.get("process")]
    assert len(process_runs) == 2
    assert process_runs[0]["run_dir"] != process_runs[1]["run_dir"]


def test_start_failure_preserves_prior_event_bus_results(tmp_path):
    bus = FrontendEventBus()
    manager = FailingStartManager(state=PipelineState.LOADED)
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=manager,
        event_bus=bus,
        recorder=RecordingRecorder(),
    )
    bus.publish(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=1,
            status=EventStatus.FINAL,
            text="previous run",
            timestamp_ms=100,
        )
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        start_response = client.post("/api/recognition/start")

    assert start_response.status_code == 500
    assert bus.snapshot_results() == {
        (0, 0): {
            "revision": 1,
            "text": "previous run",
            "status": EventStatus.FINAL,
            "result_type": ResultType.OFFLINE,
            "is_final": True,
        }
    }


def test_shutdown_stops_active_manager_unloads_models_and_flushes_results(tmp_path):
    bus = FrontendEventBus()
    manager = RecordingManager(state=PipelineState.RUNNING)
    recorder = RecordingRecorder()
    app = create_app(
        config=replace(AppConfig(lanes=2), output_dir=tmp_path),
        manager=manager,
        event_bus=bus,
        recorder=recorder,
    )

    bus.publish(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=1,
            status=EventStatus.PARTIAL,
            text="partial text",
            timestamp_ms=100,
        )
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert manager.calls == ["stop_recognition", "unload_models"]
    assert len(recorder.saved_runs) == 1
    assert recorder.saved_runs[0]["config"].lanes == 2
    assert recorder.saved_runs[0]["results"] == {
        (0, 0): {
            "revision": 1,
            "text": "partial text",
            "status": EventStatus.PARTIAL,
            "result_type": "online",
            "is_final": False,
        }
    }


def test_main_runs_uvicorn_for_web_mode(monkeypatch, tmp_path):
    create_app_calls: list[AppConfig] = []
    uvicorn_calls: list[tuple[object, str, int]] = []

    def fake_create_app(*, config: AppConfig, fake: bool = False):
        create_app_calls.append(config)
        assert fake is False
        return object()

    def fake_run(app, *, host: str, port: int) -> None:
        uvicorn_calls.append((app, host, port))

    monkeypatch.setattr("multi_stream_asr.main.create_app", fake_create_app)
    monkeypatch.setattr("multi_stream_asr.main.uvicorn.run", fake_run)

    exit_code = main(
        [
            "--lanes",
            "8",
            "--input-dir",
            str(tmp_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
        ]
    )

    assert exit_code == 0
    assert create_app_calls[0].lanes == 8
    assert create_app_calls[0].input_dir == tmp_path.resolve()
    assert uvicorn_calls == [(uvicorn_calls[0][0], "0.0.0.0", 9001)]
