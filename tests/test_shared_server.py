from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import pytest

from multi_stream_asr.asr.offline.shared_server import (
    SharedQwenServerManager,
    build_launch_command,
    build_shared_config,
)
from multi_stream_asr.config import AppConfig, ModelConfig, PROJECT_ROOT, SharedQwenConfig


class FakeProcess:
    def __init__(self, pid: int = 4321, *, timeout_until_kill: bool = False) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.timeout_until_kill = timeout_until_kill
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.timeout_until_kill and self.kill_calls == 0:
            raise subprocess.TimeoutExpired("llama-multi-asr", timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def make_valid_config(tmp_path: Path) -> AppConfig:
    model_dir = tmp_path / "models"
    runtime_dir = tmp_path / "runtime"
    (runtime_dir / "bin").mkdir(parents=True)
    (runtime_dir / "lib").mkdir()
    (runtime_dir / "ort_lib").mkdir()
    (runtime_dir / "bin" / "llama-multi-asr").write_text("binary")
    model_dir.mkdir()
    for name in (
        "Qwen3-ASR-0.6B-text-q40.gguf",
        "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
        "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
    ):
        (model_dir / name).write_text("model")
    return AppConfig(
        models=ModelConfig(offline_model_dir=model_dir),
        shared_qwen=SharedQwenConfig(runtime_dir=runtime_dir),
    )


def configure_temp_paths(manager: SharedQwenServerManager, tmp_path: Path) -> None:
    manager._config_dir = tmp_path / "generated"
    manager._pid_path = tmp_path / "service.pid"
    manager._log_path = tmp_path / "service.log"


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


def test_default_launch_command_uses_vendored_runtime():
    command = build_launch_command(AppConfig(), Path("/tmp/shared-qwen"))

    assert command[0] == str(PROJECT_ROOT / "multi-asr-runtime/bin/llama-multi-asr")


def test_scoped_cleanup_ignores_pid_with_nonmatching_cmdline(tmp_path, monkeypatch):
    manager = SharedQwenServerManager(AppConfig())
    pid_path = tmp_path / "service.pid"
    pid_path.write_text("1234\n")
    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(manager, "_pid_path", pid_path)
    monkeypatch.setattr(manager, "_read_cmdline", lambda pid: ["other-service"])
    monkeypatch.setattr("os.kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.run_best_effort",
        lambda command: subprocess.CompletedProcess(command, 0, "", ""),
    )

    manager.cleanup_before_start()

    assert killed == []
    assert not pid_path.exists()


def test_scoped_cleanup_terminates_matching_pid(tmp_path, monkeypatch):
    config = make_valid_config(tmp_path)
    manager = SharedQwenServerManager(config, stop_timeout_seconds=0.01)
    pid_path = tmp_path / "service.pid"
    pid_path.write_text("1234\n")
    command = build_launch_command(config, tmp_path / "generated")
    signals: list[signal.Signals] = []
    monkeypatch.setattr(manager, "_pid_path", pid_path)
    monkeypatch.setattr(manager, "_read_cmdline", lambda pid: command)
    monkeypatch.setattr(manager, "_pid_exists", lambda pid: not signals)
    monkeypatch.setattr("os.kill", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.run_best_effort",
        lambda command: subprocess.CompletedProcess(command, 0, "", ""),
    )

    manager.cleanup_before_start()

    assert signals == [signal.SIGTERM]
    assert not pid_path.exists()


def test_start_writes_config_waits_health_and_exposes_liveness(tmp_path, monkeypatch):
    config = make_valid_config(tmp_path)
    process = FakeProcess()
    manager = SharedQwenServerManager(config)
    configure_temp_paths(manager, tmp_path)
    launched = []
    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.subprocess.Popen",
        lambda command, **kwargs: launched.append((command, kwargs)) or process,
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.is_server_healthy",
        lambda url, timeout: True,
    )

    spec = manager.start()

    assert spec.port == 8090
    assert manager.is_alive()
    assert manager.exit_code is None
    assert json.loads((tmp_path / "generated/config.json").read_text())
    assert str(config.shared_qwen.runtime_dir / "lib") in launched[0][1]["env"][
        "LD_LIBRARY_PATH"
    ]
    assert launched[0][1]["env"]["SPACEMIT_PERFER_CORE_ID"] == "10,11,12,13,14,15"
    manager.stop()
    assert process.terminate_calls == 1
    assert not manager.is_alive()
    assert not (tmp_path / "service.pid").exists()


def test_stop_escalates_to_kill(tmp_path, monkeypatch):
    config = make_valid_config(tmp_path)
    process = FakeProcess(timeout_until_kill=True)
    manager = SharedQwenServerManager(config, stop_timeout_seconds=0.01)
    configure_temp_paths(manager, tmp_path)
    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.subprocess.Popen",
        lambda command, **kwargs: process,
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.is_server_healthy",
        lambda url, timeout: True,
    )

    manager.start()
    manager.stop()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1


def test_start_stops_process_on_health_timeout(tmp_path, monkeypatch):
    config = make_valid_config(tmp_path)
    process = FakeProcess()
    manager = SharedQwenServerManager(
        config, health_timeout_seconds=0.01, poll_interval_seconds=0.001
    )
    configure_temp_paths(manager, tmp_path)
    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.subprocess.Popen",
        lambda command, **kwargs: process,
    )
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.is_server_healthy",
        lambda url, timeout: False,
    )

    with pytest.raises(RuntimeError, match="health timed out"):
        manager.start()

    assert process.terminate_calls == 1
