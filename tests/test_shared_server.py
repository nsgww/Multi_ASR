from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import pytest

from multi_stream_asr.asr.offline.shared_server import (
    SharedQwenServerManager,
    build_launch_command,
)
from multi_stream_asr.config import AppConfig, ModelConfig, SharedQwenConfig


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
            raise subprocess.TimeoutExpired("llama-server", timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def make_valid_config(tmp_path: Path) -> AppConfig:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in (
        "Qwen3-ASR-0.6B-text-q40.gguf",
        "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
        "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
        "config.json",
    ):
        (model_dir / name).write_text("model")
    return AppConfig(models=ModelConfig(offline_model_dir=model_dir))


def configure_temp_paths(manager: SharedQwenServerManager, tmp_path: Path) -> None:
    manager._config_dir = tmp_path / "generated"
    manager._pid_path = tmp_path / "service.pid"
    manager._log_path = tmp_path / "service.log"


def test_launch_command_uses_llama_server_smt_flags(tmp_path: Path):
    config = AppConfig(
        models=ModelConfig(
            offline_model_id="qwen3-asr-0.6b-q40",
            offline_model_dir=tmp_path / "models",
        )
    )
    server = tmp_path / "bin" / "llama-server"

    command = build_launch_command(config, server)

    assert command[0] == str(server)
    assert command[command.index("-m") + 1] == str(
        tmp_path / "models" / "Qwen3-ASR-0.6B-text-q40.gguf"
    )
    assert command[command.index("--media-backend") + 1] == "smt"
    assert command[command.index("--smt-config-dir") + 1] == str(tmp_path / "models")
    assert "--smt-multi-asr" in command
    assert command[command.index("-np") + 1] == "4"
    assert command[command.index("-c") + 1] == "8192"
    assert command[command.index("-t") + 1] == "6"
    assert command[command.index("-tb") + 1] == "6"
    assert "--warmup" in command


def test_launch_command_reflects_config_overrides(tmp_path: Path):
    config = AppConfig(
        models=ModelConfig(offline_model_dir=tmp_path / "models"),
        shared_qwen=SharedQwenConfig(decoder_threads=4, n_parallel=2, n_ctx=4096),
    )

    command = build_launch_command(config, tmp_path / "llama-server")

    assert command[command.index("-np") + 1] == "2"
    assert command[command.index("-c") + 1] == "4096"
    assert command[command.index("-t") + 1] == "4"


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
    server = tmp_path / "llama-server"
    server.write_text("binary")
    pid_path = tmp_path / "service.pid"
    pid_path.write_text("1234\n")
    command = build_launch_command(config, server)
    signals: list[signal.Signals] = []
    monkeypatch.setattr(manager, "_pid_path", pid_path)
    monkeypatch.setattr(manager, "_read_cmdline", lambda pid: command)
    monkeypatch.setattr(manager, "_pid_exists", lambda pid: not signals)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.resolve_llama_server",
        lambda: server,
    )
    monkeypatch.setattr("os.kill", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.run_best_effort",
        lambda command: subprocess.CompletedProcess(command, 0, "", ""),
    )

    manager.cleanup_before_start()

    assert signals == [signal.SIGTERM]
    assert not pid_path.exists()


def test_start_waits_health_and_exposes_liveness(tmp_path, monkeypatch):
    config = make_valid_config(tmp_path)
    process = FakeProcess()
    manager = SharedQwenServerManager(config)
    configure_temp_paths(manager, tmp_path)
    server = tmp_path / "llama-server"
    server.write_text("binary")
    launched = []
    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.resolve_llama_server",
        lambda: server,
    )
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
    assert launched[0][0][0] == str(server)
    manager.stop()
    assert process.terminate_calls == 1
    assert not manager.is_alive()
    assert not (tmp_path / "service.pid").exists()


def test_stop_escalates_to_kill(tmp_path, monkeypatch):
    config = make_valid_config(tmp_path)
    process = FakeProcess(timeout_until_kill=True)
    manager = SharedQwenServerManager(config, stop_timeout_seconds=0.01)
    configure_temp_paths(manager, tmp_path)
    server = tmp_path / "llama-server"
    server.write_text("binary")
    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.resolve_llama_server",
        lambda: server,
    )
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
    server = tmp_path / "llama-server"
    server.write_text("binary")
    monkeypatch.setattr(manager, "cleanup_before_start", lambda: None)
    monkeypatch.setattr(
        "multi_stream_asr.asr.offline.shared_server.resolve_llama_server",
        lambda: server,
    )
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
