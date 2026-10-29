from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from urllib.error import URLError
from urllib.request import urlopen

from multi_stream_asr.config import AppConfig
from multi_stream_asr.utils.process import run_best_effort


SHARED_CONFIG_DIR = Path("/tmp/multi-stream-asr/shared-qwen")
SHARED_PID_PATH = SHARED_CONFIG_DIR / "llama-server.pid"
SHARED_LOG_PATH = SHARED_CONFIG_DIR / "llama-server.log"

# llama.cpp 0.2.1 把独立 `ggml-spacemit` 后端改成了 opt-in：registry 只有在
# 该变量恰好为 "1" 时才报告设备。没有它 `--device SPACEMIT0` 会因参数解析
# 失败（"invalid device: SPACEMIT0"）导致 llama-server 起不来。0.2.0 没有
# 这个门控，设置该变量无害，因此这里始终导出并覆盖继承值。
ENV_SPACEMIT_BACKEND = "GGML_SPACEMIT_BACKEND"
SPACEMIT_BACKEND_ENABLED = "1"

SUPPORT_LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
)


@dataclass(frozen=True)
class SharedQwenServerSpec:
    port: int
    config_dir: Path
    log_path: Path
    pid_path: Path
    env: dict[str, str]
    command: list[str]


def resolve_llama_server() -> Path:
    """定位系统 llama-server（SpacemiT 定制版，/usr/bin/llama-server）。"""
    candidate = shutil.which("llama-server")
    if candidate is None:
        raise FileNotFoundError(
            "system llama-server is unavailable; install llama.cpp and verify PATH"
        )
    server_path = Path(candidate).resolve(strict=True)
    if not server_path.is_file() or not os.access(server_path, os.X_OK):
        raise FileNotFoundError(f"llama-server is not executable: {server_path}")
    return server_path


def build_shared_config(config: AppConfig) -> dict[str, object]:
    """生成 --smt-config-dir 指向的 config.json（含 encoder 绑核）。

    encoder 的 intra-op 线程数与绑核（spacemit_ep_intra_thread_num /
    spacemit_ep_intra_thread_affinity）在这里下发，与 decoder 核心集
    （SPACEMIT_PERFER_CORE_ID）形成 2+6 不相交拆分。
    """
    model = config.models.offline_model
    shared = config.shared_qwen
    return {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "support_languages": list(SUPPORT_LANGUAGES),
        "audio_model": {
            "frontend_model_path": str(model.model_dir / model.frontend_filename),
            "backend_model_path": str(model.model_dir / model.backend_filename),
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
            "d_model": model.d_model,
            "hidden_size": model.hidden_size,
            "output_dim": model.output_dim,
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
            "model_path": str(model.model_dir / model.text_model_filename),
            "context_size": 65536,
            "logical_batch_size": 2048,
            "predict_token": 256,
            "vocab_size": 151936,
            "hidden_size": model.hidden_size,
        },
        "prompt": {
            "system_prefix_token": "<|im_start|>system",
            "user_prefix_token": "<|im_start|>user",
            "assistant_prefix_token": "<|im_start|>assistant",
            "message_suffix_token": "<|im_end|>",
            "asr_text_tag": "<asr_text>",
        },
    }


def build_launch_command(
    config: AppConfig, config_dir: Path, server_path: Path
) -> list[str]:
    """llama-server（--media-backend smt）启动命令（资源分配对齐性能调优线）。

    -np 跟随 decoder 核数（SharedQwenConfig.n_parallel），-b/-ub 批大小、
    --threads-http 并发 HTTP 线程、--device SPACEMIT0 强制落 SpacemiT 后端。
    --smt-config-dir 指向自生成的 config.json（含 encoder 绑核）；
    --smt-multi-asr 必须打开，否则输出会把 prompt 回声出来并截断。
    """
    shared = config.shared_qwen
    model = config.models.offline_model
    return [
        str(server_path),
        "-m",
        str(model.model_dir / model.text_model_filename),
        "--media-backend",
        "smt",
        "--smt-config-dir",
        str(config_dir),
        "--smt-multi-asr",
        "--host",
        shared.host,
        "--port",
        str(shared.port),
        "-np",
        str(shared.n_parallel),
        "-c",
        str(shared.n_ctx),
        "-b",
        str(shared.batch_size),
        "-ub",
        str(shared.ubatch_size),
        "-t",
        str(shared.decoder_threads),
        "-tb",
        str(shared.decoder_batch_threads),
        "--threads-http",
        str(shared.http_threads),
        "--cache-reuse",
        "0",
        "--warmup",
        "--device",
        shared.device,
        "--no-jinja",
        "--chat-template",
        "chatml",
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
        self._config_dir = SHARED_CONFIG_DIR
        self._pid_path = SHARED_PID_PATH
        self._log_path = SHARED_LOG_PATH
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: TextIO | None = None

    @property
    def exit_code(self) -> int | None:
        return None if self._process is None else self._process.poll()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def cleanup_before_start(self) -> None:
        pid = self._read_pid()
        if pid is not None:
            command = self._read_cmdline(pid)
            if self._matches_owned_command(command):
                self._signal_and_wait(pid)
        self._pid_path.unlink(missing_ok=True)
        run_best_effort(["spacemit-tcm-smi", "-c"])

    def start(self) -> SharedQwenServerSpec:
        if self._process is not None:
            raise RuntimeError("shared qwen server is already started")
        self.cleanup_before_start()
        spec = self._build_spec()
        self._validate_files(spec)
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()
        self._log_handle = self._log_path.open("w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                spec.command,
                env=spec.env,
                stdout=self._log_handle,
                stderr=self._log_handle,
                text=True,
                # 纠错服务对延迟不敏感：降调度优先级，把 CPU 让给流式识别主链路，
                # 避免 Qwen 编码/解码高峰期抢占导致帧缓冲溢出、整句语音被丢弃
                preexec_fn=lambda: os.nice(int(os.getenv("QWEN_NICE", "10"))),
            )
            self._pid_path.write_text(f"{self._process.pid}\n", encoding="utf-8")
            self._wait_for_health()
            return spec
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        process = self._process
        try:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=self._stop_timeout_seconds)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                        process.wait(timeout=self._stop_timeout_seconds)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        pass
        finally:
            if self._log_handle is not None:
                self._log_handle.close()
            self._log_handle = None
            self._pid_path.unlink(missing_ok=True)

    def _build_spec(self) -> SharedQwenServerSpec:
        shared = self._config.shared_qwen
        env = os.environ.copy()
        # decoder 核心集：ggml 设备只在这份核心列表上调度 worker 线程
        env["SPACEMIT_PERFER_CORE_ID"] = shared.decoder_cores
        # 0.2.1+ 用该变量门控独立后端；覆盖继承值，避免环境里残留
        # GGML_SPACEMIT_BACKEND=0 把 decoder 静默压回 CPU 后端、A100 集群闲置
        env[ENV_SPACEMIT_BACKEND] = SPACEMIT_BACKEND_ENABLED
        return SharedQwenServerSpec(
            port=shared.port,
            config_dir=self._config_dir,
            log_path=self._log_path,
            pid_path=self._pid_path,
            env=env,
            command=build_launch_command(
                self._config, self._config_dir, resolve_llama_server()
            ),
        )

    def _write_config(self) -> None:
        (self._config_dir / "config.json").write_text(
            json.dumps(build_shared_config(self._config), indent=2),
            encoding="utf-8",
        )

    def _validate_files(self, spec: SharedQwenServerSpec) -> None:
        model = self._config.models.offline_model
        required = [
            Path(spec.command[0]),
            *(model.model_dir / filename for filename in model.required_filenames),
        ]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"required shared qwen file not found: {path}")

    def _wait_for_health(self) -> None:
        shared = self._config.shared_qwen
        url = f"http://{shared.host}:{shared.port}/health"
        deadline = time.monotonic() + self._health_timeout_seconds
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited before health check: {self._process.poll()}"
                )
            if is_server_healthy(url, timeout=min(self._poll_interval_seconds, 2.0)):
                return
            time.sleep(self._poll_interval_seconds)
        raise RuntimeError(f"llama-server health timed out on port {shared.port}")

    def _read_pid(self) -> int | None:
        try:
            value = int(self._pid_path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, OSError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _read_cmdline(pid: int) -> list[str]:
        try:
            raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        except OSError:
            return []
        return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        return (Path("/proc") / str(pid)).exists()

    def _matches_owned_command(self, command: list[str]) -> bool:
        if not command:
            return False
        try:
            expected_binary = str(resolve_llama_server())
        except FileNotFoundError:
            return False
        try:
            actual_binary = str(Path(command[0]).resolve())
            port_index = command.index("--port")
        except (OSError, ValueError):
            return False
        return (
            actual_binary == expected_binary
            and port_index + 1 < len(command)
            and command[port_index + 1] == str(self._config.shared_qwen.port)
        )

    def _signal_and_wait(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + self._stop_timeout_seconds
        while time.monotonic() < deadline:
            if not self._pid_exists(pid):
                return
            time.sleep(min(self._poll_interval_seconds, 0.05))
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
