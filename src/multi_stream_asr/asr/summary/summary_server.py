from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TextIO
from urllib.error import URLError
from urllib.request import urlopen

from multi_stream_asr.asr.summary.config import SummaryConfig
from multi_stream_asr.utils.process import run_best_effort

_logger = logging.getLogger(__name__)


def is_server_healthy(url: str, timeout: float) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False
    return payload.get("status") == "ok"


class SummaryServerManager:
    """Lifecycle manager for the standalone llama-server used for summaries."""

    def __init__(
        self,
        config: SummaryConfig,
        *,
        health_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 1.0,
        stop_timeout_seconds: float = 5.0,
    ) -> None:
        self._config = config
        self._health_timeout_seconds = health_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_timeout_seconds = stop_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: TextIO | None = None

    @property
    def base_url(self) -> str:
        return self._config.base_url_or_default

    @property
    def exit_code(self) -> int | None:
        return None if self._process is None else self._process.poll()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("summary llama-server is already started")
        if not self._config.auto_start:
            raise RuntimeError("summary auto_start is disabled")
        if self._config.provider != "llama":
            raise RuntimeError(
                f"summary server can only be auto-started with provider=llama, got {self._config.provider}"
            )

        model_path = Path(self._config.model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"summary model not found: {model_path}")

        # Spacemit X100 performance cores are 10-15 (arch a064); do not mix
        # with efficiency cores (0-9, arch 5064) and keep thread count <= 6.
        num_threads = min(max(self._config.num_threads, 1), 6)
        performance_cores = "10,11,12,13,14,15"

        command = [
            self._config.llama_server_bin,
            "-m",
            str(model_path),
            "--host",
            self._config.host,
            "--port",
            str(self._config.port),
            "-c",
            str(self._config.ctx_size),
            "-t",
            str(num_threads),
            "-ngl",
            str(self._config.n_gpu_layers),
        ]

        log_dir = Path("output/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = (log_dir / "summary-server.log").open(
            "w", encoding="utf-8"
        )
        env = os.environ.copy()
        env["SPACEMIT_PERFER_CORE_ID"] = performance_cores

        try:
            self._process = subprocess.Popen(
                command,
                env=env,
                stdout=self._log_handle,
                stderr=self._log_handle,
                text=True,
            )
            self._wait_for_health()
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
            self._process = None
            run_best_effort(["spacemit-tcm-smi", "-c"])

    def _wait_for_health(self) -> None:
        url = f"{self.base_url}/health"
        deadline = time.monotonic() + self._health_timeout_seconds
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"summary llama-server exited before health check: {self._process.poll()}"
                )
            if is_server_healthy(url, timeout=min(self._poll_interval_seconds, 2.0)):
                _logger.info("summary llama-server is healthy at %s", self.base_url)
                return
            time.sleep(self._poll_interval_seconds)
        raise RuntimeError(
            f"summary llama-server health timed out on port {self._config.port}"
        )
