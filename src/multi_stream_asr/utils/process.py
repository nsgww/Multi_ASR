from __future__ import annotations

import subprocess


def run_best_effort(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)
