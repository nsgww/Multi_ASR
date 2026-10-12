from __future__ import annotations

import time


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)
