from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class OfflineRecognitionResult:
    text: str
    elapsed_ms: float
    audio_seconds: float
    rtf: float
    raw_response: dict[str, Any] | None = None
    timings: dict[str, float] | None = None


class OfflineRecognizer(Protocol):
    async def recognize(self, pcm: np.ndarray, prompt: str) -> OfflineRecognitionResult: ...

    async def close(self) -> None: ...
