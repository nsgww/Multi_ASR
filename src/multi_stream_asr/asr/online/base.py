from __future__ import annotations

from typing import Protocol

from multi_stream_asr.models import AudioFrame, OnlineRecognition


class OnlineRecognizer(Protocol):
    async def start_run(self) -> None: ...

    async def accept(self, frame: AudioFrame) -> OnlineRecognition | None: ...

    async def reset(self) -> None: ...
