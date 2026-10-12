from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from multi_stream_asr.models import AudioFrame


class AudioSource(Protocol):
    async def prepare(self) -> None: ...

    async def frames(self) -> AsyncIterator[AudioFrame]: ...

    async def stop(self) -> None: ...
