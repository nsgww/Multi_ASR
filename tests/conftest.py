from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from multi_stream_asr.models import AudioFrame, OnlineRecognition

# 这两个是板端手工排查脚本，模块级就要读 /tmp 和 /app/lg-store 的真实文件，
# 被 pytest 收集会直接 FileNotFoundError 并中断整轮回归。保留文件、跳过收集。
collect_ignore = [
    "gemma_old_template_test.py",
    "sensevoice_ab_test.py",
]


def make_frame(
    *,
    stream_id: int = 0,
    frame_index: int,
    samples: list[float],
    timestamp_ms: int,
) -> AudioFrame:
    return AudioFrame(
        stream_id=stream_id,
        frame_index=frame_index,
        pcm=np.asarray(samples, dtype=np.float32),
        timestamp_ms=timestamp_ms,
    )


def make_online(
    *,
    stream_id: int = 0,
    sentence_id: int = 0,
    revision: int = 0,
    text: str,
    timestamp_ms: int,
    is_endpoint: bool = False,
    last_token_stream_s: float | None = None,
) -> OnlineRecognition:
    return OnlineRecognition(
        stream_id=stream_id,
        sentence_id=sentence_id,
        revision=revision,
        text=text,
        timestamp_ms=timestamp_ms,
        is_endpoint=is_endpoint,
        last_token_stream_s=last_token_stream_s,
    )


class FakeAudioSource:
    def __init__(self, frames: list[AudioFrame], *, pause_between_frames: bool = False) -> None:
        self._frames = list(frames)
        self.pause_between_frames = pause_between_frames
        self.prepare_calls = 0
        self.stop_calls = 0

    async def prepare(self) -> None:
        self.prepare_calls += 1

    async def frames(self) -> AsyncIterator[AudioFrame]:
        for frame in self._frames:
            if self.pause_between_frames:
                await asyncio.sleep(0.001)
            yield frame

    async def stop(self) -> None:
        self.stop_calls += 1


class FakeOnlineRecognizer:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.start_run_count = 0
        self.reset_count = 0
        self.accepted_frames: list[AudioFrame] = []

    async def start_run(self):
        self.start_run_count += 1

    async def accept(self, frame):
        self.accepted_frames.append(frame)
        if frame.is_finish:
            return None
        return self.outputs.pop(0) if self.outputs else None

    async def reset(self):
        self.reset_count += 1


class FakeOfflineRecognizer:
    def __init__(self, text="offline final", *, release: asyncio.Event | None = None):
        self.text = text
        self.release = release
        self.requests = []

    async def recognize(self, pcm, prompt):
        self.requests.append((pcm.copy(), prompt))
        if self.release is not None:
            await self.release.wait()
        from multi_stream_asr.asr.offline.base import OfflineRecognitionResult

        return OfflineRecognitionResult(
            text=self.text,
            elapsed_ms=400,
            audio_seconds=1.0,
            rtf=0.4,
        )

    async def close(self):
        return None
