from __future__ import annotations

import math
from collections import deque

import numpy as np

from multi_stream_asr.models import AudioFrame


class SegmentCache:
    def __init__(self, sample_rate: int, capacity_seconds: float) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if capacity_seconds <= 0:
            raise ValueError("capacity_seconds must be positive")
        self.sample_rate = sample_rate
        self.capacity_seconds = capacity_seconds
        self.capacity_samples = int(sample_rate * capacity_seconds)
        self._chunks: deque[np.ndarray] = deque()
        self._samples = 0
        self._start_sample = 0
        self._end_sample = 0

    @property
    def start_time_s(self) -> float:
        return self._start_sample / self.sample_rate

    @property
    def end_time_s(self) -> float:
        return self._end_sample / self.sample_rate

    def put(self, frame: AudioFrame) -> None:
        if frame.is_finish:
            return
        chunk = np.asarray(frame.pcm, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        self._chunks.append(chunk)
        size = int(chunk.size)
        self._samples += size
        self._end_sample += size
        self._trim_left()

    def _trim_left(self) -> None:
        while self._samples > self.capacity_samples and self._chunks:
            overflow = self._samples - self.capacity_samples
            first = self._chunks[0]
            if overflow >= first.size:
                removed = int(first.size)
                self._chunks.popleft()
                self._samples -= removed
                self._start_sample += removed
                continue
            self._chunks[0] = first[overflow:]
            self._samples -= int(overflow)
            self._start_sample += int(overflow)
            break

    def audio(self) -> np.ndarray:
        if not self._chunks:
            return np.asarray([], dtype=np.float32)
        if len(self._chunks) == 1:
            return np.asarray(self._chunks[0], dtype=np.float32)
        return np.concatenate(tuple(self._chunks)).astype(np.float32, copy=False)

    def cut(self, stream_time_s: float) -> np.ndarray:
        if not math.isfinite(stream_time_s):
            raise ValueError("stream_time_s must be finite")
        if stream_time_s <= self.start_time_s:
            cut_sample = self._start_sample
        elif stream_time_s >= self.end_time_s:
            cut_sample = self._end_sample
        else:
            cut_sample = int(round(stream_time_s * self.sample_rate))
        head_samples = cut_sample - self._start_sample
        full = self.audio()
        head = full[:head_samples]
        tail = full[head_samples:]
        self._chunks.clear()
        if tail.size:
            self._chunks.append(tail)
        self._samples = int(tail.size)
        self._start_sample = cut_sample
        return head

    def reset(self) -> None:
        self._chunks.clear()
        self._samples = 0
        self._start_sample = self._end_sample
