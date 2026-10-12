from __future__ import annotations

from collections import deque
from threading import Condition

from multi_stream_asr.models import AudioFrame


class OnlineFrameQueue:
    def __init__(self, maxsize: int = 10) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.maxsize = maxsize
        self._frames: deque[AudioFrame] = deque()
        self.dropped_frames = 0
        self.backlogged_frames = 0
        self._condition = Condition()

    def put(self, frame: AudioFrame) -> bool:
        with self._condition:
            if len(self._frames) < self.maxsize:
                if self._frames:
                    self.backlogged_frames += 1
                self._frames.append(frame)
                self._condition.notify()
                return True

            if any(queued_frame.is_finish for queued_frame in self._frames):
                if frame.is_finish:
                    for index, queued_frame in enumerate(self._frames):
                        if queued_frame.is_finish:
                            self._frames[index] = frame
                            break
                    self.dropped_frames += 1
                    self._condition.notify()
                    return False
                self.dropped_frames += 1
                return False

            if frame.is_finish:
                self._frames.popleft()
                self._frames.append(frame)
                self.dropped_frames += 1
                self._condition.notify()
                return False

            self._frames.popleft()
            self._frames.append(frame)
            self.dropped_frames += 1
            self._condition.notify()
            return False

    def get(self) -> AudioFrame | None:
        with self._condition:
            if not self._frames:
                return None
            return self._frames.popleft()

    def wait_get(self, timeout: float | None = None) -> AudioFrame | None:
        with self._condition:
            if not self._frames:
                self._condition.wait(timeout=timeout)
            if not self._frames:
                return None
            return self._frames.popleft()

    def clear(self) -> None:
        with self._condition:
            self._frames.clear()
            self._condition.notify_all()

    def empty(self) -> bool:
        with self._condition:
            return not self._frames

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return len(self._frames)
