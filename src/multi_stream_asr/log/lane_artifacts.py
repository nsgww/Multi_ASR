from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import TextIO

from multi_stream_asr.log.model_output import ModelOutputRecord
from multi_stream_asr.models import ASREvent, EventStatus
from multi_stream_asr.utils.paths import ensure_dir


def compact_model_output_record(record: ModelOutputRecord) -> dict[str, object]:
    return {
        "model": record.model,
        "stream_id": record.stream_id,
        "sentence_id": record.sentence_id,
        "audio_seconds": record.audio_seconds,
        "rtf": record.rtf,
        "text": record.text,
    }


class DetailJsonlWriter:
    def __init__(
        self,
        path: Path,
        *,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.path = path
        self._monotonic = monotonic
        self._started_at = monotonic()
        ensure_dir(path.parent)
        self._file: TextIO | None = path.open("w", encoding="utf-8")
        self._lock = Lock()

    def record(self, fields: dict[str, object]) -> None:
        t_ms = (self._monotonic() - self._started_at) * 1000.0
        payload = json.dumps({"t_ms": t_ms, **fields}, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if self._file is None:
                return
            self._file.write(payload)
            self._file.write("\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is None:
                return
            self._file.close()
            self._file = None


class LaneResultWriter:
    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        ensure_dir(path.parent)
        self._file: TextIO = path.open("w", encoding="utf-8")

    def record_event(self, event: ASREvent) -> None:
        if not self.enabled or event.status is not EventStatus.FINAL:
            return
        self._file.write(f"{event.text}\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class LaneModelOutputJsonlRecorder:
    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        ensure_dir(path.parent)
        self._file: TextIO | None = path.open("w", encoding="utf-8")
        self._lock = Lock()

    def record(self, record: ModelOutputRecord) -> None:
        if not self.enabled:
            return
        payload = json.dumps(compact_model_output_record(record), ensure_ascii=False, sort_keys=True)
        with self._lock:
            if self._file is None:
                return
            self._file.write(payload)
            self._file.write("\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is None:
                return
            self._file.close()
            self._file = None
