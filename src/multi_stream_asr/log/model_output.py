from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any, TextIO


@dataclass(frozen=True)
class ModelOutputRecord:
    model: str
    stream_id: int | None
    sentence_id: int | None
    status: str
    text: str
    audio_seconds: float | None = None
    elapsed_ms: float | None = None
    rtf: float | None = None
    raw_response: dict[str, Any] | None = None

    @property
    def text_preview(self) -> str:
        return self.text[:120]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["text_preview"] = self.text_preview
        return payload


class ModelOutputRecorder:
    def __init__(self, *, stream: TextIO | None = None, emit_stdout: bool = True) -> None:
        self._stream = stream or sys.stdout
        self._emit_stdout = emit_stdout
        self._records: list[ModelOutputRecord] = []

    def record(self, record: ModelOutputRecord) -> None:
        self._records.append(record)
        if not self._emit_stdout:
            return

        audio = "-" if record.audio_seconds is None else f"{record.audio_seconds:.3f}"
        elapsed = "-" if record.elapsed_ms is None else f"{record.elapsed_ms:.1f}"
        rtf = "-" if record.rtf is None else f"{record.rtf:.3f}"
        print(
            "[model-output] "
            f"model={record.model} "
            f"lane={record.stream_id} "
            f"sentence={record.sentence_id} "
            f"status={record.status} "
            f"text_len={len(record.text)} "
            f"audio_s={audio} "
            f"elapsed_ms={elapsed} "
            f"rtf={rtf} "
            f"text={record.text_preview}",
            file=self._stream,
            flush=True,
        )

    def as_dicts(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self._records]
