from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from multi_stream_asr.asr.offline.sense_voice import SenseVoiceRecognizer
from multi_stream_asr.asr.summary.config import SummaryConfig
from multi_stream_asr.asr.summary.summary_client import SummaryClient
from multi_stream_asr.asr.text_filter import SensitiveWordFilter
from multi_stream_asr.config import AppConfig

_logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Fixed-window offline splitting (no VAD), aligned with sm-sdk pseudo-stream max window.
CHUNK_SECONDS = 5.0


def _read_wav(file: UploadFile) -> tuple[np.ndarray, int]:
    try:
        content = file.file.read()
        wav = wave.open(io.BytesIO(content), "rb")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid WAV file: {exc}") from exc

    with wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        raw = wav.readframes(n_frames)

    if sample_width == 2:
        pcm = np.frombuffer(raw, dtype="<i2")
    elif sample_width == 4:
        pcm = np.frombuffer(raw, dtype="<i4")
    else:
        raise HTTPException(status_code=422, detail="unsupported sample width")

    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(pcm.dtype)

    if sample_rate != SAMPLE_RATE:
        pcm = _simple_resample(pcm, sample_rate, SAMPLE_RATE)

    return pcm.astype(np.int16), SAMPLE_RATE


def _simple_resample(pcm: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
    if input_rate == output_rate:
        return pcm
    ratio = output_rate / input_rate
    target_size = int(np.ceil(pcm.size * ratio))
    indices = np.arange(target_size) / ratio
    floor_idx = np.floor(indices).astype(np.int64)
    ceil_idx = np.minimum(floor_idx + 1, pcm.size - 1)
    frac = indices - floor_idx
    return (pcm[floor_idx] * (1 - frac) + pcm[ceil_idx] * frac).astype(pcm.dtype)


def _transcript_filename(original_filename: str, lane_id: int, started_at: float) -> str:
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(started_at))
    return f"offline-lane{lane_id}-{stamp}.txt"


def _chunk_pcm(
    pcm: np.ndarray,
    *,
    chunk_seconds: float = CHUNK_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> list[np.ndarray]:
    """Split PCM into fixed non-overlapping windows (no VAD)."""
    pcm = np.asarray(pcm)
    if pcm.size == 0:
        return []
    chunk_samples = max(int(chunk_seconds * sample_rate), 1)
    if pcm.size <= chunk_samples:
        return [pcm]
    return [pcm[start : start + chunk_samples] for start in range(0, pcm.size, chunk_samples)]


def _join_full_text(texts: list[str]) -> str:
    """Concatenate segment texts like sm-sdk finish payload."""
    parts = [t.strip() for t in texts if t and t.strip()]
    return " ".join(parts)


class OfflineTranscriptionHandler:
    """Handles POST /v1/audio/transcriptions.

    Offline audio is split into fixed 5s windows (no VAD). Each window is
    recognized with SenseVoice and streamed as NDJSON ``transcript`` lines;
    the final ``finish`` line carries the concatenated full text (+ optional summary).
    """

    def __init__(
        self,
        config: AppConfig,
        summary_config: SummaryConfig,
        lane_id: int,
        sensitive_filter: SensitiveWordFilter | None = None,
    ) -> None:
        self._config = config
        self._summary_config = summary_config
        self._lane_id = lane_id
        self._sensitive_filter = sensitive_filter or SensitiveWordFilter([])
        # 记录开始转写的墙钟时间，落盘文件名用
        self._started_at = time.time()
        self._transcript_dir = Path(config.output_dir) / "transcripts"
        self._transcript_dir.mkdir(parents=True, exist_ok=True)

    async def transcribe(
        self,
        file: UploadFile,
        *,
        response_format: str = "json",
        enable_summary: bool = False,
    ) -> StreamingResponse:
        filename = getattr(file, "filename", None) or "audio.wav"
        pcm, _sample_rate = await asyncio.to_thread(_read_wav, file)

        if response_format == "text":
            return StreamingResponse(
                self._text_stream(pcm, filename=filename),
                media_type="text/plain",
            )

        return StreamingResponse(
            self._ndjson_stream(pcm, filename=filename, enable_summary=enable_summary),
            media_type="application/x-ndjson",
        )

    async def _ndjson_stream(
        self,
        pcm: np.ndarray,
        *,
        filename: str,
        enable_summary: bool,
    ):
        segments = _chunk_pcm(pcm, chunk_seconds=CHUNK_SECONDS, sample_rate=SAMPLE_RATE)
        duration_s = float(pcm.size) / float(SAMPLE_RATE)
        _logger.info(
            "offline transcription: file=%s pcm_samples=%d duration_s=%.2f "
            "chunk_seconds=%.1f segments=%d",
            filename,
            pcm.size,
            duration_s,
            CHUNK_SECONDS,
            len(segments),
        )

        yield json.dumps(
            {
                "type": "status",
                "status": "segmented",
                "pcm_samples": int(pcm.size),
                "duration_s": duration_s,
                "chunk_seconds": CHUNK_SECONDS,
                "segment_count": len(segments),
            },
            ensure_ascii=False,
        ) + "\n"

        recognizer = SenseVoiceRecognizer(
            model_dir=self._config.offline_asr_model_dir,
            language="zh",
        )

        full_texts: list[str] = []
        error_message: str | None = None
        try:
            for idx, segment in enumerate(segments):
                start_s = idx * CHUNK_SECONDS
                end_s = start_s + (segment.size / float(SAMPLE_RATE))
                text = await asyncio.to_thread(recognizer.recognize, segment)
                text = self._sensitive_filter.filter(text)
                _logger.info(
                    "offline transcription: file=%s segment=%d "
                    "start_s=%.2f end_s=%.2f samples=%d text=%r",
                    filename,
                    idx,
                    start_s,
                    end_s,
                    segment.size,
                    text,
                )
                if text:
                    full_texts.append(text)
                yield json.dumps(
                    {
                        "type": "transcript",
                        "is_final": True,
                        "is_partial": False,
                        "index": idx,
                        "start_s": start_s,
                        "end_s": end_s,
                        "samples": int(segment.size),
                        "segment": {
                            "text": text,
                            "start_time": start_s,
                            "end_time": end_s,
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"
        except Exception as exc:
            _logger.exception("offline transcription failed")
            error_message = f"{type(exc).__name__}: {exc}"
            yield json.dumps({"error": {"message": error_message}}) + "\n"
            return
        finally:
            recognizer.close()

        full_text = _join_full_text(full_texts)
        transcript_filename = _transcript_filename(filename, self._lane_id, self._started_at)
        transcript_path = self._transcript_dir / transcript_filename
        await asyncio.to_thread(transcript_path.write_text, full_text, encoding="utf-8")

        summary: dict[str, Any] = {"provider": "disabled", "reason": "not requested"}
        if enable_summary and self._summary_config.offline_enabled:
            if full_text.strip():
                summary = await self._generate_summary(full_text)
            else:
                summary = {
                    "provider": "skipped",
                    "reason": "empty transcript",
                }

        finish_payload: dict[str, Any] = {
            "type": "finish",
            "text": full_text,
            "segment_count": len(segments),
            "non_empty_segments": len(full_texts),
            "chunk_seconds": CHUNK_SECONDS,
            "transcript_file": {
                "filename": transcript_filename,
                "download_url": f"/v1/transcripts/{transcript_filename}",
            },
            "summary": summary,
        }
        if error_message:
            finish_payload["error"] = {"message": error_message}
        elif not full_text.strip():
            finish_payload["error"] = {
                "message": "empty transcript: no speech recognized",
            }
        yield json.dumps(finish_payload, ensure_ascii=False) + "\n"

    async def _text_stream(self, pcm: np.ndarray, *, filename: str):
        segments = _chunk_pcm(pcm, chunk_seconds=CHUNK_SECONDS, sample_rate=SAMPLE_RATE)
        recognizer = SenseVoiceRecognizer(
            model_dir=self._config.offline_asr_model_dir,
            language="zh",
        )
        try:
            for segment in segments:
                text = await asyncio.to_thread(recognizer.recognize, segment)
                text = self._sensitive_filter.filter(text)
                if text:
                    yield text + "\n"
        finally:
            recognizer.close()

    async def _generate_summary(self, text: str) -> dict[str, Any]:
        client = SummaryClient(config=self._summary_config)
        try:
            result = await client.generate(text)
            return result.to_payload()
        except Exception as exc:
            _logger.exception("summary generation failed")
            return {
                "provider": self._summary_config.provider,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            await client.close()


async def handle_offline_transcription(
    config: AppConfig,
    summary_config: SummaryConfig,
    file: UploadFile,
    *,
    response_format: str = "json",
    enable_summary: bool = False,
    lane_id: int = 0,
    sensitive_filter: SensitiveWordFilter | None = None,
) -> StreamingResponse:
    handler = OfflineTranscriptionHandler(
        config=config,
        summary_config=summary_config,
        lane_id=lane_id,
        sensitive_filter=sensitive_filter,
    )
    return await handler.transcribe(
        file,
        response_format=response_format,
        enable_summary=enable_summary,
    )
