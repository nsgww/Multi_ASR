from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)

DEFAULT_VAD_MODEL_FILENAME = "silero_vad.onnx"
SAMPLE_RATE = 16000


class VadSegmenter:
    """Voice activity detection using sherpa-onnx VAD.

    Splits a PCM16 16kHz mono buffer into speech segments. Segments longer than
    ``max_segment_duration_seconds`` are further split to avoid feeding overly
    long audio to the ASR model.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        model_filename: str | None = None,
        threshold: float = 0.5,
        min_silence_duration: float = 0.25,
        min_speech_duration: float = 0.25,
        max_segment_duration_seconds: float = 30.0,
        buffer_size_in_seconds: float = 60.0,
        num_threads: int = 1,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_path = self._resolve_model_path(model_filename)
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_segment_duration_seconds = max_segment_duration_seconds
        self.buffer_size_in_seconds = buffer_size_in_seconds
        self.num_threads = num_threads
        self._vad: Any | None = None

    def _resolve_model_path(self, model_filename: str | None) -> Path:
        if model_filename is not None:
            path = self.model_dir / model_filename
            if path.is_file():
                return path
            raise FileNotFoundError(f"specified VAD model not found: {path}")

        # Try common filenames
        for name in (DEFAULT_VAD_MODEL_FILENAME, "model.onnx", "vad.onnx"):
            path = self.model_dir / name
            if path.is_file():
                return path

        # If directory contains exactly one .onnx file, use it
        onnx_files = list(self.model_dir.glob("*.onnx"))
        if len(onnx_files) == 1:
            return onnx_files[0]

        raise FileNotFoundError(
            f"no usable VAD model found in {self.model_dir}"
        )

    def _ensure_loaded(self) -> Any:
        if self._vad is not None:
            return self._vad
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx is required for VadSegmenter") from exc

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(self.model_path)
        config.silero_vad.threshold = self.threshold
        config.silero_vad.min_silence_duration = self.min_silence_duration
        config.silero_vad.min_speech_duration = self.min_speech_duration
        config.sample_rate = SAMPLE_RATE
        config.num_threads = self.num_threads

        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=self.buffer_size_in_seconds
        )
        return self._vad

    def segment(self, pcm: np.ndarray) -> list[np.ndarray]:
        """Return a list of int16 speech segments."""
        pcm = np.asarray(pcm)
        if pcm.dtype != np.int16:
            raise ValueError("VAD input must be int16 PCM")
        if pcm.ndim != 1:
            raise ValueError("VAD input must be mono PCM")

        # sherpa-onnx Silero VAD expects float32 samples in [-1, 1].
        float_pcm = pcm.astype(np.float32) / 32768.0
        vad = self._ensure_loaded()
        try:
            vad.accept_waveform(float_pcm)
            # Without flush(), short files or trailing speech without enough
            # silence never emit a segment → empty offline transcript.
            flush = getattr(vad, "flush", None)
            if callable(flush):
                flush()
            segments: list[np.ndarray] = []
            while not vad.empty():
                segment = vad.front
                samples = np.asarray(segment.samples)
                if samples.dtype != np.int16:
                    samples = (
                        np.clip(samples.astype(np.float32), -1.0, 1.0 - 1e-6) * 32767.0
                    ).astype(np.int16)
                segments.extend(self._split_long_segment(samples))
                vad.pop()
            return segments
        finally:
            reset = getattr(vad, "reset", None)
            if callable(reset):
                reset()

    def _split_long_segment(self, segment: np.ndarray) -> list[np.ndarray]:
        max_samples = int(self.max_segment_duration_seconds * SAMPLE_RATE)
        if segment.size <= max_samples:
            return [segment]
        pieces: list[np.ndarray] = []
        for start in range(0, segment.size, max_samples):
            pieces.append(segment[start : start + max_samples])
        return pieces

    def reset(self) -> None:
        if self._vad is not None:
            self._vad.reset()
