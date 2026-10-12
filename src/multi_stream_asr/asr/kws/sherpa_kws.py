from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)

ENCODER_NAME = "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
DECODER_NAME = "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
JOINER_NAME = "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
TOKENS_NAME = "tokens.txt"


@dataclass(frozen=True)
class KWSResult:
    keyword: str
    confidence: float
    start_time: float
    end_time: float


class SherpaKeywordSpotter:
    """Sherpa-ONNX KeywordSpotter wrapper (sm-sdk compatible).

    Model files live under ``model_dir``; keyword lexicon is a separate
    ``keywords_file`` (project ``config/keywords.txt``), not the model copy.
    """

    def __init__(
        self,
        *,
        model_dir: Path,
        keywords_file: Path,
        sample_rate: int = 16000,
        num_threads: int = 1,
        max_active_paths: int = 4,
        keywords_score: float = 1.0,
        keywords_threshold: float = 0.25,
        num_trailing_blanks: int = 1,
        provider: str = "cpu",
        repeat_interval: float = 1.0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.keywords_file = Path(keywords_file)
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.max_active_paths = max_active_paths
        self.keywords_score = keywords_score
        self.keywords_threshold = keywords_threshold
        self.num_trailing_blanks = num_trailing_blanks
        self.provider = provider
        self.repeat_interval = repeat_interval
        self._spotter = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import sherpa_onnx

        encoder = self.model_dir / ENCODER_NAME
        decoder = self.model_dir / DECODER_NAME
        joiner = self.model_dir / JOINER_NAME
        tokens = self.model_dir / TOKENS_NAME
        for path in (encoder, decoder, joiner, tokens, self.keywords_file):
            if not path.is_file():
                raise FileNotFoundError(f"KWS file not found: {path}")

        keywords_text = self.keywords_file.read_text(encoding="utf-8").strip()
        _logger.info("KWS keywords file: %s\n%s", self.keywords_file, keywords_text)

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=self.num_threads,
            max_active_paths=self.max_active_paths,
            keywords_file=str(self.keywords_file),
            keywords_score=self.keywords_score,
            keywords_threshold=self.keywords_threshold,
            num_trailing_blanks=self.num_trailing_blanks,
            provider=self.provider,
        )

    def reload(self) -> None:
        """按当前 keywords_file 重建 spotter（运行时换唤醒词）。

        先构造成功再替换，失败不影响在用的旧 spotter；
        已存在的会话沿用旧词表，新会话生效。
        """
        import sherpa_onnx

        encoder = self.model_dir / ENCODER_NAME
        decoder = self.model_dir / DECODER_NAME
        joiner = self.model_dir / JOINER_NAME
        tokens = self.model_dir / TOKENS_NAME
        new_spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=self.num_threads,
            max_active_paths=self.max_active_paths,
            keywords_file=str(self.keywords_file),
            keywords_score=self.keywords_score,
            keywords_threshold=self.keywords_threshold,
            num_trailing_blanks=self.num_trailing_blanks,
            provider=self.provider,
        )
        with self._lock:
            self._spotter = new_spotter
        _logger.info("KWS spotter reloaded with %s", self.keywords_file)

    def create_session(self) -> "SherpaKeywordSpotterSession":
        if self._spotter is None:
            raise RuntimeError("KWS model is not loaded")
        return SherpaKeywordSpotterSession(
            spotter=self._spotter,
            lock=self._lock,
            sample_rate=self.sample_rate,
            repeat_interval=self.repeat_interval,
        )

    def close(self) -> None:
        self._spotter = None


class SherpaKeywordSpotterSession:
    """Per-session KWS stream state."""

    def __init__(
        self,
        *,
        spotter,
        lock: threading.Lock,
        sample_rate: int,
        repeat_interval: float,
    ) -> None:
        self._spotter = spotter
        self._lock = lock
        self._stream = spotter.create_stream()
        self.sample_rate = sample_rate
        self.repeat_interval = repeat_interval
        self._current_time = 0.0
        self._last_detection_time = 0.0

    def detect(self, audio_chunk: np.ndarray) -> list[KWSResult]:
        audio = np.asarray(audio_chunk, dtype=np.float32)
        if audio.size == 0:
            return []
        chunk_duration = float(audio.size) / float(self.sample_rate)
        results: list[KWSResult] = []
        with self._lock:
            self._stream.accept_waveform(self.sample_rate, audio)
            self._current_time += chunk_duration
            while self._spotter.is_ready(self._stream):
                self._spotter.decode_stream(self._stream)
                keyword_result = self._spotter.get_result(self._stream)
                if not keyword_result or not str(keyword_result).strip():
                    continue
                if self._current_time - self._last_detection_time <= self.repeat_interval:
                    continue
                keyword_text = str(keyword_result).strip()
                results.append(
                    KWSResult(
                        keyword=keyword_text,
                        confidence=1.0,
                        start_time=self._current_time - chunk_duration,
                        end_time=self._current_time,
                    )
                )
                self._last_detection_time = self._current_time
                _logger.info("KWS detected: %s @ %.2fs", keyword_text, self._current_time)
        return results

    def reset(self) -> None:
        with self._lock:
            self._stream = self._spotter.create_stream()
            self._current_time = 0.0
            self._last_detection_time = 0.0

    def close(self) -> None:
        self._stream = None
