from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


class SenseVoiceRecognizer:
    """Offline ASR using sherpa-onnx SenseVoice model.

    Defaults to CPU provider and looks for quantized models first:
    ``model_quant_optimized.onnx`` / ``model_quant.onnx`` / ``model.int8.onnx``
    / ``model.onnx`` / ``model.q.onnx``.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        language: str = "zh",
        use_itn: bool = True,
        num_threads: int = 1,
        provider: str = "cpu",
        debug: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.language = language
        self.use_itn = use_itn
        self.num_threads = num_threads
        self.provider = provider.strip().lower()
        self.debug = debug
        self._recognizer: Any | None = None

    def _resolve_model_and_provider(self) -> tuple[Path, str]:
        if self.provider == "spacemit":
            model_path = self.model_dir / "model.q.onnx"
            config_path = self.model_dir / "sensevoice.config"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"required SenseVoice quantized model not found: {model_path}"
                )
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"required SenseVoice SpacemiT config not found: {config_path}"
                )
            return model_path, f"spacemit:{config_path}"

        if self.provider == "cpu":
            for name in (
                "model_quant_optimized.onnx",
                "model_quant.onnx",
                "model.int8.onnx",
                "model.onnx",
                "model.q.onnx",
            ):
                candidate = self.model_dir / name
                if candidate.is_file():
                    return candidate, "cpu"
            raise FileNotFoundError(
                f"no SenseVoice model found in {self.model_dir} "
                "(tried model_quant_optimized.onnx / model_quant.onnx / "
                "model.int8.onnx / model.onnx / model.q.onnx)"
            )

        raise RuntimeError(f"Unsupported SenseVoice provider: {self.provider}")

    def _ensure_loaded(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx is required for SenseVoiceRecognizer") from exc

        model_path, provider_arg = self._resolve_model_and_provider()
        # sherpa-onnx 需要整数 ID 的 tokens；原 SenseVoice tokens.txt 是 token<TAB>score 格式
        # 如果目录下已有转换后的 tokens_sherpa.txt 则优先使用
        tokens_path = (
            self.model_dir / "tokens_sherpa.txt"
            if (self.model_dir / "tokens_sherpa.txt").is_file()
            else self._require_file("tokens.txt")
        )

        kwargs: dict[str, Any] = {
            "model": str(model_path),
            "tokens": str(tokens_path),
            "language": self.language,
            "use_itn": self.use_itn,
            "num_threads": self.num_threads,
            "sample_rate": 16000,
            "feature_dim": 80,
            "provider": provider_arg,
            "decoding_method": "greedy_search",
            "debug": self.debug,
        }

        for key, filename in (("lexicon", "lexicon.txt"),):
            path = self.model_dir / filename
            if path.is_file():
                kwargs[key] = str(path)

        # Filter kwargs to what this sherpa-onnx build accepts.
        sig = inspect.signature(sherpa_onnx.OfflineRecognizer.from_sense_voice)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        _logger.info(
            "loading SenseVoice model=%s provider=%s",
            model_path.name,
            kwargs.get("provider", "default"),
        )
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(**kwargs)
        return self._recognizer

    def _require_file(self, filename: str) -> Path:
        path = self.model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"required SenseVoice file not found: {path}")
        return path

    def recognize(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        """Recognize a mono int16 or float32 PCM buffer."""
        pcm = np.asarray(pcm)
        if pcm.ndim != 1:
            raise ValueError("SenseVoice input must be mono PCM")
        # sm-sdk / sherpa-onnx SenseVoice examples feed float32 in [-1, 1].
        if pcm.dtype == np.int16:
            float_pcm = pcm.astype(np.float32) / 32768.0
        else:
            float_pcm = pcm.astype(np.float32)

        recognizer = self._ensure_loaded()
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, float_pcm)
        recognizer.decode_stream(stream)
        try:
            result = recognizer.get_result(stream)
        except Exception:
            result = getattr(stream, "result", None)
        text = getattr(result, "text", "") if result is not None else ""
        if not text and result is not None and not hasattr(result, "text"):
            text = str(result)
        return str(text).strip()

    def recognize_segments(
        self, segments: list[np.ndarray], sample_rate: int = 16000
    ) -> list[str]:
        """Recognize a list of PCM segments and return their texts."""
        return [self.recognize(segment, sample_rate) for segment in segments]

    def close(self) -> None:
        self._recognizer = None
