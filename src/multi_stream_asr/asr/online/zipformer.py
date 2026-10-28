from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar

from multi_stream_asr.config import EndpointConfig
from multi_stream_asr.models import AudioFrame, OnlineRecognition

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class ZipformerRecognizer:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        endpoint: EndpointConfig | None = None,
        *,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        provider: str = "spacemit",
        provider_config: str | Path | None = None,
        hotwords_file: str | Path | None = None,
        hotwords_score: float = 1.5,
    ) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx is required for ZipformerRecognizer") from exc

        self._sample_rate = sample_rate
        self._lock = threading.Lock()

        # vad_driven=True 的识别器（如 funasr paraformer）端点检测不可用，
        # 断句交给 pipeline 的能量 VAD；zipformer 自身恒为 False
        self.vad_driven = False
        if model_dir is None or endpoint is None:
            raise ValueError("model_dir and endpoint are required")
        model_path = Path(model_dir)
        tokens = self._require_file(model_path / "tokens.txt")
        decoder = self._pick_onnx(model_path, "decoder")
        joiner = self._pick_onnx(model_path, "joiner")
        resolved_provider = provider.strip().lower()
        if resolved_provider == "spacemit":
            if provider_config is not None:
                config_path = self._require_file(Path(provider_config))
            else:
                config_path = model_path / "zipformer.config"
                if not config_path.is_file():
                    # 社区模型没有进迭 EP 配置时生成默认配置（与 lane 配置同内容）
                    config_path.write_text(
                        "SPACEMIT_EP_PERFER_CORE_ARCH = 0x5064\n"
                        "SPACEMIT_EP_INTRA_THREAD_NUM = 1\n"
                        "SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1\n",
                        encoding="utf-8",
                    )
                    _logger.info("生成默认 zipformer.config: %s", config_path)
            provider_arg = f"spacemit:{config_path}"
            # 兜底顺序：.q.onnx（进迭定制量化，最快）→ int8（EP 实测可加速）→ fp32
            encoder = self._pick_onnx(model_path, "encoder", prefer_q=True)
        elif resolved_provider == "cpu":
            provider_arg = "cpu"
            encoder = self._pick_onnx(model_path, "encoder")
        else:
            raise RuntimeError(f"Unsupported Zipformer provider: {provider}")

        def _create() -> Any:
            decoding_method = "greedy_search"
            extra: dict[str, Any] = {}
            if hotwords_file is not None:
                # 热词偏置只在 modified_beam_search 下生效
                decoding_method = "modified_beam_search"
                extra["hotwords_file"] = str(hotwords_file)
                extra["hotwords_score"] = hotwords_score
            return sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(tokens),
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                num_threads=1,
                sample_rate=sample_rate,
                feature_dim=feature_dim,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=endpoint.rule1_min_trailing_silence,
                rule2_min_trailing_silence=endpoint.rule2_min_trailing_silence,
                rule3_min_utterance_length=endpoint.rule3_min_utterance_length,
                decoding_method=decoding_method,
                provider=provider_arg,
                **extra,
            )

        self._recognizer: Any = self._call(_create)
        self._stream = self._call(self._recognizer.create_stream)

    def _call(self, fn: Callable[[], T]) -> T:
        with self._lock:
            return fn()

    async def start_run(self) -> None:
        def work() -> None:
            self._stream = self._recognizer.create_stream()

        self._call(work)

    async def accept(self, frame: AudioFrame) -> OnlineRecognition | None:
        if frame.is_finish:
            return None

        def work() -> OnlineRecognition:
            self._stream.accept_waveform(self._sample_rate, frame.pcm)
            decode_count = 0
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
                decode_count += 1

            get_result_all = getattr(self._recognizer, "get_result_all", None)
            if get_result_all is None:
                raw_result = self._recognizer.get_result(self._stream)
                text = raw_result.text if hasattr(raw_result, "text") else str(raw_result)
                last_token_stream_s = None
            else:
                raw_result = get_result_all(self._stream)
                text = raw_result.text if hasattr(raw_result, "text") else str(raw_result)
                timestamps = list(getattr(raw_result, "timestamps", ()))
                start_time = getattr(raw_result, "start_time", None)
                last_token_stream_s = None
                if timestamps and start_time is not None:
                    candidate = float(start_time) + float(timestamps[-1])
                    if math.isfinite(candidate):
                        last_token_stream_s = candidate

            is_endpoint = self._recognizer.is_endpoint(self._stream)
            if is_endpoint:
                # 端点切分必须换新流，不能 reset() 复用：sherpa-onnx 的
                # reset(stream) 在静音期触发后会让后续解码持续输出 blank
                # （整句静默丢失，本地仿真稳定复现）；create_stream 无此问题。
                # 注意此处已在 _call 序列化上下文内，直接调用避免嵌套死锁。
                self._stream = self._recognizer.create_stream()

            stripped = text.strip()
            if decode_count or stripped:
                _logger.info(
                    "zipformer decode: stream=%d decode_count=%d text=%r endpoint=%s",
                    frame.stream_id,
                    decode_count,
                    text,
                    is_endpoint,
                )

            return OnlineRecognition(
                stream_id=frame.stream_id,
                sentence_id=0,
                revision=0,
                text=stripped,
                timestamp_ms=frame.timestamp_ms,
                is_endpoint=is_endpoint,
                last_token_stream_s=last_token_stream_s,
            )

        return self._call(work)

    async def reset(self) -> None:
        def work() -> None:
            self._recognizer.reset(self._stream)

        self._call(work)

    async def flush(self) -> str | None:
        """句尾冲刷（仅 vad_driven 模型实现，如 funasr paraformer）。

        把流内剩余音频以 is_final=True 解完并返回整句文本；
        识别器不支持冲刷（zipformer）时返回 None，调用方忽略即可。"""

        def work() -> str | None:
            flush_stream = getattr(self._recognizer, "flush_stream", None)
            if flush_stream is None:
                return None
            return flush_stream(self._stream)

        return self._call(work)

    async def close(self) -> None:
        def work() -> None:
            self._recognizer = None
            self._stream = None

        self._call(work)

    @staticmethod
    def _require_file(path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"required file not found: {path}")
        return path

    @staticmethod
    def _pick_onnx(model_path: Path, stem: str, prefer_q: bool = False) -> Path:
        """在模型目录里按偏好挑 onnx 文件，文件名不再限定 epoch 命名。

        prefer_q=True（spacemit provider）：.q.onnx（进迭定制量化，最快）
            → int8（EP 实测可加速，RTF 0.35 vs CPU 0.49）→ fp32
        prefer_q=False（cpu provider）：int8 → fp32（不选 .q.onnx）
        """
        cands = sorted(model_path.glob(f"*{stem}*.onnx"))
        if not cands:
            raise FileNotFoundError(
                f"required file not found: {model_path}/*{stem}*.onnx"
            )
        if prefer_q:
            for c in cands:
                if c.name.endswith(".q.onnx"):
                    _logger.info("zipformer %s 使用进迭定制量化: %s", stem, c.name)
                    return c
        for c in cands:
            if ".int8." in c.name:
                _logger.info("zipformer %s 使用 int8: %s", stem, c.name)
                return c
        for c in cands:
            if ".int8." not in c.name and not c.name.endswith(".q.onnx"):
                _logger.info("zipformer %s 使用 fp32: %s", stem, c.name)
                return c
        return cands[0]


def write_zipformer_provider_config(
    *,
    model_dir: Path,
    stream_id: int,
    lanes: int,
) -> Path:
    intra_threads = 1
    config_path = model_dir / f"lane-{stream_id}" / "zipformer.config"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config_path.with_suffix(".tmp")
    temporary_path.write_text(
        "SPACEMIT_EP_PERFER_CORE_ARCH = 0x5064\n"
        f"SPACEMIT_EP_INTRA_THREAD_NUM = {intra_threads}\n"
        "SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1\n",
        encoding="utf-8",
    )
    temporary_path.replace(config_path)
    return config_path
