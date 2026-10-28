"""声纹 embedding 提取（sherpa-onnx + CAM++ / WeSpeaker）。

设计要点：

* **模型懒加载**：声纹关闭或模型文件缺失时不占启动时间，也不影响转写主链路。
* **多窗平均**：单个 3s 段在混响下极不稳定，切成 ``window/hop`` 多窗后取
  单位球面均值，实测能显著压低类内方差。
* **首尾裁剪**：VAD 切出的段首尾常带混响拖尾，先按能量裁掉再提特征。
* **每 lane 一个实例**：共享 ONNX session 在定制 EP 上有线程安全风险，
  而且会让 4 路互相等锁。每路一个实例多花约 28MB，可以接受。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from multi_stream_asr.speaker.preflight import verify_speaker_model

_logger = logging.getLogger(__name__)

# 小于这个长度提不出有意义的声纹，直接放弃（交给上层继承上一句）
MIN_EMBED_SECONDS = 0.6


class SpeakerEmbedder(Protocol):
    """声纹提取器接口，便于测试注入假实现。"""

    @property
    def available(self) -> bool:
        """模型是否可用。``False`` 时上层应整体跳过声纹。"""
        ...

    def embed(self, pcm: np.ndarray) -> np.ndarray | None:
        """提取 L2 归一化后的声纹向量；失败或过短返回 ``None``。"""
        ...


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 归一化；零向量原样返回（避免除零）。"""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return (arr / norm).astype(np.float32)


def trim_silence(
    pcm: np.ndarray,
    sample_rate: int,
    *,
    floor_ratio: float = 0.25,
    pad_seconds: float = 0.05,
) -> np.ndarray:
    """按能量裁掉首尾低能量部分，去掉混响拖尾。

    用 25ms 帧算 RMS，保留超过峰值 ``floor_ratio`` 的首末帧，并向两侧各留
    ``pad_seconds`` 的余量。全静音返回空数组。
    """
    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    frame = max(int(0.025 * sample_rate), 1)
    if audio.size < frame * 3:
        return audio

    usable = (audio.size // frame) * frame
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 1e-9:
        return np.zeros(0, dtype=np.float32)

    loud = np.flatnonzero(rms >= peak * floor_ratio)
    if loud.size == 0:
        return np.zeros(0, dtype=np.float32)

    pad = int(pad_seconds * sample_rate)
    start = max(int(loud[0]) * frame - pad, 0)
    end = min((int(loud[-1]) + 1) * frame + pad, audio.size)
    return np.ascontiguousarray(audio[start:end])


class SherpaSpeakerEmbedder:
    """基于 ``sherpa_onnx.SpeakerEmbeddingExtractor`` 的声纹提取器。"""

    def __init__(
        self,
        *,
        model_path: str | Path,
        num_threads: int = 1,
        provider: str = "cpu",
        sample_rate: int = 16000,
        window_seconds: float = 1.5,
        hop_seconds: float = 0.5,
        trim_floor_ratio: float = 0.25,
    ) -> None:
        self._model_path = Path(model_path)
        self._num_threads = int(num_threads)
        self._provider = str(provider)
        self._sample_rate = int(sample_rate)
        self._window = max(int(window_seconds * sample_rate), 1)
        self._hop = max(int(hop_seconds * sample_rate), 1)
        self._trim_floor_ratio = float(trim_floor_ratio)
        self._extractor = None
        self._load_failed = False
        self._last_error: str | None = None

    @property
    def available(self) -> bool:
        if self._load_failed:
            return False
        return self._model_path.is_file()

    @property
    def last_error(self) -> str | None:
        """最近一次加载/推理失败原因，供健康检查与日志使用。"""
        return self._last_error

    def embed(self, pcm: np.ndarray) -> np.ndarray | None:
        extractor = self._ensure_extractor()
        if extractor is None:
            return None

        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if audio.size < int(MIN_EMBED_SECONDS * self._sample_rate):
            return None
        audio = trim_silence(audio, self._sample_rate, floor_ratio=self._trim_floor_ratio)
        if audio.size < int(MIN_EMBED_SECONDS * self._sample_rate):
            return None

        if audio.size <= self._window:
            return self._embed_window(extractor, audio)

        vectors: list[np.ndarray] = []
        for start in range(0, audio.size - self._window + 1, self._hop):
            vec = self._embed_window(extractor, audio[start : start + self._window])
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return None
        if len(vectors) == 1:
            return vectors[0]
        return l2_normalize(np.mean(np.stack(vectors), axis=0))

    # -- 内部实现 ---------------------------------------------------------

    def _ensure_extractor(self):
        if self._extractor is not None:
            return self._extractor
        if self._load_failed:
            return None
        if not self._model_path.is_file():
            self._load_failed = True
            self._last_error = f"model not found: {self._model_path}"
            _logger.warning("声纹模型不存在，声纹功能关闭: %s", self._model_path)
            return None

        # 先在子进程里试一次：sherpa-onnx 遇到缺 model_type / 不支持的算子会
        # 直接段错误，进程内 try/except 拦不住，必须隔离。
        verdict = verify_speaker_model(self._model_path)
        if not verdict.ok:
            self._load_failed = True
            self._last_error = f"preflight failed: {verdict.detail}"
            return None

        try:
            import sherpa_onnx
        except ImportError as exc:
            self._load_failed = True
            self._last_error = f"sherpa_onnx not installed: {exc}"
            _logger.warning("未安装 sherpa_onnx，声纹功能关闭")
            return None

        try:
            self._extractor = self._build_extractor(sherpa_onnx)
        except Exception as exc:  # noqa: BLE001 - 任何加载失败都必须降级
            self._load_failed = True
            self._last_error = f"{type(exc).__name__}: {exc}"
            _logger.exception("加载声纹模型失败，声纹功能关闭: %s", self._model_path)
            return None

        _logger.info("声纹模型加载完成: %s provider=%s", self._model_path, self._provider)
        return self._extractor

    def _build_extractor(self, sherpa_onnx):
        config_cls = sherpa_onnx.SpeakerEmbeddingExtractorConfig
        try:
            config = config_cls(
                model=str(self._model_path),
                num_threads=self._num_threads,
                debug=False,
                provider=self._provider,
            )
        except TypeError:
            # 老版本没有 provider 参数，退回默认 provider
            config = config_cls(
                model=str(self._model_path),
                num_threads=self._num_threads,
                debug=False,
            )
        return sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def _embed_window(self, extractor, samples: np.ndarray) -> np.ndarray | None:
        try:
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=self._sample_rate, waveform=samples)
            stream.input_finished()
            if not extractor.is_ready(stream):
                return None
            vec = np.asarray(extractor.compute(stream), dtype=np.float32).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"inference failed: {type(exc).__name__}: {exc}"
            _logger.debug("声纹推理失败（已跳过该窗）: %s", exc)
            return None
        if vec.size == 0:
            return None
        return l2_normalize(vec)
