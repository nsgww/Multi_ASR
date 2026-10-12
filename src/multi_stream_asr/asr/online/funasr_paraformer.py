"""FunASR paraformer-online 识别器适配层。

把 funasr_onnx 的 Paraformer（官方在线实现，cache 递推流式）包成
sherpa-onnx OnlineRecognizer 的接口形状，供 ZipformerRecognizer/管线复用。

与 sherpa 转换版的区别：端点检测不可用（is_endpoint 恒 False），
断句交给 pipeline 的能量 VAD（vad_driven=True 标记）。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_logger = logging.getLogger(__name__)


def _collapse_cjk_spaces(text: str) -> str:
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


def _extract_text(result) -> str:
    """funasr 返回 [ {preds: ...} ]，preds 可能是 str 或 [str]。"""
    head = result[0] if isinstance(result, (list, tuple)) and result else result
    if not isinstance(head, dict):
        return ""
    preds = head.get("preds")
    if isinstance(preds, str):
        return preds.strip()
    if isinstance(preds, (list, tuple)) and preds and isinstance(preds[0], str):
        return preds[0].strip()
    return ""


class _Stream:
    """sherpa stream 形状：accept_waveform(rate, samples) 累积待解码音频。"""

    __slots__ = ("cache", "pending", "text")

    def __init__(self) -> None:
        self.cache: dict = {}
        self.pending = np.zeros(0, dtype=np.float32)
        self.text = ""

    def accept_waveform(self, sample_rate: int, samples) -> None:
        self.pending = np.concatenate(
            (self.pending, np.asarray(samples, dtype=np.float32)))


class FunAsrParaformerRecognizer:
    """funasr_onnx paraformer-online 的 sherpa 接口包装。

    vad_driven=True：告知 pipeline 这个模型的断句由能量 VAD 负责
    （模型自身没有可用的端点检测）。
    """

    vad_driven = True

    def __init__(self, model_dir, chunk_size=(5, 10, 5), threads: int = 2):
        from funasr_onnx.paraformer_online_bin import Paraformer

        self._model = Paraformer(
            str(model_dir),
            batch_size=1,
            device_id="-1",
            quantize=False,
            intra_op_num_threads=threads,
            chunk_size=list(chunk_size),
        )
        self._stride = int(chunk_size[1]) * 960   # chunk_size[1]=10 → 600ms
        if self._stride <= 0:
            self._stride = 9600
        _logger.info("FunASR paraformer 加载完成: %s stride=%d", model_dir,
                     self._stride)

    def create_stream(self) -> _Stream:
        return _Stream()

    def is_ready(self, stream: _Stream) -> bool:
        return stream.pending.size >= self._stride

    def decode_stream(self, stream: _Stream) -> None:
        chunk = stream.pending[: self._stride]
        stream.pending = stream.pending[self._stride:]
        result = self._model(
            chunk, param_dict={"cache": stream.cache, "is_final": False})
        piece = _collapse_cjk_spaces(_extract_text(result))
        if piece:
            stream.text += piece

    def flush_stream(self, stream: _Stream) -> str:
        """句尾冲刷：pending 里不足一个 stride 的尾巴以 is_final=True 解完，
        返回整句完整文本（对齐 demo 的每句收尾行为）。"""
        if stream.pending.size:
            result = self._model(
                stream.pending,
                param_dict={"cache": stream.cache, "is_final": True})
            piece = _collapse_cjk_spaces(_extract_text(result))
            if piece:
                stream.text += piece
            stream.pending = np.zeros(0, dtype=np.float32)
        return stream.text

    def get_result(self, stream: _Stream):
        return SimpleNamespace(text=stream.text)

    def is_endpoint(self, stream: _Stream) -> bool:
        return False     # 无端点检测：断句由 pipeline 的 silero VAD 负责

    def reset(self, stream: _Stream) -> None:
        stream.cache = {}
        stream.pending = np.zeros(0, dtype=np.float32)
        stream.text = ""
