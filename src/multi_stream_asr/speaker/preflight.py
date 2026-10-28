"""在子进程里验证声纹模型能不能被 sherpa-onnx 加载。

**为什么必须隔离**：sherpa-onnx 2.x 在 ``model_type`` 缺失/不匹配时不是抛
异常，而是在 C++ 层直接段错误，把整个 Python 进程带走。``try/except`` 拦不住
SIGSEGV，所以只能在子进程里试一次，把结论缓存下来。

典型触发场景：从 ModelScope / 3D-Speaker 仓库直接导出的 onnx 没有
``model_type`` 元数据 —— 服务第一条句子上线就崩。用
``scripts/patch_speaker_model.py`` 补好元数据即可。
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)

PREFLIGHT_TIMEOUT_SECONDS = 90.0

_PROBE_CODE = r"""
import sys

import numpy as np
import sherpa_onnx

model = sys.argv[1]
cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model=model, num_threads=1, debug=False, provider="cpu"
)
extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
stream = extractor.create_stream()
stream.accept_waveform(
    sample_rate=16000, waveform=np.zeros(16000, dtype=np.float32)
)
stream.input_finished()
if not extractor.is_ready(stream):
    print("NOT_READY")
    sys.exit(2)
embedding = extractor.compute(stream)
print("OK", len(embedding))
"""


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    detail: str


_CACHE: dict[tuple[str, int, int], PreflightResult] = {}


def _cache_key(model_path: Path) -> tuple[str, int, int]:
    stat = model_path.stat()
    return (str(model_path.resolve()), int(stat.st_mtime), int(stat.st_size))


def verify_speaker_model(
    model_path: str | Path,
    *,
    timeout: float = PREFLIGHT_TIMEOUT_SECONDS,
) -> PreflightResult:
    """子进程加载模型并跑一次推理；结果按 (路径, mtime, 大小) 缓存。

    非 0 退出（含段错误/缺元数据/算子不支持）一律视为不可用，调用方应整体
    关闭声纹而不是继续尝试。
    """
    path = Path(model_path)
    if not path.is_file():
        return PreflightResult(False, f"model not found: {path}")

    try:
        key = _cache_key(path)
    except OSError as exc:
        return PreflightResult(False, f"stat failed: {exc}")
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result = PreflightResult(False, f"preflight timed out after {timeout:.0f}s")
        _CACHE[key] = result
        return result
    except OSError as exc:
        result = PreflightResult(False, f"preflight spawn failed: {exc}")
        _CACHE[key] = result
        return result

    if completed.returncode == 0:
        result = PreflightResult(True, (completed.stdout or "").strip() or "OK")
    else:
        stderr = (completed.stderr or "").strip().splitlines()
        tail = " | ".join(line for line in stderr[-3:] if line)
        result = PreflightResult(
            False,
            f"exit={completed.returncode} {tail or '(no stderr)'}",
        )

    _CACHE[key] = result
    if not result.ok:
        _logger.warning("声纹模型预检失败，声纹功能关闭: %s —— %s", path, result.detail)
    return result


def clear_preflight_cache() -> None:
    """测试用：清掉缓存。"""
    _CACHE.clear()
