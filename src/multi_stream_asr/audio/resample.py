from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


class SpeexResamplerError(RuntimeError):
    pass


def _load_libspeexdsp(lib_path: str | Path | None = None) -> ctypes.CDLL:
    candidates: list[str] = []
    if lib_path is not None:
        candidates.append(str(lib_path))
    env = os.environ.get("LIBSPEEXDSP_PATH", "")
    if env:
        candidates.append(env)
    candidates.extend(
        [
            "libspeexdsp.so.1",
            "libspeexdsp.so",
            # 源码编译（./configure --prefix=/usr/local）的默认落点。通常
            # /usr/local/lib 已在 ld.so.conf 里、上面两条就能命中；这里显式列出，
            # 是为了装完还没跑 ldconfig 时也能加载。
            "/usr/local/lib/libspeexdsp.so.1",
            "/usr/local/lib/libspeexdsp.so",
            "/lib/aarch64-linux-gnu/libspeexdsp.so.1",
            "/usr/lib/aarch64-linux-gnu/libspeexdsp.so.1",
            "/usr/lib/riscv64-linux-gnu/libspeexdsp.so.1",
            "/usr/lib/x86_64-linux-gnu/libspeexdsp.so.1",
            "speexdsp.dll",
            "libspeexdsp.dylib",
        ]
    )
    last_err: OSError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            last_err = exc
            continue
    raise SpeexResamplerError(
        "无法加载 libspeexdsp；请安装 libspeexdsp1 或设置 LIBSPEEXDSP_PATH"
    ) from last_err


class SpeexResampler:
    """PCM16 streaming resampler backed by libspeexdsp (ctypes).

    ABI matches sm-sdk / upstream speexdsp:
    ``SpeexResamplerState *speex_resampler_init(nb_channels, in_rate, out_rate, quality, &err)``.
    """

    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        channels: int = 1,
        *,
        quality: int = 5,
        skip_zeros: bool = True,
        warmup_input_samples: int = 1536,
        lib_path: str | Path | None = None,
        allow_fallback: bool = False,
    ) -> None:
        self.input_rate = int(input_rate)
        self.output_rate = int(output_rate)
        self.channels = int(channels)
        self._quality = int(quality)
        self._state: int | None = None
        self._lib: Any | None = None
        self._fallback = False

        try:
            self._lib = _load_libspeexdsp(lib_path)
            self._bind(self._lib)
            self._init_state(skip_zeros=skip_zeros, warmup_input_samples=warmup_input_samples)
        except SpeexResamplerError:
            if not allow_fallback:
                raise
            _logger.warning("libspeexdsp unavailable; using linear resampling fallback")
            self._fallback = True

    @staticmethod
    def _bind(lib: Any) -> None:
        # Correct ABI (sm-sdk / speexdsp):
        # SpeexResamplerState *speex_resampler_init(
        #     spx_uint32_t nb_channels,
        #     spx_uint32_t in_rate,
        #     spx_uint32_t out_rate,
        #     int quality,
        #     int *err);
        lib.speex_resampler_init.restype = ctypes.c_void_p
        lib.speex_resampler_init.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]

        lib.speex_resampler_process_int.restype = ctypes.c_int
        lib.speex_resampler_process_int.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_uint32),
        ]

        lib.speex_resampler_destroy.restype = None
        lib.speex_resampler_destroy.argtypes = [ctypes.c_void_p]

        lib.speex_resampler_skip_zeros.restype = ctypes.c_int
        lib.speex_resampler_skip_zeros.argtypes = [ctypes.c_void_p]

        if hasattr(lib, "speex_resampler_strerror"):
            lib.speex_resampler_strerror.restype = ctypes.c_char_p
            lib.speex_resampler_strerror.argtypes = [ctypes.c_int]

    def _init_state(self, *, skip_zeros: bool, warmup_input_samples: int) -> None:
        assert self._lib is not None
        err = ctypes.c_int(0)
        state = self._lib.speex_resampler_init(
            ctypes.c_uint32(self.channels),
            ctypes.c_uint32(self.input_rate),
            ctypes.c_uint32(self.output_rate),
            ctypes.c_int(self._quality),
            ctypes.byref(err),
        )
        if not state or err.value != 0:
            detail = str(err.value)
            strerror = getattr(self._lib, "speex_resampler_strerror", None)
            if callable(strerror):
                msg = strerror(err.value)
                if msg:
                    detail = msg.decode(errors="ignore")
            raise SpeexResamplerError(f"speex_resampler_init failed: {detail}")
        self._state = state
        if skip_zeros:
            self._lib.speex_resampler_skip_zeros(self._state)
        if warmup_input_samples > 0:
            silence = np.zeros(warmup_input_samples, dtype=np.int16)
            self.resample(silence)

    def resample(self, pcm: np.ndarray) -> np.ndarray:
        """Resample a mono int16 PCM buffer → int16 at ``output_rate``."""
        pcm = np.asarray(pcm)
        if pcm.dtype != np.int16:
            raise SpeexResamplerError("input must be int16 PCM")
        if pcm.ndim != 1:
            raise SpeexResamplerError("only mono PCM is supported")
        if self._fallback:
            return self._linear_resample(pcm)
        return self._speex_resample(pcm)

    def _speex_resample(self, pcm: np.ndarray) -> np.ndarray:
        assert self._lib is not None and self._state is not None
        if not pcm.flags["C_CONTIGUOUS"]:
            pcm = np.ascontiguousarray(pcm)
        in_len = ctypes.c_uint32(pcm.size)
        out_capacity = int(pcm.size * self.output_rate / self.input_rate) + 64
        out = np.zeros(out_capacity, dtype=np.int16)
        out_len = ctypes.c_uint32(out_capacity)

        result = self._lib.speex_resampler_process_int(
            self._state,
            ctypes.c_uint32(0),
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.byref(in_len),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.byref(out_len),
        )
        if result != 0:
            detail = str(result)
            strerror = getattr(self._lib, "speex_resampler_strerror", None)
            if callable(strerror):
                msg = strerror(result)
                if msg:
                    detail = msg.decode(errors="ignore")
            raise SpeexResamplerError(f"speex_resampler_process_int failed: {detail}")
        return out[: out_len.value]

    def _linear_resample(self, pcm: np.ndarray) -> np.ndarray:
        if pcm.size == 0:
            return pcm
        ratio = self.output_rate / self.input_rate
        target_size = int(np.ceil(pcm.size * ratio))
        indices = np.arange(target_size) / ratio
        floor_idx = np.floor(indices).astype(np.int64)
        ceil_idx = np.minimum(floor_idx + 1, pcm.size - 1)
        frac = indices - floor_idx
        return (
            pcm[floor_idx].astype(np.float32) * (1 - frac)
            + pcm[ceil_idx].astype(np.float32) * frac
        ).astype(np.int16)

    def close(self) -> None:
        if self._state is not None and self._lib is not None:
            self._lib.speex_resampler_destroy(self._state)
            self._state = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
