# -*- coding: utf-8 -*-
"""纯 numpy 实现 kaldi fbank（对齐 sherpa-onnx / kaldifeat 口径）。

参数：16kHz，frame 25ms / shift 10ms，povey 窗，preemph 0.97，去直流，
80 bin mel 三角滤波（面积归一），log。snip_edges=False 两端各补
(frame_length - frame_shift)/2 样本。dither=0。

只用于量化校准特征，分布与真实特征一致即可。
"""
from __future__ import annotations

import wave

import numpy as np

SR = 16000
FRAME_LEN = 400  # 25ms
FRAME_SHIFT = 160  # 10ms
FFT = 512
NUM_BINS = 80
PAD = (FRAME_LEN - FRAME_SHIFT) // 2  # 120


def read_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _povey_window(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return np.sin(np.pi * (i + 0.5) / n) ** 0.85


def _mel_filterbank() -> np.ndarray:
    def hz_to_mel(f):
        return 1127.0 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (np.exp(np.asarray(m, dtype=np.float64) / 1127.0) - 1.0)

    nyquist = SR / 2
    low, high = 20.0, nyquist
    mel_pts = np.linspace(hz_to_mel(low), hz_to_mel(high), NUM_BINS + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((FFT + 1) * hz_pts / SR).astype(int)
    bins = np.clip(bins, 0, FFT // 2)
    fb = np.zeros((NUM_BINS, FFT // 2 + 1), dtype=np.float64)
    for i in range(NUM_BINS):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c <= l:
            c = l + 1
        if r <= c:
            r = c + 1
        left = np.arange(l, c)
        right = np.arange(c, r)
        fb[i, left] = (left - l) / max(c - l, 1)
        fb[i, c] = 1.0
        fb[i, right] = (r - right) / max(r - c, 1)
    # 面积归一（kaldi 风格：每 bin 除以 bin 宽）
    widths = np.diff(hz_pts)
    fb /= widths[:NUM_BINS][:, None] * (2.0 / (SR / FFT))
    return fb.astype(np.float32)


_FB = _mel_filterbank()
_WIN = _povey_window(FRAME_LEN).astype(np.float32)


def fbank(samples: np.ndarray) -> np.ndarray:
    """[N] float32 PCM -> [T, 80] fbank。"""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    # 预加重（kaldi 首样本不做）
    x = np.concatenate([x[:1], x[1:] - 0.97 * x[:-1]])
    # snip_edges=False：两端补边样本
    x = np.pad(x, PAD, mode="edge")
    n_frames = 1 + (x.size - FRAME_LEN) // FRAME_SHIFT
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_SHIFT * np.arange(n_frames)[:, None]
    frames = x[idx]
    # 去直流 + 加窗
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames = frames * _WIN[None, :]
    spec = np.abs(np.fft.rfft(frames, n=FFT)) ** 2  # [T, 257]
    mel = spec @ _FB.T
    mel = np.maximum(mel, 1.0e-10)
    return np.log(mel).astype(np.float32)
