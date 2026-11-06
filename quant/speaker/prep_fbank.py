# -*- coding: utf-8 -*-
"""用 kaldifeat 生成声纹模型校准 fbank（与 sherpa-onnx 特征口径一致）。

把长 wav 按线上同样的 1.5s 窗 / 0.5s hop 切段，每段提 [T, 80] fbank，
逐段保存为 calib_fbank/*.npy。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import fbank_np

SR = 16000
WINDOW = int(1.5 * SR)
HOP = int(0.5 * SR)

HERE = Path(__file__).resolve().parent
WAVS = [
    Path(r"E:\ASR\last\4-asr-c\audio\00_simple_news_16000hz_1ch.wav"),
    Path(r"E:\ASR\last\4-asr-c\audio\01_general_mickey_16000hz_1ch.wav"),
    Path(r"E:\ASR\last\4-asr-c\audio\01_general_xinmama_16000hz_1ch.wav"),
    Path(r"E:\ASR\last\4-asr-c\audio\02_middle_sales_16000hz_1ch.wav"),
]
OUT_DIR = HERE / "calib_fbank"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for wav in WAVS:
        samples = fbank_np.read_wav(wav)
        for start in range(0, max(len(samples) - WINDOW, 0) + 1, HOP):
            seg = samples[start : start + WINDOW]
            feat = fbank_np.fbank(seg)
            if feat.shape[0] < 10:
                continue
            np.save(OUT_DIR / f"{wav.stem}_{n:04d}.npy", feat)
            n += 1
        print(f"{wav.name}: total {n} segments so far")
    print(f"done: {n} fbank segments -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
