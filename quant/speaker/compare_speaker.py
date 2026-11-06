# -*- coding: utf-8 -*-
"""对比 int8 声纹模型与 fp32 的 embedding 余弦相似度。

运行: .venv-quant/Scripts/python quant/speaker/compare_speaker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
FP32 = HERE / "campplus_epoch50_sv_zh-cn_16k.onnx"
INT8 = HERE / "campplus_epoch50_sv_zh-cn_16k.int8.onnx"
EVAL_DIR = HERE / "calib_fbank"


def embed(sess: ort.InferenceSession, feat: np.ndarray) -> np.ndarray:
    (name,) = [i.name for i in sess.get_inputs()]
    vec = sess.run(None, {name: feat[None, :, :].astype(np.float32)})[0]
    vec = vec.reshape(-1).astype(np.float64)
    return vec / (np.linalg.norm(vec) + 1e-12)


def main() -> int:
    opts = ort.SessionOptions()
    s_fp = ort.InferenceSession(str(FP32), opts, providers=["CPUExecutionProvider"])
    s_q = ort.InferenceSession(str(INT8), opts, providers=["CPUExecutionProvider"])

    files = sorted(EVAL_DIR.glob("*.npy"))
    sims = []
    for f in files:
        feat = np.load(f).astype(np.float32)
        a, b = embed(s_fp, feat), embed(s_q, feat)
        sims.append(float(np.dot(a, b)))
    sims = np.array(sims)
    print(f"n={len(sims)} cosine mean={sims.mean():.4f} min={sims.min():.4f} max={sims.max():.4f}")
    print("ALL PASS" if sims.min() > 0.95 else "CHECK NEEDED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
