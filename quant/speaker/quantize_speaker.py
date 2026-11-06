# -*- coding: utf-8 -*-
"""把 CAM++ 声纹模型量化为 int8（QDQ，per-channel，weight QInt8 / act QUInt8）。

输入是 [1, frames, 80] fbank 特征；校准数据为板端 sherpa 同口径的 fbank npy。
运行: .venv-quant/Scripts/python quant/speaker/quantize_speaker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

HERE = Path(__file__).resolve().parent
MODEL = HERE / "campplus_epoch50_sv_zh-cn_16k.onnx"
OUT = HERE / "campplus_epoch50_sv_zh-cn_16k.int8.onnx"
CALIB_DIR = HERE / "calib_fbank"


class FbankReader(CalibrationDataReader):
    def __init__(self, calib_dir: Path, input_name: str):
        self.files = sorted(calib_dir.glob("*.npy"))
        assert self.files, f"no npy in {calib_dir}"
        self.input_name = input_name
        self._iter = iter(self.files)

    def get_next(self):
        try:
            f = next(self._iter)
        except StopIteration:
            return None
        feat = np.load(f).astype(np.float32)  # [T, 80]
        return {self.input_name: feat[None, :, :]}


def main() -> int:
    import onnx

    m = onnx.load(str(MODEL))
    (input_name,) = [i.name for i in m.graph.input]
    print(f"input={input_name}")

    dr = FbankReader(CALIB_DIR, input_name)
    quantize_static(
        model_input=str(MODEL),
        model_output=str(OUT),
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        calibrate_method="MinMax",
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    print(f"written: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
