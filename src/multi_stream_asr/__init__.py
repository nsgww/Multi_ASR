# 必须在 sherpa_onnx 之前 import onnxruntime：sherpa 内置的 SpacemiT ort
# fork 和 python 版 onnxruntime 符号冲突——sherpa 先加载的话，之后 python-ort
# 建的 session 会报 "Failed to find input name in the mapping: enc"（实测）。
# 仅 import 顺序即可规避（funasr_onnx 依赖 python 版 onnxruntime）。
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass

__version__ = "0.1.0"
