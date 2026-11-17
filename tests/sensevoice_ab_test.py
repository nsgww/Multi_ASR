"""SenseVoice A/B：同一段音频对比 model_quant_optimized vs model.int8 的精度与速度。"""
import sys
import time
import wave

import numpy as np

from multi_stream_asr.asr.offline.sense_voice import SenseVoiceRecognizer

wav_path = "/tmp/test_long16k.wav"
cases = [
    ("quant_optimized", "/app/lg-store/database/models/runtime/offline-asr/sensevoice/sensevoice"),
    ("int8(sherpa)", "/app/lg-store/database/models/runtime/offline-asr/sensevoice-sherpa/sensevoice-sherpa"),
]

with wave.open(wav_path) as w:
    pcm = w.readframes(w.getnframes())
    duration = w.getnframes() / w.getframerate()

results = {}
pcm_arr = np.frombuffer(pcm, dtype=np.int16)
for name, model_dir in cases:
    rec = SenseVoiceRecognizer(model_dir, num_threads=4)
    # 预热（首次推理含图优化/线程池初始化）
    rec.recognize(pcm_arr[: 16000 * 5])
    t0 = time.perf_counter()
    text = rec.recognize(pcm_arr)
    elapsed = time.perf_counter() - t0
    results[name] = (text, elapsed)
    print(f"=== {name} ===")
    print(f"耗时 {elapsed:.1f}s / 音频 {duration:.1f}s / RTF {elapsed/duration:.3f}")
    print(text[:400])

print("\n=== 差异对比 ===")
a, ta = results[cases[0][0]]
b, tb = results[cases[1][0]]
print("文本相同" if a == b else "文本不同")
if a != b:
    import difflib
    for line in difflib.unified_diff(a.split("，"), b.split("，"), lineterm="", n=0):
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            print(line)
