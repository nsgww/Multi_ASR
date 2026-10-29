"""声纹注册用的音频解码：wav 字节流 → 16k 单声道 float32。"""

from __future__ import annotations

import io
import wave

import numpy as np

SAMPLE_RATE = 16000
_MAX_WAV_BYTES = 32 * 1024 * 1024  # 单次注册音频上限 32MB


def wav_bytes_to_pcm16k(data: bytes) -> np.ndarray:
    """解码 int16 wav 字节流；多声道混单声道，非 16k 线性重采样。"""
    if len(data) > _MAX_WAV_BYTES:
        raise ValueError(f"音频文件过大（>{_MAX_WAV_BYTES // 1024 // 1024}MB）")
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"无法解码 wav 文件: {exc}") from None
    if width != 2:
        raise ValueError(f"只支持 16bit PCM wav（当前 sampwidth={width}）")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        axis = np.arange(audio.size)
        target = np.linspace(0, audio.size - 1, int(audio.size * SAMPLE_RATE / rate))
        audio = np.interp(target, axis, audio).astype(np.float32)
    if audio.size == 0:
        raise ValueError("音频为空")
    return np.asarray(audio, dtype=np.float32)
