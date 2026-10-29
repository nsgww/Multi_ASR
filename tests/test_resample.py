from __future__ import annotations

import numpy as np
import pytest

from multi_stream_asr.audio.resample import SpeexResampler, SpeexResamplerError


def test_linear_resample_fallback_on_missing_library() -> None:
    resampler = SpeexResampler(
        input_rate=48000,
        output_rate=16000,
        lib_path="/nonexistent/libspeexdsp.so",
        allow_fallback=True,
    )
    input_pcm = np.sin(np.linspace(0, 2 * np.pi * 500, 4800)).astype(np.int16)
    output = resampler.resample(input_pcm)
    expected_size = int(np.ceil(input_pcm.size * 16000 / 48000))
    assert abs(output.size - expected_size) <= 2
    resampler.close()


def test_resample_requires_int16() -> None:
    resampler = SpeexResampler(
        input_rate=48000,
        output_rate=16000,
        lib_path="/nonexistent/libspeexdsp.so",
        allow_fallback=True,
    )
    with pytest.raises(SpeexResamplerError):
        resampler.resample(np.asarray([0.1, 0.2]))
    resampler.close()


def test_resample_empty_buffer() -> None:
    resampler = SpeexResampler(
        input_rate=48000,
        output_rate=16000,
        lib_path="/nonexistent/libspeexdsp.so",
        allow_fallback=True,
    )
    output = resampler.resample(np.asarray([], dtype=np.int16))
    assert output.size == 0
    resampler.close()
