from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from multi_stream_asr.asr.online.zipformer import (
    ZipformerRecognizer,
    write_zipformer_provider_config,
)
from multi_stream_asr.config import EndpointConfig
from multi_stream_asr.models import AudioFrame


def write_reference_zipformer_files(model_dir: Path) -> None:
    for name in (
        "tokens.txt",
        "encoder-epoch-99-avg-1.int8.onnx",
        "encoder-epoch-99-avg-1.q.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.int8.onnx",
        "joiner-epoch-99-avg-1.onnx",
        "zipformer.config",
    ):
        (model_dir / name).write_text("stub", encoding="utf-8")


def test_zipformer_recognizer_defaults_to_spacemit_with_quantized_encoder_and_original_joiner(
    monkeypatch, tmp_path: Path
):
    write_reference_zipformer_files(tmp_path)

    captured: dict[str, object] = {}

    class FakeStream:
        pass

    class FakeRecognizer:
        def create_stream(self):
            return FakeStream()

    class FakeOnlineRecognizerFactory:
        @staticmethod
        def from_transducer(**kwargs):
            captured.update(kwargs)
            return FakeRecognizer()

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(OnlineRecognizer=FakeOnlineRecognizerFactory),
    )

    ZipformerRecognizer(tmp_path, EndpointConfig())

    assert captured["num_threads"] == 1
    assert captured["tokens"] == str(tmp_path / "tokens.txt")
    assert captured["encoder"] == str(tmp_path / "encoder-epoch-99-avg-1.q.onnx")
    assert captured["decoder"] == str(tmp_path / "decoder-epoch-99-avg-1.onnx")
    assert captured["joiner"] == str(tmp_path / "joiner-epoch-99-avg-1.onnx")
    assert captured["provider"] == f"spacemit:{tmp_path / 'zipformer.config'}"


@pytest.mark.parametrize("lanes", [2, 4, 6, 8])
def test_write_zipformer_provider_config_uses_one_intra_thread_for_all_lane_counts(
    tmp_path: Path,
    lanes: int,
):
    model_dir = tmp_path / "models" / "zipformer"
    path = write_zipformer_provider_config(
        model_dir=model_dir,
        stream_id=3,
        lanes=lanes,
    )

    assert path == model_dir / "lane-3" / "zipformer.config"
    assert path.read_text(encoding="utf-8") == (
        "SPACEMIT_EP_PERFER_CORE_ARCH = 0x5064\n"
        "SPACEMIT_EP_INTRA_THREAD_NUM = 1\n"
        "SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD = 1\n"
    )


def test_zipformer_recognizer_cpu_provider_uses_int8_encoder_and_streaming_joiner(monkeypatch, tmp_path: Path):
    write_reference_zipformer_files(tmp_path)

    captured: dict[str, object] = {}

    class FakeStream:
        pass

    class FakeRecognizer:
        def create_stream(self):
            return FakeStream()

    class FakeOnlineRecognizerFactory:
        @staticmethod
        def from_transducer(**kwargs):
            captured.update(kwargs)
            return FakeRecognizer()

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(OnlineRecognizer=FakeOnlineRecognizerFactory),
    )

    ZipformerRecognizer(tmp_path, EndpointConfig(), provider="cpu")

    assert captured["encoder"] == str(tmp_path / "encoder-epoch-99-avg-1.int8.onnx")
    assert captured["decoder"] == str(tmp_path / "decoder-epoch-99-avg-1.onnx")
    assert captured["joiner"] == str(tmp_path / "joiner-epoch-99-avg-1.onnx")
    assert captured["provider"] == "cpu"


def test_zipformer_recognizer_fails_when_spacemit_encoder_file_is_missing(monkeypatch, tmp_path: Path):
    for name in (
        "tokens.txt",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.onnx",
        "zipformer.config",
    ):
        (tmp_path / name).write_text("stub", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(
            OnlineRecognizer=types.SimpleNamespace(from_transducer=lambda **kwargs: None),
        ),
    )

    with pytest.raises(FileNotFoundError, match="encoder-epoch-99-avg-1.q.onnx"):
        ZipformerRecognizer(tmp_path, EndpointConfig())


@pytest.mark.asyncio
async def test_zipformer_accept_uses_text_attribute_from_sherpa_result(monkeypatch, tmp_path: Path):
    write_reference_zipformer_files(tmp_path)

    class FakeStream:
        def accept_waveform(self, sample_rate, pcm):
            self.sample_rate = sample_rate
            self.pcm = pcm

    class FakeResult:
        text = "decoded text"

    class FakeRecognizer:
        def create_stream(self):
            return FakeStream()

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):
            raise AssertionError("decode should not run")

        def get_result(self, stream):
            return FakeResult()

        def is_endpoint(self, stream):
            return True

        def reset(self, stream):
            self.reset_stream = stream

    class FakeOnlineRecognizerFactory:
        @staticmethod
        def from_transducer(**kwargs):
            return FakeRecognizer()

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(OnlineRecognizer=FakeOnlineRecognizerFactory),
    )
    recognizer = ZipformerRecognizer(tmp_path, EndpointConfig())

    result = await recognizer.accept(
        AudioFrame(stream_id=1, frame_index=0, pcm=np.asarray([0.1], dtype=np.float32), timestamp_ms=100)
    )

    assert result is not None
    assert result.text == "decoded text"
    assert result.is_endpoint is True


def build_fake_zipformer(
    monkeypatch,
    tmp_path: Path,
    *,
    result: object | None = None,
    text_result: object = "legacy text",
    expose_get_result_all: bool = True,
) -> ZipformerRecognizer:
    write_reference_zipformer_files(tmp_path)

    class FakeStream:
        def accept_waveform(self, sample_rate, pcm):
            self.sample_rate = sample_rate
            self.pcm = pcm

    class BaseFakeRecognizer:
        def create_stream(self):
            return FakeStream()

        def is_ready(self, stream):
            return False

        def decode_stream(self, stream):
            raise AssertionError("decode should not run")

        def get_result(self, stream):
            return text_result

        def is_endpoint(self, stream):
            return False

        def reset(self, stream):
            self.reset_stream = stream

    class TimedFakeRecognizer(BaseFakeRecognizer):
        def get_result_all(self, stream):
            return result

    fake = TimedFakeRecognizer() if expose_get_result_all else BaseFakeRecognizer()
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(
            OnlineRecognizer=types.SimpleNamespace(from_transducer=lambda **kwargs: fake)
        ),
    )
    return ZipformerRecognizer(tmp_path, EndpointConfig())


@pytest.mark.asyncio
async def test_zipformer_reads_token_time_before_endpoint_reset(monkeypatch, tmp_path: Path):
    write_reference_zipformer_files(tmp_path)
    calls: list[str] = []

    class FakeStream:
        def accept_waveform(self, sample_rate, pcm):
            calls.append("accept")

    class FakeResult:
        text = "decoded text"
        start_time = 1.2
        timestamps = [0.04, 0.36]

    class FakeRecognizer:
        def create_stream(self):
            calls.append("create")
            return FakeStream()

        def is_ready(self, stream):
            return False

        def get_result_all(self, stream):
            calls.append("result")
            return FakeResult()

        def is_endpoint(self, stream):
            return True

        def reset(self, stream):
            calls.append("reset")

    fake = FakeRecognizer()
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        types.SimpleNamespace(
            OnlineRecognizer=types.SimpleNamespace(from_transducer=lambda **kwargs: fake)
        ),
    )
    recognizer = ZipformerRecognizer(tmp_path, EndpointConfig())
    await recognizer.start_run()
    result = await recognizer.accept(
        AudioFrame(0, 0, np.asarray([0.1], dtype=np.float32), 100)
    )

    assert result is not None
    assert result.text == "decoded text"
    assert result.last_token_stream_s == pytest.approx(1.56)
    assert calls.count("create") == 2
    assert calls.index("result") < calls.index("reset")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start_time,timestamps",
    [(0.0, []), (float("nan"), [0.1]), (0.0, [float("inf")])],
)
async def test_zipformer_omits_invalid_token_time(
    monkeypatch, tmp_path: Path, start_time: float, timestamps: list[float]
):
    recognizer = build_fake_zipformer(
        monkeypatch,
        tmp_path,
        result=types.SimpleNamespace(
            text="text", start_time=start_time, timestamps=timestamps
        ),
    )
    result = await recognizer.accept(AudioFrame(0, 0, np.ones(1, dtype=np.float32), 100))
    assert result is not None
    assert result.last_token_stream_s is None


@pytest.mark.asyncio
async def test_zipformer_falls_back_when_get_result_all_is_unavailable(
    monkeypatch, tmp_path: Path
):
    recognizer = build_fake_zipformer(
        monkeypatch,
        tmp_path,
        text_result="legacy text",
        expose_get_result_all=False,
    )
    result = await recognizer.accept(AudioFrame(0, 0, np.ones(1, dtype=np.float32), 100))
    assert result is not None
    assert result.text == "legacy text"
    assert result.last_token_stream_s is None
