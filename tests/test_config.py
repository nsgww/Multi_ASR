import hashlib
import os
import tomllib
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from multi_stream_asr.config import (
    AppConfig,
    EndpointConfig,
    LaneInputConfig,
    ModelConfig,
    OFFLINE_MODELS,
    PROJECT_ROOT,
    SharedQwenConfig,
    ZIPFORMER_WARMUP_AUDIO_PATH,
    build_lane_configs,
    validate_offline_model_files,
)


def test_zipformer_warmup_audio_is_fixed_valid_package_data() -> None:
    audio = ZIPFORMER_WARMUP_AUDIO_PATH

    assert audio == Path(__file__).resolve().parents[1] / (
        "src/multi_stream_asr/assets/zipformer_warmup.wav"
    )
    assert hashlib.sha256(audio.read_bytes()).hexdigest() == (
        "d29d16456d9c7662135fbd696605391745ed56ce47eee3385a37c4b120a9691f"
    )
    with wave.open(str(audio), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 22080

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    assert pyproject["tool"]["setuptools"]["package-data"]["multi_stream_asr"] == [
        "assets/*.wav",
        "assets/wav16k/*.wav",
    ]


def test_default_input_dir_ships_valid_package_wavs() -> None:
    input_dir = AppConfig().input_dir

    assert input_dir == Path(__file__).resolve().parents[1] / (
        "src/multi_stream_asr/assets/wav16k"
    )
    wav_files = sorted(input_dir.glob("*.wav"))
    assert len(wav_files) == 6
    for wav_path in wav_files:
        with wave.open(str(wav_path), "rb") as wav_file:
            assert wav_file.getframerate() == 16000
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2


def test_shared_qwen_defaults_match_validated_four_lane_service():
    shared = SharedQwenConfig()

    assert shared.host == "127.0.0.1"
    assert shared.port == 8090
    assert shared.decoder_threads == 6
    assert shared.n_parallel == 4
    assert shared.n_ctx == 8192
    assert shared.request_timeout_ms == 120000


def test_app_config_exposes_shared_qwen_config():
    config = AppConfig()

    assert config.shared_qwen == SharedQwenConfig()


def test_default_config_values():
    config = AppConfig()

    assert config.lanes == 4
    assert config.frame_duration_ms == 100
    assert config.input_dir == (
        Path(__file__).resolve().parents[1]
        / "src/multi_stream_asr/assets/wav16k"
    )
    assert config.models.online_num_threads == 1
    assert config.models.online_model_dir == Path(
        "models/asr/"
        "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
    ).resolve()
    assert config.models.offline_model_dir == (
        Path(__file__).resolve().parents[1] / "models/asr/qwen3asr"
    )
    assert config.endpoint == EndpointConfig(
        rule1_min_trailing_silence=1.0,
        rule2_min_trailing_silence=0.5,
        rule3_min_utterance_length=10.0,
        cut_margin_seconds=0.1,
    )
    assert config.segment_cache_seconds == 15.0
    assert config.shared_qwen.request_timeout_ms == 120000


@pytest.mark.parametrize("margin", [-0.001, 0.5, 0.6, float("nan"), float("inf")])
def test_endpoint_rejects_invalid_cut_margin(margin: float):
    with pytest.raises(ValueError, match="cut_margin_seconds"):
        EndpointConfig(cut_margin_seconds=margin)


def test_endpoint_accepts_margin_strictly_below_rule2():
    endpoint = EndpointConfig(
        rule2_min_trailing_silence=0.25,
        cut_margin_seconds=0.249,
    )
    assert endpoint.cut_margin_seconds == 0.249


@pytest.mark.parametrize("lanes", [2, 4, 6, 8])
def test_supported_lane_counts(lanes):
    assert AppConfig(lanes=lanes).lanes == lanes


@pytest.mark.parametrize("lanes", [0, 1, 3, 5, 7, 9])
def test_rejects_unsupported_lane_counts(lanes):
    with pytest.raises(ValueError, match="lanes must be one of"):
        AppConfig(lanes=lanes)


def test_build_lane_configs_assigns_only_stream_identity():
    lanes = build_lane_configs(4)

    assert [lane.stream_id for lane in lanes] == [0, 1, 2, 3]
    assert all(set(vars(lane)) == {"stream_id"} for lane in lanes)


def test_offline_model_catalog_contains_q40_and_q4km():
    q40 = OFFLINE_MODELS["qwen3-asr-0.6b-q40"]
    q4km = OFFLINE_MODELS["qwen3-asr-1.7b-q4km"]

    assert q40.text_model_filename == "Qwen3-ASR-0.6B-text-q40.gguf"
    assert (q40.d_model, q40.hidden_size, q40.output_dim) == (896, 1024, 1024)
    assert q4km.text_model_filename == "Qwen3-ASR-1.7B-text-q4km.gguf"
    assert (q4km.d_model, q4km.hidden_size, q4km.output_dim) == (1024, 2048, 2048)


def test_app_config_expands_default_file_inputs_for_every_lane():
    config = AppConfig(lanes=2)

    assert config.input_for_lane(0) == LaneInputConfig(kind="default_files")
    assert config.input_for_lane(1) == LaneInputConfig(kind="default_files")


def test_lane_input_rejects_missing_local_path_and_invalid_kind():
    with pytest.raises(ValueError, match="local_wav requires local_wav_path"):
        LaneInputConfig(kind="local_wav")
    with pytest.raises(ValueError, match="unsupported lane input kind"):
        LaneInputConfig(kind="network")
    with pytest.raises(ValueError, match="microphone_device must be an integer or null"):
        LaneInputConfig(kind="microphone", microphone_device=True)
    assert LaneInputConfig(
        kind="microphone",
        microphone_device=21,
        microphone_device_id="alsa_input.usb-HM5, PulseAudio",
    ).microphone_device_id == "alsa_input.usb-HM5, PulseAudio"
    with pytest.raises(ValueError, match="microphone_device_id must be non-empty"):
        LaneInputConfig(kind="microphone", microphone_device_id=" ")
    assert LaneInputConfig(kind="default_files", loop_audio=True).loop_audio
    assert LaneInputConfig(
        kind="local_wav", local_wav_path=Path("input.wav"), loop_audio=True
    ).loop_audio
    with pytest.raises(ValueError, match="only file inputs accept loop_audio"):
        LaneInputConfig(kind="microphone", loop_audio=True)


def test_validate_offline_model_files_returns_all_missing_paths(tmp_path: Path):
    spec = replace(OFFLINE_MODELS["qwen3-asr-0.6b-q40"], model_dir=tmp_path)

    assert validate_offline_model_files(spec) == [
        tmp_path / name for name in spec.required_filenames
    ]


def test_validate_offline_model_files_returns_unreadable_required_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec = replace(OFFLINE_MODELS["qwen3-asr-0.6b-q40"], model_dir=tmp_path)
    unreadable_path = tmp_path / spec.frontend_filename
    for filename in spec.required_filenames:
        (tmp_path / filename).write_bytes(b"model")

    real_access = os.access
    monkeypatch.setattr(
        os,
        "access",
        lambda path, mode: False if Path(path) == unreadable_path and mode == os.R_OK else real_access(path, mode),
    )

    assert validate_offline_model_files(spec) == [unreadable_path]


def test_validate_offline_model_files_rejects_required_file_symlink_outside_model_directory(
    tmp_path: Path,
):
    model_directory = tmp_path / "models"
    external_directory = tmp_path / "external"
    model_directory.mkdir()
    external_directory.mkdir()
    spec = replace(OFFLINE_MODELS["qwen3-asr-0.6b-q40"], model_dir=model_directory)
    for filename in spec.required_filenames:
        (model_directory / filename).write_bytes(b"model")

    external_target = external_directory / spec.frontend_filename
    external_target.write_bytes(b"model")
    external_link = model_directory / spec.frontend_filename
    external_link.unlink()
    external_link.symlink_to(external_target)

    assert validate_offline_model_files(spec) == [external_link]
