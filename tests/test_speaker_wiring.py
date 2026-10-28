from __future__ import annotations

import queue
from pathlib import Path

import pytest

from multi_stream_asr.config import AppConfig, LaneConfig, SpeakerConfig
from multi_stream_asr.main import build_parser, build_speaker_config, parse_args
from multi_stream_asr.pipeline.lane_runtime import LaneProcessConfig, LaneRuntime
from multi_stream_asr.speaker import VoiceprintRegistry
from multi_stream_asr.web.transcript_buffer import TranscriptBuffer, render_transcript


# ---------- transcript 渲染 ----------


def test_render_plain_transcript_when_no_speaker() -> None:
    lines = [(None, None, "第一句"), (None, None, "第二句")]
    assert render_transcript(lines) == "第一句\n第二句"


def test_render_adds_speaker_prefix_and_header() -> None:
    lines = [("S1", None, "你好"), ("S2", None, "在的")]
    rendered = render_transcript(lines)

    assert rendered.splitlines() == [
        "# 说话人",
        "# S1（1 句）",
        "# S2（1 句）",
        "",
        "[S1] 你好",
        "[S2] 在的",
    ]


def test_render_includes_enrolled_name_in_header() -> None:
    lines = [("S1", "张三", "你好"), ("S1", "张三", "再说一句")]
    rendered = render_transcript(lines)

    assert "# S1 → 张三（2 句）" in rendered
    assert "[S1] 你好" in rendered


def test_render_sorts_speakers_numerically() -> None:
    lines = [("S10", None, "十"), ("S2", None, "二"), ("S1", None, "一")]
    header = [line for line in render_transcript(lines).splitlines() if line.startswith("# S")]

    assert header == ["# S1（1 句）", "# S2（1 句）", "# S10（1 句）"]


def test_render_applies_remap_atomically_for_swaps() -> None:
    lines = [("S1", None, "甲说的"), ("S2", None, "乙说的")]
    # 二遍重聚类判定两人标签互换了：必须一次性套用，不能链式替换
    rendered = render_transcript(lines, {"S1": "S2", "S2": "S1"})

    assert "[S2] 甲说的" in rendered
    assert "[S1] 乙说的" in rendered


def test_render_merges_when_remap_points_to_same_label() -> None:
    lines = [("S1", None, "一"), ("S3", None, "二")]
    rendered = render_transcript(lines, {"S3": "S1"})

    assert "# S1（2 句）" in rendered
    assert "[S1] 一" in rendered
    assert "[S1] 二" in rendered
    assert "S3" not in rendered


def test_render_keeps_unlabelled_lines_untouched() -> None:
    lines = [("S1", None, "有说话人"), (None, None, "没说话人")]
    rendered = render_transcript(lines)

    assert "[S1] 有说话人" in rendered
    assert rendered.rstrip().endswith("没说话人")


def test_render_empty_input() -> None:
    assert render_transcript([]) == ""


# ---------- CLI 开关 ----------


def test_speaker_disabled_by_default() -> None:
    assert SpeakerConfig().enabled is False
    config = build_speaker_config(parse_args([]))
    assert config.enabled is False


def test_speaker_flag_enables() -> None:
    config = build_speaker_config(parse_args(["--speaker"]))
    assert config.enabled is True


def test_speaker_model_path_implies_enable() -> None:
    config = build_speaker_config(parse_args(["--speaker-model", "/tmp/m.onnx"]))
    assert config.enabled is True
    assert config.model_path == Path("/tmp/m.onnx")


def test_speaker_registry_path_implies_enable() -> None:
    config = build_speaker_config(parse_args(["--speaker-registry", "/tmp/vp"]))
    assert config.enabled is True
    assert config.registry_dir == Path("/tmp/vp")


def test_parser_help_mentions_speaker() -> None:
    help_text = build_parser().format_help()
    assert "--speaker" in help_text
    assert "--speaker-model" in help_text


# ---------- lane 进程内声纹降级 ----------


def _make_runtime(config: AppConfig, *, fake: bool = False) -> LaneRuntime:
    return LaneRuntime(
        process_config=LaneProcessConfig(
            app_config=config,
            lane=LaneConfig(stream_id=0),
            fake=fake,
            streaming=True,
        ),
        event_queue=queue.Queue(),
        command_queue=queue.Queue(),
        audio_queue=queue.Queue(),
    )


def test_create_tracker_returns_none_when_disabled() -> None:
    runtime = _make_runtime(AppConfig())
    assert runtime._create_speaker_tracker() is None


def test_create_tracker_returns_none_when_model_missing(tmp_path: Path) -> None:
    config = AppConfig(
        speaker=SpeakerConfig(enabled=True, model_path=tmp_path / "nope.onnx")
    )
    runtime = _make_runtime(config)
    runtime._load_speaker()
    assert runtime._create_speaker_tracker() is None


def test_fake_runtime_never_creates_tracker(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"")
    config = AppConfig(speaker=SpeakerConfig(enabled=True, model_path=model))
    runtime = _make_runtime(config, fake=True)
    runtime._speaker_registry = VoiceprintRegistry(tmp_path / "vp")
    assert runtime._create_speaker_tracker() is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"new_speaker_threshold": 0.9, "match_threshold": 0.5},
        {"max_speakers": 0},
        {"trim_floor_ratio": 1.5},
    ],
)
def test_speaker_config_rejects_invalid(overrides: dict) -> None:
    with pytest.raises(ValueError):
        SpeakerConfig(**overrides)
