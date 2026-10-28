from __future__ import annotations

import pytest

from multi_stream_asr.web.transcript_buffer import TranscriptBuffer, consume_event


def final_message(
    text: str,
    *,
    label: str | None = None,
    name: str | None = None,
    sentence_id: int = 0,
) -> dict:
    segment: dict = {"text": text}
    if label is not None:
        speaker = {"id": f"spk_{label[1:]}", "label": label, "confidence": 0.8}
        if name is not None:
            speaker["name"] = name
        segment["speaker"] = speaker
    return {
        "type": "transcript",
        "lane_id": 0,
        "sentence_id": sentence_id,
        "revision": 1,
        "is_partial": False,
        "is_final": True,
        "segment": segment,
    }


def test_buffer_collects_final_sentences() -> None:
    buffer = TranscriptBuffer()
    assert buffer.add_transcript(final_message("第一句")) is True
    assert buffer.add_transcript(final_message("第二句")) is True

    assert len(buffer) == 2
    assert bool(buffer) is True
    assert buffer.render() == "第一句\n第二句"


def test_buffer_ignores_partial_messages() -> None:
    buffer = TranscriptBuffer()
    message = dict(final_message("临时"), is_final=False, is_partial=True)

    assert buffer.add_transcript(message) is False
    assert len(buffer) == 0
    assert buffer.render() == ""


def test_buffer_ignores_empty_text() -> None:
    buffer = TranscriptBuffer()
    assert buffer.add_transcript(final_message("")) is False
    assert len(buffer) == 0


def test_buffer_ignores_missing_segment() -> None:
    buffer = TranscriptBuffer()
    assert buffer.add_transcript({"type": "transcript", "is_final": True}) is False


def test_buffer_records_speaker_label_and_name() -> None:
    buffer = TranscriptBuffer()
    buffer.add_transcript(final_message("你好", label="S1", name="张三"))

    assert buffer.lines == (("S1", "张三", "你好"),)
    assert "[S1] 你好" in buffer.render()
    assert "# S1 → 张三（1 句）" in buffer.render()


def test_buffer_remap_reorders_labels_atomically() -> None:
    buffer = TranscriptBuffer()
    buffer.add_transcript(final_message("甲说的", label="S1", sentence_id=0))
    buffer.add_transcript(final_message("乙说的", label="S2", sentence_id=1))

    assert buffer.add_speaker_remap({"mapping": {"S1": "S2", "S2": "S1"}}) is True
    rendered = buffer.render()

    assert "[S2] 甲说的" in rendered
    assert "[S1] 乙说的" in rendered


def test_buffer_rejects_empty_remap() -> None:
    buffer = TranscriptBuffer()
    assert buffer.add_speaker_remap({"mapping": {}}) is False
    assert buffer.add_speaker_remap({}) is False
    assert buffer.speaker_remap == {}


def test_buffer_reset_clears_everything() -> None:
    buffer = TranscriptBuffer()
    buffer.add_transcript(final_message("你好", label="S1"))
    buffer.add_speaker_remap({"mapping": {"S1": "S2"}})

    buffer.reset()

    assert len(buffer) == 0
    assert bool(buffer) is False
    assert buffer.speaker_remap == {}
    assert buffer.render() == ""


def test_consume_event_dispatches_both_types() -> None:
    buffer = TranscriptBuffer()
    consume_event(buffer, final_message("你好", label="S1"))
    consume_event(buffer, {"type": "speaker_remap", "mapping": {"S1": "S3"}})
    consume_event(buffer, {"type": "status", "status": "started"})

    assert len(buffer) == 1
    assert buffer.speaker_remap == {"S1": "S3"}
    assert "[S3] 你好" in buffer.render()


def test_rendered_output_is_stable_for_plain_text() -> None:
    """没开声纹时必须与改动前的纯文本格式逐字节一致。"""
    buffer = TranscriptBuffer()
    buffer.add_transcript(final_message("第一句"))
    buffer.add_transcript(final_message("第二句"))

    assert buffer.render() == "第一句\n第二句"


@pytest.mark.parametrize("label", ["S1", "S2", "S10"])
def test_consume_event_accepts_any_label(label: str) -> None:
    buffer = TranscriptBuffer()
    consume_event(buffer, final_message("文本", label=label))
    assert buffer.lines[0][0] == label
