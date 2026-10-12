from __future__ import annotations

import io

from multi_stream_asr.log.model_output import ModelOutputRecord, ModelOutputRecorder


def test_model_output_recorder_prints_concise_line_and_stores_record():
    stream = io.StringIO()
    recorder = ModelOutputRecorder(stream=stream)
    record = ModelOutputRecord(
        model="qwen3-asr",
        stream_id=2,
        sentence_id=5,
        status="final",
        text="这是模型识别出来的一段文字",
        audio_seconds=1.25,
        elapsed_ms=500.0,
        rtf=0.4,
        raw_response={"text": "<asr_text>这是模型识别出来的一段文字</asr_text>"},
    )

    recorder.record(record)

    printed = stream.getvalue()
    assert "[model-output]" in printed
    assert "model=qwen3-asr" in printed
    assert "lane=2" in printed
    assert "sentence=5" in printed
    assert "status=final" in printed
    assert "text_len=13" in printed
    assert "rtf=0.400" in printed
    assert "这是模型识别出来的一段文字" in printed

    stored = recorder.as_dicts()
    assert stored == [
        {
            "model": "qwen3-asr",
            "stream_id": 2,
            "sentence_id": 5,
            "status": "final",
            "text": "这是模型识别出来的一段文字",
            "text_preview": "这是模型识别出来的一段文字",
            "audio_seconds": 1.25,
            "elapsed_ms": 500.0,
            "rtf": 0.4,
            "raw_response": {"text": "<asr_text>这是模型识别出来的一段文字</asr_text>"},
        }
    ]


def test_model_output_recorder_can_store_without_printing():
    stream = io.StringIO()
    recorder = ModelOutputRecorder(stream=stream, emit_stdout=False)

    recorder.record(
        ModelOutputRecord(
            model="zipformer",
            stream_id=1,
            sentence_id=0,
            status="endpoint",
            text="online endpoint",
        )
    )

    assert stream.getvalue() == ""
    assert recorder.as_dicts()[0]["text"] == "online endpoint"
