from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier, Thread

import pytest

from multi_stream_asr.log.lane_artifacts import (
    DetailJsonlWriter,
    LaneModelOutputJsonlRecorder,
    LaneResultWriter,
)
from multi_stream_asr.log.model_output import ModelOutputRecord
from multi_stream_asr.models import ASREvent, EventStatus, ResultType


def test_lane_result_writer_appends_only_final_events(tmp_path: Path):
    writer = LaneResultWriter(tmp_path / "results" / "lane0.txt")

    writer.record_event(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=1,
            status=EventStatus.PENDING,
            text="draft",
            timestamp_ms=100,
        )
    )
    writer.record_event(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=0,
            revision=2,
            status=EventStatus.FINAL,
            text="final one",
            timestamp_ms=200,
        )
    )
    writer.record_event(
        ASREvent.asr_result(
            stream_id=0,
            sentence_id=1,
            revision=2,
            status=EventStatus.FINAL,
            text="",
            timestamp_ms=300,
        )
    )
    writer.close()

    assert (tmp_path / "results" / "lane0.txt").read_text(encoding="utf-8") == "final one\n\n"


def test_lane_model_output_jsonl_recorder_writes_compact_records(tmp_path: Path):
    recorder = LaneModelOutputJsonlRecorder(tmp_path / "model_outputs" / "lane1.jsonl")

    recorder.record(
        ModelOutputRecord(
            model="qwen3-asr",
            stream_id=1,
            sentence_id=2,
            status="final",
            text="recognized",
            audio_seconds=1.5,
            elapsed_ms=600.0,
            rtf=0.4,
            raw_response={"large": "not written"},
        )
    )
    recorder.close()

    lines = (tmp_path / "model_outputs" / "lane1.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {
        "audio_seconds": 1.5,
        "model": "qwen3-asr",
        "rtf": 0.4,
        "sentence_id": 2,
        "stream_id": 1,
        "text": "recognized",
    }


def test_lane_model_output_jsonl_recorder_handles_concurrent_records_and_idempotent_close(tmp_path: Path):
    path = tmp_path / "model_outputs" / "lane1.jsonl"
    recorder = LaneModelOutputJsonlRecorder(path)
    start_records = Barrier(3)

    def record_range(start: int) -> None:
        start_records.wait()
        for sentence_id in range(start, start + 50):
            recorder.record(
                ModelOutputRecord(
                    model="zipformer",
                    stream_id=1,
                    sentence_id=sentence_id,
                    status="final",
                    text=f"text-{sentence_id}",
                    audio_seconds=1.0,
                    rtf=0.2,
                )
            )

    record_threads = [Thread(target=record_range, args=(start,)) for start in (0, 50)]
    for thread in record_threads:
        thread.start()
    start_records.wait()
    for thread in record_threads:
        thread.join()

    start_closes = Barrier(3)

    def close_recorder() -> None:
        start_closes.wait()
        recorder.close()

    close_threads = [Thread(target=close_recorder) for _ in range(2)]
    for thread in close_threads:
        thread.start()
    start_closes.wait()
    for thread in close_threads:
        thread.join()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {record["sentence_id"] for record in records} == set(range(100))
    assert all(set(record) == {"model", "audio_seconds", "rtf", "stream_id", "sentence_id", "text"} for record in records)


def test_loop_artifact_writers_keep_files_empty(tmp_path: Path) -> None:
    result_writer = LaneResultWriter(tmp_path / "results.txt", enabled=False)
    output_writer = LaneModelOutputJsonlRecorder(tmp_path / "outputs.jsonl", enabled=False)
    event = ASREvent(
        event_type="asr_result",
        stream_id=0,
        sentence_id=0,
        revision=1,
        status=EventStatus.FINAL,
        result_type=ResultType.OFFLINE,
        is_final=True,
        text="should not be persisted",
        timestamp_ms=100,
    )
    result_writer.record_event(event)
    output_writer.record(
        ModelOutputRecord(
            model="qwen3-asr",
            stream_id=0,
            sentence_id=0,
            status="final",
            text="should not be persisted",
        )
    )
    result_writer.close()
    output_writer.close()

    assert (tmp_path / "results.txt").read_text() == ""
    assert (tmp_path / "outputs.jsonl").read_text() == ""


def test_detail_jsonl_writer_appends_t_ms_and_records_in_order(tmp_path: Path) -> None:
    clock_values = iter([10.0, 10.05, 10.2])
    writer = DetailJsonlWriter(
        tmp_path / "metrics_detail" / "frame_latency" / "lane0.jsonl",
        monotonic=lambda: next(clock_values),
    )

    writer.record({"latency_ms": 12.5})
    writer.record({"latency_ms": 8.0})
    writer.close()

    lines = [
        json.loads(line)
        for line in (tmp_path / "metrics_detail" / "frame_latency" / "lane0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(lines) == 2
    assert lines[0]["t_ms"] == pytest.approx(50.0)
    assert lines[0]["latency_ms"] == 12.5
    assert lines[1]["t_ms"] == pytest.approx(200.0)
    assert lines[1]["latency_ms"] == 8.0


def test_detail_jsonl_writer_close_is_idempotent_and_stops_writes(tmp_path: Path) -> None:
    writer = DetailJsonlWriter(tmp_path / "detail.jsonl")
    writer.record({"rtf": 0.4, "sentence_id": 0})
    writer.close()
    writer.close()
    writer.record({"rtf": 0.5, "sentence_id": 1})

    lines = (tmp_path / "detail.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["sentence_id"] == 0


def test_detail_jsonl_writer_handles_concurrent_records(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.jsonl"
    writer = DetailJsonlWriter(path)
    start = Barrier(3)

    def record_range(offset: int) -> None:
        start.wait()
        for i in range(offset, offset + 50):
            writer.record({"sentence_id": i})

    threads = [Thread(target=record_range, args=(offset,)) for offset in (0, 50)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()
    writer.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {record["sentence_id"] for record in records} == set(range(100))
