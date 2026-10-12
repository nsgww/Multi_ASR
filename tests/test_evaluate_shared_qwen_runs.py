from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_shared_qwen_runs import evaluate_runs


TIMING_FIELDS = ("queue_ms", "encode_ms", "prefill_ms", "decode_ms", "total_ms")


def timing_summary(value: float = 1.0) -> dict[str, object]:
    return {
        "count": 1,
        **{
            field: {
                "count": 1,
                "total_ms": value,
                "average_ms": value,
                "max_ms": value,
            }
            for field in TIMING_FIELDS
        },
    }


def make_run(
    path: Path,
    *,
    drops: list[int],
    rtf_total: float,
    rtf_count: int,
) -> Path:
    (path / "results").mkdir(parents=True)
    (path / "model_outputs").mkdir()
    lanes = []
    for stream_id, dropped in enumerate(drops):
        lane_rtf = rtf_total / rtf_count
        lanes.append(
            {
                "stream_id": stream_id,
                "processed_frames": 10,
                "dropped_frames": dropped,
                "offline_requests": 1,
                "final_state": "finished",
                "online_frame_latency": {
                    "count": 10,
                    "total_ms": 400.0,
                    "average_ms": 40.0,
                    "max_ms": 60.0,
                },
                "offline_rtf": {
                    "count": 1,
                    "total": lane_rtf,
                    "average": lane_rtf,
                    "max": lane_rtf,
                },
                "event_loop_lag": {
                    "count": 10,
                    "total_ms": 10.0,
                    "average_ms": 1.0,
                    "max_ms": 2.0,
                },
                "offline_shared_timings": timing_summary(),
                "results_path": f"results/lane{stream_id}.txt",
                "model_outputs_path": f"model_outputs/lane{stream_id}.jsonl",
            }
        )
        (path / "results" / f"lane{stream_id}.txt").write_text(
            "final text\n", encoding="utf-8"
        )
        records = [
            {
                "model": "zipformer",
                "audio_seconds": None,
                "rtf": None,
                "stream_id": stream_id,
                "sentence_id": 0,
                "text": "endpoint",
            },
            {
                "model": "qwen3-asr",
                "audio_seconds": 1.0,
                "rtf": lane_rtf,
                "stream_id": stream_id,
                "sentence_id": 0,
                "text": "final text",
            },
        ]
        output = path / "model_outputs" / f"lane{stream_id}.jsonl"
        output.write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
    metrics = {
        "metrics_schema_version": 3,
        "online_frame_latency": {
            "count": 40,
            "total_ms": 1600.0,
            "average_ms": 40.0,
            "max_ms": 60.0,
        },
        "offline_rtf": {
            "count": rtf_count,
            "total": rtf_total,
            "average": rtf_total / rtf_count,
            "max": rtf_total / rtf_count,
        },
        "event_loop_lag": {
            "count": 40,
            "total_ms": 40.0,
            "average_ms": 1.0,
            "max_ms": 2.0,
        },
        "offline_shared_timings": timing_summary(),
        "lanes": lanes,
    }
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return path


def make_three_runs(tmp_path: Path, *, rtf_total: float = 4.70) -> list[Path]:
    return [
        make_run(
            tmp_path / f"run{i}",
            drops=[1, 2, 3, 4],
            rtf_total=rtf_total,
            rtf_count=10,
        )
        for i in range(3)
    ]


def test_three_run_evaluator_accepts_strict_thresholds(tmp_path):
    result = evaluate_runs(make_three_runs(tmp_path))

    assert result["passed"] is True
    assert result["combined_offline_rtf"] == pytest.approx(0.47)


def test_evaluator_rejects_any_lane_with_ten_drops(tmp_path):
    runs = make_three_runs(tmp_path)
    metrics = json.loads((runs[1] / "metrics.json").read_text())
    metrics["lanes"][2]["dropped_frames"] = 10
    (runs[1] / "metrics.json").write_text(json.dumps(metrics))

    result = evaluate_runs(runs)

    assert result["passed"] is False
    assert "run1 lane2 dropped_frames=10" in result["failures"]


def test_evaluator_rejects_combined_rtf_above_point_471(tmp_path):
    result = evaluate_runs(make_three_runs(tmp_path, rtf_total=4.72))

    assert result["passed"] is False
    assert any("exceeds 0.471" in failure for failure in result["failures"])


def test_evaluator_rejects_misaligned_sentence_ids_and_extra_keys(tmp_path):
    runs = make_three_runs(tmp_path)
    output = runs[0] / "model_outputs" / "lane0.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    rows[1]["sentence_id"] = 7
    rows[1]["raw_response"] = {"forbidden": True}
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = evaluate_runs(runs)

    assert result["passed"] is False
    assert any("model-output keys differ" in failure for failure in result["failures"])
    assert any("sentence IDs do not align" in failure for failure in result["failures"])
