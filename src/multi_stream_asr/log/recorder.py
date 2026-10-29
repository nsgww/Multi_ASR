from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from multi_stream_asr.config import AppConfig
from multi_stream_asr.log.metrics import (
    METRICS_SCHEMA_VERSION,
    EventLoopLagTracker,
    LatencyTracker,
    RtfTracker,
    SharedTimingsTracker,
    merge_latency_summaries,
    merge_rtf_summaries,
    merge_shared_timing_summaries,
)
from multi_stream_asr.models import LaneMetrics
from multi_stream_asr.utils.paths import ensure_dir


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _status_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _write_lane_results(output_dir: Path, *, config: AppConfig, results: Any) -> None:
    stale_path = output_dir / "results.json"
    if stale_path.exists():
        stale_path.unlink()

    results_dir = ensure_dir(output_dir / "results")
    final_texts: dict[int, list[tuple[int, str]]] = {lane: [] for lane in range(config.lanes)}
    if isinstance(results, dict):
        for key, value in results.items():
            if not isinstance(value, dict):
                continue
            if _status_value(value.get("status")) != "final":
                continue
            stream_id = value.get("stream_id")
            sentence_id = value.get("sentence_id")
            if (not isinstance(stream_id, int) or not isinstance(sentence_id, int)) and (
                isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[0], int)
                and isinstance(key[1], int)
            ):
                stream_id, sentence_id = key
            text = value.get("text", "")
            if not isinstance(stream_id, int) or not isinstance(sentence_id, int) or not isinstance(text, str):
                continue
            if stream_id not in final_texts:
                continue
            final_texts[stream_id].append((sentence_id, text))

    for stream_id in range(config.lanes):
        lines = [text for _, text in sorted(final_texts[stream_id], key=lambda item: item[0])]
        content = "".join(f"{line}\n" for line in lines)
        (results_dir / f"lane{stream_id}.txt").write_text(content, encoding="utf-8")


def _compact_model_output(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": record.get("model"),
        "stream_id": record.get("stream_id"),
        "sentence_id": record.get("sentence_id"),
        "audio_seconds": record.get("audio_seconds"),
        "rtf": record.get("rtf"),
        "text": record.get("text", ""),
    }


def _write_lane_model_outputs(output_dir: Path, *, config: AppConfig, model_outputs: Any) -> None:
    stale_path = output_dir / "model_outputs.jsonl"
    if stale_path.exists():
        stale_path.unlink()

    outputs_dir = ensure_dir(output_dir / "model_outputs")
    by_lane: dict[int, list[dict[str, Any]]] = {lane: [] for lane in range(config.lanes)}
    for record in _jsonable(model_outputs):
        if not isinstance(record, dict):
            continue
        stream_id = record.get("stream_id")
        if not isinstance(stream_id, int) or stream_id not in by_lane:
            continue
        by_lane[stream_id].append(_compact_model_output(record))

    for stream_id in range(config.lanes):
        with (outputs_dir / f"lane{stream_id}.jsonl").open("w", encoding="utf-8") as file:
            for record in by_lane[stream_id]:
                file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                file.write("\n")


class RunRecorder:
    def save(
        self,
        run_dir: str | Path,
        *,
        config: AppConfig,
        lane_metrics: list[LaneMetrics] | tuple[LaneMetrics, ...] = (),
        latency_tracker: LatencyTracker | None = None,
        rtf_tracker: RtfTracker | None = None,
        event_loop_lag_tracker: EventLoopLagTracker | None = None,
        results: Any = (),
        model_outputs: Any = None,
    ) -> Path:
        output_dir = ensure_dir(run_dir)

        config_payload = _jsonable(config)
        config_payload["segment_cache_seconds"] = config.segment_cache_seconds
        (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True))

        metrics_payload = {
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "online_frame_latency": (
                latency_tracker.as_dict() if latency_tracker is not None else LatencyTracker().as_dict()
            ),
            "offline_rtf": rtf_tracker.as_dict() if rtf_tracker is not None else RtfTracker().as_dict(),
            "offline_shared_timings": SharedTimingsTracker().as_dict(),
            "event_loop_lag": (
                event_loop_lag_tracker.as_dict()
                if event_loop_lag_tracker is not None
                else EventLoopLagTracker().as_dict()
            ),
            "lanes": _jsonable(lane_metrics),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))

        _write_lane_results(output_dir, config=config, results=results)
        if model_outputs is not None:
            _write_lane_model_outputs(output_dir, config=config, model_outputs=model_outputs)
        return output_dir

    def save_process_run(
        self,
        run_dir: str | Path,
        *,
        config: AppConfig,
        lane_summaries: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> Path:
        output_dir = ensure_dir(run_dir)
        ensure_dir(output_dir / "results")
        ensure_dir(output_dir / "model_outputs")

        config_payload = _jsonable(config)
        config_payload["segment_cache_seconds"] = config.segment_cache_seconds
        (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True))

        lanes: list[dict[str, Any]] = []
        for summary in lane_summaries:
            lane = dict(_jsonable(summary))
            lane_metrics = dict(lane.pop("lane_metrics", {}))
            lanes.append({**lane_metrics, **lane})

        metrics_payload = {
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "online_frame_latency": merge_latency_summaries(
                [dict(lane.get("online_frame_latency", {})) for lane in lanes]
            ),
            "offline_rtf": merge_rtf_summaries([dict(lane.get("offline_rtf", {})) for lane in lanes]),
            "offline_shared_timings": merge_shared_timing_summaries(
                [dict(lane.get("offline_shared_timings", {})) for lane in lanes]
            ),
            "event_loop_lag": merge_latency_summaries(
                [dict(lane.get("event_loop_lag", {})) for lane in lanes]
            ),
            "lanes": lanes,
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
        return output_dir
