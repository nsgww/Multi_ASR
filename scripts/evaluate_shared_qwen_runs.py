from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_OUTPUT_KEYS = {
    "model",
    "audio_seconds",
    "rtf",
    "stream_id",
    "sentence_id",
    "text",
}
TIMING_FIELDS = ("queue_ms", "encode_ms", "prefill_ms", "decode_ms", "total_ms")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _valid_timings(value: object) -> bool:
    if not isinstance(value, dict) or any(field not in value for field in TIMING_FIELDS):
        return False
    required = {"count", "total_ms", "average_ms", "max_ms"}
    return all(
        isinstance(value[field], dict) and required.issubset(value[field])
        for field in TIMING_FIELDS
    )


def evaluate_run(path: Path) -> dict[str, Any]:
    path = Path(path)
    metrics = _load_json(path / "metrics.json")
    failures: list[str] = []
    if metrics.get("metrics_schema_version") != 3:
        failures.append(f"{path.name} metrics_schema_version is not 3")
    lanes = metrics.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 4:
        failures.append(f"{path.name} must contain exactly four lanes")
        lanes = []
    if not _valid_timings(metrics.get("offline_shared_timings")):
        failures.append(f"{path.name} missing offline_shared_timings")

    drops: list[int] = []
    for stream_id, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            failures.append(f"{path.name} lane{stream_id} is not an object")
            drops.append(-1)
            continue
        dropped = lane.get("dropped_frames")
        drops.append(int(dropped) if isinstance(dropped, int) else -1)
        if lane.get("stream_id") != stream_id:
            failures.append(f"{path.name} lane{stream_id} stream_id mismatch")
        if lane.get("final_state") != "finished":
            failures.append(f"{path.name} lane{stream_id} is not finished")
        if not _valid_timings(lane.get("offline_shared_timings")):
            failures.append(f"{path.name} lane{stream_id} missing shared timings")

        result_path = path / "results" / f"lane{stream_id}.txt"
        result_lines = (
            [line for line in result_path.read_text(encoding="utf-8").splitlines() if line]
            if result_path.is_file()
            else []
        )
        if not result_lines:
            failures.append(f"{path.name} lane{stream_id} result is empty")

        output_path = path / "model_outputs" / f"lane{stream_id}.jsonl"
        rows: list[dict[str, Any]] = []
        if output_path.is_file():
            try:
                rows = [
                    json.loads(line)
                    for line in output_path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
            except (json.JSONDecodeError, TypeError):
                failures.append(f"{path.name} lane{stream_id} model-output is invalid JSON")
        else:
            failures.append(f"{path.name} lane{stream_id} model-output is missing")
        if any(not isinstance(row, dict) or set(row) != MODEL_OUTPUT_KEYS for row in rows):
            failures.append(f"{path.name} lane{stream_id} model-output keys differ")
        valid_rows = [row for row in rows if isinstance(row, dict)]
        zip_ids = {
            row.get("sentence_id")
            for row in valid_rows
            if row.get("model") == "zipformer"
        }
        qwen_ids = {
            row.get("sentence_id")
            for row in valid_rows
            if row.get("model") == "qwen3-asr"
        }
        if zip_ids != qwen_ids:
            failures.append(f"{path.name} lane{stream_id} sentence IDs do not align")
        if len(result_lines) != len(qwen_ids):
            failures.append(f"{path.name} lane{stream_id} result/Qwen counts differ")

    return {
        "path": str(path),
        "metrics": metrics,
        "drops": drops,
        "failures": failures,
    }


def evaluate_runs(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError("exactly three run directories are required")
    runs = [evaluate_run(path) for path in paths]
    failures = [failure for run in runs for failure in run["failures"]]
    total = 0.0
    count = 0
    for run in runs:
        rtf = run["metrics"].get("offline_rtf", {})
        if isinstance(rtf, dict):
            total += float(rtf.get("total", 0.0))
            count += int(rtf.get("count", 0))
        for stream_id, dropped in enumerate(run["drops"]):
            if dropped >= 10:
                failures.append(
                    f"{Path(run['path']).name} lane{stream_id} dropped_frames={dropped}"
                )
            elif dropped < 0:
                failures.append(
                    f"{Path(run['path']).name} lane{stream_id} dropped_frames invalid"
                )
    combined = total / count if count else float("inf")
    if combined > 0.471:
        failures.append(f"combined offline_rtf={combined:.6f} exceeds 0.471")
    return {
        "passed": not failures,
        "combined_offline_rtf": combined,
        "offline_rtf_total": total,
        "offline_rtf_count": count,
        "runs": runs,
        "failures": failures,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 多进程 Zipformer + 单共享 Qwen3-ASR 验证报告",
        "",
        "## 拓扑",
        "",
        "1 个主 Python 进程、4 个 lane Python 子进程、1 个 llama-server。",
        "",
        "## 性能结论",
        "",
        f"- 验收：{'PASS' if result['passed'] else 'FAIL'}",
        f"- 三次合并 offline RTF：{result['combined_offline_rtf']:.6f}",
        f"- offline 请求数：{result['offline_rtf_count']}",
        "",
        "| Run | lane0 drops | lane1 drops | lane2 drops | lane3 drops | "
        "online avg/max ms | RTF avg/max | loop lag avg/max ms |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for run in result["runs"]:
        metrics = run["metrics"]
        online = metrics["online_frame_latency"]
        rtf = metrics["offline_rtf"]
        lag = metrics["event_loop_lag"]
        drops = run["drops"]
        lines.append(
            f"| {Path(run['path']).name} | {drops[0]} | {drops[1]} | "
            f"{drops[2]} | {drops[3]} | "
            f"{online['average_ms']:.3f}/{online['max_ms']:.3f} | "
            f"{rtf['average']:.6f}/{rtf['max']:.6f} | "
            f"{lag['average_ms']:.3f}/{lag['max_ms']:.3f} |"
        )
    lines.extend(["", "## 共享服务 timings", ""])
    for run in result["runs"]:
        shared = run["metrics"]["offline_shared_timings"]
        values = ", ".join(
            f"{field}={shared[field]['average_ms']:.3f}/"
            f"{shared[field]['max_ms']:.3f}ms"
            for field in TIMING_FIELDS
        )
        lines.append(f"- `{Path(run['path']).name}`: {values}")
    lines.extend(["", "## 功能检查", ""])
    lines.append("四路结果非空、model_outputs 六字段、Zipformer/Qwen 句 ID 对齐。")
    if result["failures"]:
        lines.extend(["", "## 失败项", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs=3, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_runs([path.resolve() for path in args.runs])
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
