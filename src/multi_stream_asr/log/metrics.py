from __future__ import annotations

import threading
import math
from dataclasses import dataclass, field
from typing import Any


SHARED_TIMING_FIELDS = (
    "queue_ms",
    "encode_ms",
    "prefill_ms",
    "decode_ms",
    "total_ms",
)
# Per-request stage RTF fields: stage milliseconds divided by that request's input
# audio duration in milliseconds, then averaged per request.  The end-to-end
# per-request RTF is already reported as ``offline_rtf`` (client-observed), so no
# ``total_rtf`` counterpart is emitted here: that would only duplicate it.
STAGE_RTF_FIELDS = (
    "queue_rtf",
    "encode_rtf",
    "prefill_rtf",
    "decode_rtf",
)
_STAGE_RTF_TIMING_FIELDS = ("queue_ms", "encode_ms", "prefill_ms", "decode_ms")
_TIMING_FIELD_TO_RTF_FIELD = {
    timing_field: f"{timing_field[:-3]}_rtf" for timing_field in _STAGE_RTF_TIMING_FIELDS
}
RECENT_SAMPLE_LIMIT = 1024
P95_QUANTILE = 0.95
# Whether queue wait grows as recognition proceeds.  The encoder is serial, so a
# queue that only ever gets longer is the signature of a workload the server
# cannot keep up with; a steady average queue_ms hides it completely.  Computed
# from the queue_ms samples in request order, so it is bounded by the same
# recent-sample window as the P95.
QUEUE_DRIFT_FIELD = "queue_drift"
# Bumped to 4 when every metric block gained a windowed P95 and the shared
# timings block gained per-request stage RTFs plus input audio duration, and to
# 5 when it gained the queue drift block.
METRICS_SCHEMA_VERSION = 5


def _kahan_add(total: float, compensation: float, value: float) -> tuple[float, float]:
    adjusted = value - compensation
    updated = total + adjusted
    return updated, (updated - total) - adjusted


def _nearest_rank_percentile(
    values: list[float],
    quantile: float = P95_QUANTILE,
) -> float:
    """Nearest-rank percentile of the retained recent samples (0.0 when empty).

    The samples are the bounded ``RECENT_SAMPLE_LIMIT`` window, so the result is
    a windowed percentile and not a percentile over the whole run.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(quantile * len(ordered))
    index = min(max(rank, 1), len(ordered)) - 1
    return float(ordered[index])


def _finite_positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _queue_drift(values: list[float]) -> dict[str, float | int]:
    """How queue wait moves as recognition proceeds, in request order.

    ``values`` is the retained recent-sample window in the order the requests
    completed.  A shared serial encoder that slowly falls behind shows a
    positive ``slope_ms_per_request`` and ``ratio`` above 1 here even while the
    run-average ``queue_ms`` looks flat, which is exactly the signal a mean
    cannot carry.  ``first_ms``/``last_ms`` are the means of the older and newer
    half of the window.  Fewer than two samples yields zeros.

    ``ratio`` is ``last_ms / first_ms`` and is only meaningful while
    ``first_ms`` is positive; when the older half waited no time at all (or
    there are fewer than two samples) it is reported as ``0.0`` instead of a
    division by zero, and ``slope_ms_per_request`` together with ``first_ms``/
    ``last_ms`` remains the authoritative reading.
    """
    count = len(values)
    if count < 2:
        return {
            "count": count,
            "first_ms": 0.0,
            "last_ms": 0.0,
            "slope_ms_per_request": 0.0,
            "ratio": 0.0,
        }
    half = count // 2
    first_ms = sum(values[:half]) / half
    last_ms = sum(values[-half:]) / half
    length = float(count)
    mean_x = (length - 1) / 2.0
    mean_y = sum(values) / length
    numerator = 0.0
    denominator = 0.0
    for index, value in enumerate(values):
        delta = index - mean_x
        numerator += delta * (value - mean_y)
        denominator += delta * delta
    return {
        "count": count,
        "first_ms": first_ms,
        "last_ms": last_ms,
        "slope_ms_per_request": numerator / denominator if denominator else 0.0,
        "ratio": last_ms / first_ms if first_ms > 0 else 0.0,
    }


@dataclass
class LatencyTracker:
    samples_ms: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        initial = [float(value) for value in self.samples_ms]
        self.samples_ms = initial[-RECENT_SAMPLE_LIMIT:]
        self._count = 0
        self._total_ms = 0.0
        self._total_compensation = 0.0
        self._max_ms = 0.0
        for value in initial:
            self._add_unlocked(value)

    def _add_unlocked(self, value: float) -> None:
        self._count += 1
        self._total_ms, self._total_compensation = _kahan_add(
            self._total_ms, self._total_compensation, value
        )
        self._max_ms = max(self._max_ms, value)

    def add(self, duration_ms: float) -> None:
        value = float(duration_ms)
        with self._lock:
            self._add_unlocked(value)
            self.samples_ms.append(value)
            if len(self.samples_ms) > RECENT_SAMPLE_LIMIT:
                del self.samples_ms[:-RECENT_SAMPLE_LIMIT]

    def _snapshot(self) -> tuple[int, float, float, float]:
        """One consistent view: count/total/max over the run, P95 over samples."""
        with self._lock:
            return (
                self._count,
                self._total_ms,
                self._max_ms,
                _nearest_rank_percentile(self.samples_ms),
            )

    def _count_snapshot(self) -> tuple[int, float, float]:
        with self._lock:
            return self._count, self._total_ms, self._max_ms

    @property
    def count(self) -> int:
        count, _, _ = self._count_snapshot()
        return count

    @property
    def total_ms(self) -> float:
        _, total_ms, _ = self._count_snapshot()
        return total_ms

    @property
    def average_ms(self) -> float:
        count, total_ms, _ = self._count_snapshot()
        return total_ms / count if count else 0.0

    @property
    def max_ms(self) -> float:
        _, _, max_ms = self._count_snapshot()
        return max_ms

    @property
    def p95_ms(self) -> float:
        _, _, _, p95_ms = self._snapshot()
        return p95_ms

    @property
    def average(self) -> float:
        return self.average_ms

    def as_dict(self) -> dict[str, float | int]:
        count, total_ms, max_ms, p95_ms = self._snapshot()
        return {
            "count": count,
            "total_ms": total_ms,
            "average_ms": total_ms / count if count else 0.0,
            "p95_ms": p95_ms,
            "max_ms": max_ms,
        }


class EventLoopLagTracker(LatencyTracker):
    pass


@dataclass
class RtfTracker:
    values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        initial = [float(value) for value in self.values]
        self.values = initial[-RECENT_SAMPLE_LIMIT:]
        self._count = 0
        self._total = 0.0
        self._total_compensation = 0.0
        self._max_value = 0.0
        for value in initial:
            self._add_unlocked(value)

    def _add_unlocked(self, value: float) -> None:
        self._count += 1
        self._total, self._total_compensation = _kahan_add(
            self._total, self._total_compensation, value
        )
        self._max_value = max(self._max_value, value)

    def add(self, audio_seconds: float, elapsed_ms: float) -> None:
        value = (
            0.0
            if audio_seconds <= 0
            else (float(elapsed_ms) / 1000.0) / float(audio_seconds)
        )
        with self._lock:
            self._add_unlocked(value)
            self.values.append(value)
            if len(self.values) > RECENT_SAMPLE_LIMIT:
                del self.values[:-RECENT_SAMPLE_LIMIT]

    def _snapshot(self) -> tuple[int, float, float, float]:
        """One consistent view: count/total/max over the run, P95 over samples."""
        with self._lock:
            return (
                self._count,
                self._total,
                self._max_value,
                _nearest_rank_percentile(self.values),
            )

    def _count_snapshot(self) -> tuple[int, float, float]:
        with self._lock:
            return self._count, self._total, self._max_value

    @property
    def count(self) -> int:
        count, _, _ = self._count_snapshot()
        return count

    @property
    def total(self) -> float:
        _, total, _ = self._count_snapshot()
        return total

    @property
    def average(self) -> float:
        count, total, _ = self._count_snapshot()
        return total / count if count else 0.0

    @property
    def max(self) -> float:
        _, _, max_value = self._count_snapshot()
        return max_value

    @property
    def p95(self) -> float:
        _, _, _, p95 = self._snapshot()
        return p95

    def as_dict(self) -> dict[str, float | int]:
        count, total, max_value, p95 = self._snapshot()
        return {
            "count": count,
            "total": total,
            "average": total / count if count else 0.0,
            "p95": p95,
            "max": max_value,
        }


@dataclass
class SharedTimingsTracker:
    samples: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in SHARED_TIMING_FIELDS}
    )
    stage_rtf_samples: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in STAGE_RTF_FIELDS}
    )
    audio_seconds_samples: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._field_counts = {key: 0 for key in SHARED_TIMING_FIELDS}
        self._field_totals = {key: 0.0 for key in SHARED_TIMING_FIELDS}
        self._field_compensations = {key: 0.0 for key in SHARED_TIMING_FIELDS}
        self._field_maxima = {key: 0.0 for key in SHARED_TIMING_FIELDS}
        self._stage_rtf_counts = {key: 0 for key in STAGE_RTF_FIELDS}
        self._stage_rtf_totals = {key: 0.0 for key in STAGE_RTF_FIELDS}
        self._stage_rtf_compensations = {key: 0.0 for key in STAGE_RTF_FIELDS}
        self._stage_rtf_maxima = {key: 0.0 for key in STAGE_RTF_FIELDS}
        self._audio_count = 0
        self._audio_total = 0.0
        self._audio_compensation = 0.0
        self._audio_max = 0.0
        self._count = max(
            (len(self.samples.get(key, [])) for key in SHARED_TIMING_FIELDS),
            default=0,
        )
        for key in SHARED_TIMING_FIELDS:
            self.samples[key] = list(self.samples.get(key, []))[-RECENT_SAMPLE_LIMIT:]
            for value in self.samples[key]:
                self._add_field_unlocked(key, float(value))
        for key in STAGE_RTF_FIELDS:
            self.stage_rtf_samples[key] = list(self.stage_rtf_samples.get(key, []))[
                -RECENT_SAMPLE_LIMIT:
            ]
            for value in self.stage_rtf_samples[key]:
                self._add_stage_rtf_unlocked(key, float(value))
        self.audio_seconds_samples = list(self.audio_seconds_samples)[-RECENT_SAMPLE_LIMIT:]
        for value in self.audio_seconds_samples:
            self._add_audio_seconds_unlocked(float(value))

    def _add_field_unlocked(self, key: str, value: float) -> None:
        self._field_counts[key] += 1
        self._field_totals[key], self._field_compensations[key] = _kahan_add(
            self._field_totals[key], self._field_compensations[key], value
        )
        self._field_maxima[key] = max(self._field_maxima[key], value)

    def _add_stage_rtf_unlocked(self, key: str, value: float) -> None:
        self._stage_rtf_counts[key] += 1
        self._stage_rtf_totals[key], self._stage_rtf_compensations[key] = _kahan_add(
            self._stage_rtf_totals[key], self._stage_rtf_compensations[key], value
        )
        self._stage_rtf_maxima[key] = max(self._stage_rtf_maxima[key], value)

    def _add_audio_seconds_unlocked(self, value: float) -> None:
        self._audio_count += 1
        self._audio_total, self._audio_compensation = _kahan_add(
            self._audio_total, self._audio_compensation, value
        )
        self._audio_max = max(self._audio_max, value)

    def add(self, timings: dict[str, float], *, audio_seconds: float | None = None) -> None:
        accepted = {
            key: float(value)
            for key, value in timings.items()
            if key in SHARED_TIMING_FIELDS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        }
        if not accepted:
            return
        audio = _finite_positive(audio_seconds)
        with self._lock:
            self._count += 1
            for key, value in accepted.items():
                self._add_field_unlocked(key, value)
                self.samples[key].append(value)
                if len(self.samples[key]) > RECENT_SAMPLE_LIMIT:
                    del self.samples[key][:-RECENT_SAMPLE_LIMIT]
                rtf_field = _TIMING_FIELD_TO_RTF_FIELD.get(key)
                if rtf_field is not None and audio is not None:
                    self._append_stage_rtf_unlocked(rtf_field, (value / 1000.0) / audio)
            if audio is not None:
                self._add_audio_seconds_unlocked(audio)
                self.audio_seconds_samples.append(audio)
                if len(self.audio_seconds_samples) > RECENT_SAMPLE_LIMIT:
                    del self.audio_seconds_samples[:-RECENT_SAMPLE_LIMIT]

    def _append_stage_rtf_unlocked(self, key: str, value: float) -> None:
        self._add_stage_rtf_unlocked(key, value)
        self.stage_rtf_samples[key].append(value)
        if len(self.stage_rtf_samples[key]) > RECENT_SAMPLE_LIMIT:
            del self.stage_rtf_samples[key][:-RECENT_SAMPLE_LIMIT]

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            count = self._count
            field_stats = {
                key: (
                    self._field_counts[key],
                    self._field_totals[key],
                    self._field_maxima[key],
                    _nearest_rank_percentile(self.samples[key]),
                )
                for key in SHARED_TIMING_FIELDS
            }
            stage_rtf_stats = {
                key: (
                    self._stage_rtf_counts[key],
                    self._stage_rtf_totals[key],
                    self._stage_rtf_maxima[key],
                    _nearest_rank_percentile(self.stage_rtf_samples[key]),
                )
                for key in STAGE_RTF_FIELDS
            }
            drift = _queue_drift(self.samples["queue_ms"])
            audio_stats = (
                self._audio_count,
                self._audio_total,
                self._audio_max,
                _nearest_rank_percentile(self.audio_seconds_samples),
            )
        payload: dict[str, Any] = {"count": count}
        for key, (field_count, total_ms, max_ms, p95_ms) in field_stats.items():
            payload[key] = {
                "count": field_count,
                "total_ms": total_ms,
                "average_ms": total_ms / field_count if field_count else 0.0,
                "p95_ms": p95_ms,
                "max_ms": max_ms,
            }
        for key, (field_count, total, max_value, p95) in stage_rtf_stats.items():
            payload[key] = {
                "count": field_count,
                "total": total,
                "average": total / field_count if field_count else 0.0,
                "p95": p95,
                "max": max_value,
            }
        audio_count, audio_total, audio_max, audio_p95 = audio_stats
        payload["audio_seconds"] = {
            "count": audio_count,
            "total": audio_total,
            "average": audio_total / audio_count if audio_count else 0.0,
            "p95": audio_p95,
            "max": audio_max,
        }
        payload[QUEUE_DRIFT_FIELD] = drift
        return payload


def _merged_p95(values: list[dict[str, Any]], key: str) -> float:
    """Upper bound of the pooled P95 across lanes.

    Each lane keeps only its own bounded recent-sample window, so the exact
    pooled percentile cannot be recovered after the fact.  The worst lane's P95
    is used as a conservative upper bound and is paired with ``max`` (which is
    exact because it is accumulated over the whole run).
    """
    return max((float(value.get(key, 0.0)) for value in values), default=0.0)


def merge_latency_summaries(summaries: list[dict[str, float | int]]) -> dict[str, float | int]:
    count = sum(int(summary.get("count", 0)) for summary in summaries)
    total_ms = float(sum(float(summary.get("total_ms", 0.0)) for summary in summaries))
    max_ms = max((float(summary.get("max_ms", 0.0)) for summary in summaries), default=0.0)
    p95_ms = _merged_p95(list(summaries), "p95_ms")
    return {
        "count": count,
        "total_ms": total_ms,
        "average_ms": total_ms / count if count else 0.0,
        "p95_ms": p95_ms,
        "max_ms": max_ms,
    }


def merge_rtf_summaries(summaries: list[dict[str, float | int]]) -> dict[str, float | int]:
    count = sum(int(summary.get("count", 0)) for summary in summaries)
    total = float(sum(float(summary.get("total", 0.0)) for summary in summaries))
    max_value = max((float(summary.get("max", 0.0)) for summary in summaries), default=0.0)
    return {
        "count": count,
        "total": total,
        "average": total / count if count else 0.0,
        "p95": _merged_p95(list(summaries), "p95"),
        "max": max_value,
    }


def _merge_queue_drift(summaries: list[dict[str, Any]]) -> dict[str, float | int]:
    """Pool the lane queue-drift blocks.

    ``first_ms``/``last_ms`` are averaged with the sample counts behind them, so
    the merged ``ratio`` is the pooled newer-half mean over the pooled older-half
    mean.  ``slope_ms_per_request`` is the count-weighted mean of the lane
    slopes: like ``_merged_p95`` this is an approximation, because a lane ships
    only its summary and not the per-request samples a pooled regression needs.
    Only lanes that saw at least two requests can report a trend, so the merged
    ``count`` is the number of samples actually behind ``first_ms``/``last_ms``
    and may be smaller than the ``queue_ms`` sample count; when no lane has a
    usable drift block every value is zero, matching a fresh tracker.
    """
    blocks = [
        value
        for summary in summaries
        if isinstance((value := summary.get(QUEUE_DRIFT_FIELD)), dict)
        and int(value.get("count", 0)) >= 2
    ]
    if not blocks:
        return {
            "count": 0,
            "first_ms": 0.0,
            "last_ms": 0.0,
            "slope_ms_per_request": 0.0,
            "ratio": 0.0,
        }
    halves = [max(int(block.get("count", 0)) // 2, 1) for block in blocks]
    total_half = sum(halves)
    first_ms = sum(
        float(block.get("first_ms", 0.0)) * half for block, half in zip(blocks, halves)
    ) / total_half
    last_ms = sum(
        float(block.get("last_ms", 0.0)) * half for block, half in zip(blocks, halves)
    ) / total_half
    weights = [max(int(block.get("count", 0)), 1) for block in blocks]
    slope = sum(
        float(block.get("slope_ms_per_request", 0.0)) * weight
        for block, weight in zip(blocks, weights)
    ) / sum(weights)
    return {
        "count": sum(int(block.get("count", 0)) for block in blocks),
        "first_ms": first_ms,
        "last_ms": last_ms,
        "slope_ms_per_request": slope,
        "ratio": last_ms / first_ms if first_ms > 0 else 0.0,
    }


def merge_shared_timing_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "count": sum(int(summary.get("count", 0)) for summary in summaries)
    }
    for key in SHARED_TIMING_FIELDS:
        field_summaries = [
            value
            for summary in summaries
            if isinstance((value := summary.get(key)), dict)
        ]
        count = sum(int(value.get("count", 0)) for value in field_summaries)
        total_ms = float(
            sum(float(value.get("total_ms", 0.0)) for value in field_summaries)
        )
        max_ms = max(
            (float(value.get("max_ms", 0.0)) for value in field_summaries),
            default=0.0,
        )
        merged[key] = {
            "count": count,
            "total_ms": total_ms,
            "average_ms": total_ms / count if count else 0.0,
            "p95_ms": _merged_p95(field_summaries, "p95_ms"),
            "max_ms": max_ms,
        }
    for key in STAGE_RTF_FIELDS:
        stage_summaries = [
            value
            for summary in summaries
            if isinstance((value := summary.get(key)), dict)
        ]
        count = sum(int(value.get("count", 0)) for value in stage_summaries)
        total = float(sum(float(value.get("total", 0.0)) for value in stage_summaries))
        merged[key] = {
            "count": count,
            "total": total,
            "average": total / count if count else 0.0,
            "p95": _merged_p95(stage_summaries, "p95"),
            "max": max(
                (float(value.get("max", 0.0)) for value in stage_summaries),
                default=0.0,
            ),
        }
    audio_summaries = [
        value
        for summary in summaries
        if isinstance((value := summary.get("audio_seconds")), dict)
    ]
    audio_count = sum(int(value.get("count", 0)) for value in audio_summaries)
    audio_total = float(sum(float(value.get("total", 0.0)) for value in audio_summaries))
    merged["audio_seconds"] = {
        "count": audio_count,
        "total": audio_total,
        "average": audio_total / audio_count if audio_count else 0.0,
        "p95": _merged_p95(audio_summaries, "p95"),
        "max": max(
            (float(value.get("max", 0.0)) for value in audio_summaries), default=0.0
        ),
    }
    merged[QUEUE_DRIFT_FIELD] = _merge_queue_drift(summaries)
    return merged
