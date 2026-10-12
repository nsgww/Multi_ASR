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
RECENT_SAMPLE_LIMIT = 1024


def _kahan_add(total: float, compensation: float, value: float) -> tuple[float, float]:
    adjusted = value - compensation
    updated = total + adjusted
    return updated, (updated - total) - adjusted


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

    def _snapshot(self) -> tuple[int, float, float]:
        with self._lock:
            return self._count, self._total_ms, self._max_ms

    @property
    def count(self) -> int:
        count, _, _ = self._snapshot()
        return count

    @property
    def total_ms(self) -> float:
        _, total_ms, _ = self._snapshot()
        return total_ms

    @property
    def average_ms(self) -> float:
        count, total_ms, _ = self._snapshot()
        return total_ms / count if count else 0.0

    @property
    def max_ms(self) -> float:
        _, _, max_ms = self._snapshot()
        return max_ms

    @property
    def average(self) -> float:
        return self.average_ms

    def as_dict(self) -> dict[str, float | int]:
        count, total_ms, max_ms = self._snapshot()
        return {
            "count": count,
            "total_ms": total_ms,
            "average_ms": total_ms / count if count else 0.0,
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

    def _snapshot(self) -> tuple[int, float, float]:
        with self._lock:
            return self._count, self._total, self._max_value

    @property
    def count(self) -> int:
        count, _, _ = self._snapshot()
        return count

    @property
    def total(self) -> float:
        _, total, _ = self._snapshot()
        return total

    @property
    def average(self) -> float:
        count, total, _ = self._snapshot()
        return total / count if count else 0.0

    @property
    def max(self) -> float:
        _, _, max_value = self._snapshot()
        return max_value

    def as_dict(self) -> dict[str, float | int]:
        count, total, max_value = self._snapshot()
        return {
            "count": count,
            "total": total,
            "average": total / count if count else 0.0,
            "max": max_value,
        }


@dataclass
class SharedTimingsTracker:
    samples: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in SHARED_TIMING_FIELDS}
    )

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._field_counts = {key: 0 for key in SHARED_TIMING_FIELDS}
        self._field_totals = {key: 0.0 for key in SHARED_TIMING_FIELDS}
        self._field_compensations = {key: 0.0 for key in SHARED_TIMING_FIELDS}
        self._field_maxima = {key: 0.0 for key in SHARED_TIMING_FIELDS}
        self._count = max(
            (len(self.samples.get(key, [])) for key in SHARED_TIMING_FIELDS),
            default=0,
        )
        for key in SHARED_TIMING_FIELDS:
            self.samples[key] = list(self.samples.get(key, []))[-RECENT_SAMPLE_LIMIT:]
            for value in self.samples[key]:
                self._add_field_unlocked(key, float(value))

    def _add_field_unlocked(self, key: str, value: float) -> None:
        self._field_counts[key] += 1
        self._field_totals[key], self._field_compensations[key] = _kahan_add(
            self._field_totals[key], self._field_compensations[key], value
        )
        self._field_maxima[key] = max(self._field_maxima[key], value)

    def add(self, timings: dict[str, float]) -> None:
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
        with self._lock:
            self._count += 1
            for key, value in accepted.items():
                self._add_field_unlocked(key, value)
                self.samples[key].append(value)
                if len(self.samples[key]) > RECENT_SAMPLE_LIMIT:
                    del self.samples[key][:-RECENT_SAMPLE_LIMIT]

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            count = self._count
            field_stats = {
                key: (
                    self._field_counts[key],
                    self._field_totals[key],
                    self._field_maxima[key],
                )
                for key in SHARED_TIMING_FIELDS
            }
        payload: dict[str, Any] = {"count": count}
        for key, (field_count, total_ms, max_ms) in field_stats.items():
            payload[key] = {
                "count": field_count,
                "total_ms": total_ms,
                "average_ms": total_ms / field_count if field_count else 0.0,
                "max_ms": max_ms,
            }
        return payload


def merge_latency_summaries(summaries: list[dict[str, float | int]]) -> dict[str, float | int]:
    count = sum(int(summary.get("count", 0)) for summary in summaries)
    total_ms = float(sum(float(summary.get("total_ms", 0.0)) for summary in summaries))
    max_ms = max((float(summary.get("max_ms", 0.0)) for summary in summaries), default=0.0)
    return {
        "count": count,
        "total_ms": total_ms,
        "average_ms": total_ms / count if count else 0.0,
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
        "max": max_value,
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
            "max_ms": max_ms,
        }
    return merged
