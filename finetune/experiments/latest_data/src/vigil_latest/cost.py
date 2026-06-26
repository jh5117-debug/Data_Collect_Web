from __future__ import annotations


def false_accepts_per_hour(false_accepts: int, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    return float(false_accepts) * 3600.0 / float(duration_seconds)


def candidate_rate(candidates: int, total_windows: int) -> float:
    if total_windows <= 0:
        raise ValueError("total_windows must be positive")
    return float(candidates) / float(total_windows)
