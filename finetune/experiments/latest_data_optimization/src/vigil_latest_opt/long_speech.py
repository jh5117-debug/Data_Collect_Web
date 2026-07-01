from __future__ import annotations


def false_accepts_per_hour(false_accepts: int, total_seconds: float) -> float:
    if total_seconds <= 0:
        raise ValueError("total_seconds must be positive")
    return false_accepts / (total_seconds / 3600.0)


def sliding_window_count(duration_seconds: float, window_seconds: float, stride_seconds: float) -> int:
    if duration_seconds <= window_seconds:
        return 1
    return int((duration_seconds - window_seconds) // stride_seconds) + 1
