from __future__ import annotations

from typing import Any


def false_activations_per_hour(false_accepts: int, total_audio_seconds: float) -> float:
    if total_audio_seconds <= 0:
        raise ValueError("total_audio_seconds must be positive")
    return float(false_accepts) / (float(total_audio_seconds) / 3600.0)


def window_count(duration_seconds: float, window_seconds: float, stride_seconds: float) -> int:
    if duration_seconds < window_seconds:
        return 1
    return int((duration_seconds - window_seconds) // stride_seconds) + 1


def summarize_stress(rows: list[dict[str, Any]], total_audio_seconds: float) -> dict[str, Any]:
    false_accepts = sum(1 for row in rows if bool(row.get("final_trigger")))
    total_windows = sum(int(row.get("windows", 0)) for row in rows)
    stage1_candidates = sum(int(row.get("stage1_candidates", 0)) for row in rows)
    return {
        "utterances": len(rows),
        "total_audio_hours": total_audio_seconds / 3600.0,
        "total_windows": total_windows,
        "stage1_candidates": stage1_candidates,
        "stage1_candidates_per_hour": false_activations_per_hour(stage1_candidates, total_audio_seconds),
        "final_false_accepts": false_accepts,
        "final_false_accepts_per_hour": false_activations_per_hour(false_accepts, total_audio_seconds),
    }
