from __future__ import annotations

from typing import Any

from .metrics import binary_metrics


def cascade_decision(stage1_score: float, theta_1: float, stage2_score: float | None, theta_2: float | None) -> bool:
    candidate = stage1_score >= theta_1
    if not candidate:
        return False
    if stage2_score is None or theta_2 is None:
        return False
    return stage2_score >= theta_2


def evaluate_stage_predictions(rows: list[dict[str, Any]], threshold: float, score_key: str = "score") -> dict[str, Any]:
    return binary_metrics([int(r["label"]) for r in rows], [float(r[score_key]) for r in rows], threshold)
