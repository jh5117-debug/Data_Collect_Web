from __future__ import annotations

from typing import Any

import numpy as np

from .cascade import apply_threshold
from .metrics import group_metrics, metrics_from_rows


def hard_negative(row: dict[str, Any]) -> bool:
    if int(row.get("label", 0)) != 0:
        return False
    text = str(row.get("transcript", "")).lower()
    hard_words = ("visual", "visuals", "visible", "digital", "individual", "vigilant", "video", "vital", "residual")
    return any(word in text for word in hard_words) or str(row.get("prompt_group", "")).startswith("P4")


def threshold_grid(scores: list[float]) -> list[float]:
    finite = sorted({float(score) for score in scores if float(score) > -1e8})
    if not finite:
        return [0.5]
    lo, hi = min(finite), max(finite)
    grid = set(finite)
    grid.update(float(x) for x in np.linspace(lo, hi, 401))
    return sorted(grid)


def threshold_for_recall_target(rows: list[dict[str, Any]], recall_target: float) -> dict[str, Any]:
    if not rows:
        return {"threshold": 0.5, "reason": "no_rows", "metrics": {"n": 0}}
    feasible: list[tuple[float, dict[str, Any]]] = []
    all_points: list[tuple[float, dict[str, Any]]] = []
    for threshold in threshold_grid([float(row["score"]) for row in rows]):
        decisions = apply_threshold(rows, threshold)
        metrics = metrics_from_rows(decisions)
        all_points.append((threshold, metrics))
        recall = metrics.get("recall")
        if recall is not None and float(recall) >= recall_target:
            feasible.append((threshold, metrics))
    if feasible:
        threshold, metrics = sorted(
            feasible,
            key=lambda item: (
                item[1].get("false_positive_rate") if item[1].get("false_positive_rate") is not None else 1.0,
                -(item[1].get("precision") if item[1].get("precision") is not None else 0.0),
                -(item[1].get("f1") if item[1].get("f1") is not None else 0.0),
                -item[0],
            ),
        )[0]
        return {"threshold": float(threshold), "reason": f"met_recall_target_{recall_target}", "metrics": metrics}
    threshold, metrics = sorted(
        all_points,
        key=lambda item: (
            -(item[1].get("recall") if item[1].get("recall") is not None else -1.0),
            item[1].get("false_positive_rate") if item[1].get("false_positive_rate") is not None else 1.0,
            -(item[1].get("f1") if item[1].get("f1") is not None else -1.0),
            -item[0],
        ),
    )[0]
    return {"threshold": float(threshold), "reason": f"recall_target_{recall_target}_not_achievable", "metrics": metrics}


def threshold_for_safe_f1(rows: list[dict[str, Any]], *, preferred_fpr: float = 0.01, allowed_fpr: float = 0.02) -> dict[str, Any]:
    if not rows:
        return {"threshold": 0.5, "reason": "no_rows", "metrics": {"n": 0}, "safety_band": "none"}
    candidates: list[tuple[float, dict[str, Any]]] = []
    for threshold in threshold_grid([float(row["score"]) for row in rows]):
        decisions = apply_threshold(rows, threshold)
        metrics = metrics_from_rows(decisions)
        candidates.append((threshold, metrics))
    for band, limit in (("preferred_fpr", preferred_fpr), ("allowed_fpr", allowed_fpr)):
        feasible = [
            (threshold, metrics)
            for threshold, metrics in candidates
            if metrics.get("false_positive_rate") is not None and float(metrics["false_positive_rate"]) <= limit
        ]
        if feasible:
            threshold, metrics = sorted(
                feasible,
                key=lambda item: (
                    item[1].get("f1") if item[1].get("f1") is not None else -1.0,
                    item[1].get("recall") if item[1].get("recall") is not None else -1.0,
                    item[1].get("precision") if item[1].get("precision") is not None else -1.0,
                    -item[1].get("false_positive_rate") if item[1].get("false_positive_rate") is not None else -1.0,
                    item[0],
                ),
                reverse=True,
            )[0]
            return {"threshold": float(threshold), "reason": f"best_f1_under_{limit}", "metrics": metrics, "safety_band": band}
    threshold, metrics = sorted(
        candidates,
        key=lambda item: (
            -(item[1].get("false_positive_rate") if item[1].get("false_positive_rate") is not None else 1.0),
            item[1].get("f1") if item[1].get("f1") is not None else -1.0,
            item[1].get("recall") if item[1].get("recall") is not None else -1.0,
        ),
        reverse=True,
    )[0]
    return {"threshold": float(threshold), "reason": "no_threshold_met_allowed_fpr", "metrics": metrics, "safety_band": "unsafe"}


def detailed_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    decisions = apply_threshold(rows, threshold)
    metrics = metrics_from_rows(decisions)
    prompt = group_metrics(decisions, "prompt_group")
    hard = [row for row in decisions if hard_negative(row)]
    stage1_candidates = [row for row in decisions if bool(row.get("stage1_candidate"))]
    rejected = [row for row in stage1_candidates if not bool(row["decision"])]
    return {
        **metrics,
        "p1_recall": prompt.get("P1_vigil_only", {}).get("recall"),
        "p2_recall": prompt.get("P2_phrase_plus_vigil", {}).get("recall"),
        "p3_recall": prompt.get("P3_vigil_plus_phrase", {}).get("recall"),
        "p4_false_positive_rate": prompt.get("P4_negative", {}).get("false_positive_rate")
        or prompt.get("P4_negative_examples", {}).get("false_positive_rate"),
        "hard_negative_false_positive_rate": metrics_from_rows(hard).get("false_positive_rate") if hard else None,
        "stage2_rejection_rate": len(rejected) / len(stage1_candidates) if stage1_candidates else None,
        "stage1_candidates": len(stage1_candidates),
        "stage2_rejections": len(rejected),
    }
