from __future__ import annotations

from typing import Any

import numpy as np

from vigil_two_stage.metrics import binary_metrics


def _fast_binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0:
        return {"n": 0, "defined": False}
    pred = (s >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "n": int(y.size),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_reject_rate": (fn / (tp + fn)) if tp + fn else None,
        "false_positive_rate": (fp / (fp + tn)) if fp + tn else None,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0 if recall is not None and specificity is not None else None,
    }


def threshold_for_recall_target(labels: list[int], scores: list[float], recall_target: float) -> dict[str, Any]:
    if not labels:
        return {"threshold": 0.5, "reason": "no_rows", "metrics": binary_metrics(labels, scores, 0.5)}
    grid = sorted(set(float(x) for x in np.concatenate([np.asarray(scores, dtype=np.float64), np.linspace(0, 1, 401)])))
    feasible = []
    for threshold in grid:
        metrics = _fast_binary_metrics(labels, scores, threshold)
        recall = metrics.get("recall")
        if recall is not None and recall >= recall_target:
            feasible.append((threshold, metrics))
    if feasible:
        threshold, metrics = sorted(
            feasible,
            key=lambda item: (
                item[1].get("false_positive_rate") if item[1].get("false_positive_rate") is not None else 1.0,
                -item[0],
            ),
        )[0]
        return {"threshold": float(threshold), "reason": f"met_recall_target_{recall_target}", "metrics": metrics}
    threshold, metrics = sorted(
        ((threshold, _fast_binary_metrics(labels, scores, threshold)) for threshold in grid),
        key=lambda item: (
            -(item[1].get("recall") if item[1].get("recall") is not None else -1.0),
            item[1].get("false_positive_rate") if item[1].get("false_positive_rate") is not None else 1.0,
            -(item[1].get("f1") if item[1].get("f1") is not None else -1.0),
        ),
    )[0]
    return {"threshold": float(threshold), "reason": f"recall_target_{recall_target}_not_achievable", "metrics": metrics}


def choose_variant(dev_metrics: dict[str, dict[str, Any]], recall_target: float = 0.90) -> str:
    def key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, float, float]:
        _name, metrics = item
        recall = float(metrics.get("recall") or 0.0)
        fpr = float(metrics.get("false_positive_rate") or 1.0)
        precision = float(metrics.get("precision") or 0.0)
        f1 = float(metrics.get("f1") or 0.0)
        return (1 if recall >= recall_target else 0, -fpr, precision, f1)

    return max(dev_metrics.items(), key=key)[0]


def operating_points(labels: list[int], scores: list[float], recall_targets: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "recall_target": target,
            **threshold_for_recall_target(labels, scores, target),
        }
        for target in recall_targets
    ]
