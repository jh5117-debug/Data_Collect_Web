from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import binary_metrics


def select_recall_first_threshold(labels: list[int], scores: list[float], recall_target: float) -> dict[str, Any]:
    if not labels or not scores:
        return {"threshold": 0.5, "reason": "no_validation_predictions", "metrics": binary_metrics(labels, scores, 0.5)}
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if int((y == 1).sum()) == 0:
        return {"threshold": float(np.max(s) + 1e-6), "reason": "no_validation_positives", "metrics": binary_metrics(labels, scores, float(np.max(s) + 1e-6))}
    grid = sorted(set(float(x) for x in np.concatenate([s, np.linspace(0, 1, 101)])))
    best = None
    feasible = []
    for threshold in grid:
        m = binary_metrics(labels, scores, threshold)
        recall = m.get("recall")
        if recall is not None and recall >= recall_target:
            feasible.append((threshold, m))
    if feasible:
        best = sorted(feasible, key=lambda item: (item[1].get("false_positive_rate") if item[1].get("false_positive_rate") is not None else 1.0, -item[0]))[0]
        return {"threshold": float(best[0]), "reason": f"met_recall_target_{recall_target}", "metrics": best[1]}
    candidates = [(threshold, binary_metrics(labels, scores, threshold)) for threshold in grid]
    best = sorted(
        candidates,
        key=lambda item: (
            -(item[1].get("recall") if item[1].get("recall") is not None else -1),
            item[1].get("false_positive_rate") if item[1].get("false_positive_rate") is not None else 1.0,
            -(item[1].get("f1") if item[1].get("f1") is not None else -1),
        ),
    )[0]
    return {"threshold": float(best[0]), "reason": f"recall_target_{recall_target}_not_achievable", "metrics": best[1]}
