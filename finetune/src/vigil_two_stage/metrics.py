from __future__ import annotations

from typing import Any

import numpy as np


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
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
    balanced = None
    if recall is not None and specificity is not None:
        balanced = (recall + specificity) / 2.0
    out: dict[str, Any] = {
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
        "balanced_accuracy": balanced,
    }
    if len(set(y.tolist())) == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            out["auroc"] = float(roc_auc_score(y, s))
            out["auprc"] = float(average_precision_score(y, s))
        except Exception as exc:
            out["auroc"] = None
            out["auprc"] = None
            out["curve_error"] = str(exc)
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out
