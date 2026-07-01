from __future__ import annotations

from typing import Any


def select_validation_variant(dev_metrics: dict[str, dict[str, float | None]], *, recall_target: float = 0.9) -> str:
    def key(item: tuple[str, dict[str, float | None]]) -> tuple[int, float, float, float]:
        _name, metrics = item
        recall = float(metrics.get("recall") or 0.0)
        fpr = float(metrics.get("false_positive_rate") or 1.0)
        precision = float(metrics.get("precision") or 0.0)
        f1 = float(metrics.get("f1") or 0.0)
        return (1 if recall >= recall_target else 0, -fpr, precision, f1)

    return max(dev_metrics.items(), key=key)[0]
