from __future__ import annotations

from statistics import mean, stdev
from typing import Any


def mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    return {"mean": float(mean(values)), "std": float(stdev(values)) if len(values) > 1 else 0.0}


def metric_mean_std(rows: list[dict[str, Any]], metric: str) -> dict[str, float | None]:
    return mean_std([float(row[metric]) for row in rows if row.get(metric) is not None])
