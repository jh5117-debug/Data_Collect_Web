from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Iterator

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@contextmanager
def cuda_synchronized_timer(device: str | None = None) -> Iterator[dict[str, float]]:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result: dict[str, float] = {}
    yield result
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    result["seconds"] = time.perf_counter() - start


def summarize_seconds(values: list[float]) -> dict[str, float | int | None]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p90_ms": None, "p95_ms": None, "std_ms": None}

    def pct(q: float) -> float:
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
        return vals[idx] * 1000.0

    return {
        "n": len(vals),
        "mean_ms": statistics.mean(vals) * 1000.0,
        "median_ms": statistics.median(vals) * 1000.0,
        "p90_ms": pct(0.90),
        "p95_ms": pct(0.95),
        "std_ms": statistics.stdev(vals) * 1000.0 if len(vals) > 1 else 0.0,
    }
