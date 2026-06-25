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
    start_cpu = time.process_time()
    start = time.perf_counter()
    result: dict[str, float] = {}
    yield result
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    result["wall_seconds"] = time.perf_counter() - start
    result["cpu_seconds"] = time.process_time() - start_cpu


def summarize_latencies(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "std": None}
    ordered = sorted(float(v) for v in values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[idx]

    return {
        "n": len(ordered),
        "mean": float(statistics.mean(ordered)),
        "median": float(statistics.median(ordered)),
        "p90": float(pct(0.90)),
        "p95": float(pct(0.95)),
        "std": float(statistics.stdev(ordered)) if len(ordered) > 1 else 0.0,
    }
