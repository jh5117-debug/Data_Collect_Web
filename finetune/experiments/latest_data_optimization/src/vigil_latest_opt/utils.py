from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def write_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "std": None}

    def pct(q: float) -> float:
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
        return vals[idx]

    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "p90": float(pct(0.90)),
        "p95": float(pct(0.95)),
        "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
    }
