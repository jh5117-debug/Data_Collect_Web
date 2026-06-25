from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clip_key(row: dict[str, Any]) -> str:
    return str(row["clip_id"])


def window_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["clip_id"]),
        int(row.get("window_index", 0)),
        str(row.get("window_audio_sha256") or row.get("audio_sha256") or ""),
    )


def alias_to_fold(folds: dict[str, Any]) -> dict[str, int]:
    return {str(alias): int(fold["fold"]) for fold in folds["folds"] for alias in fold["participant_aliases"]}


def fold_to_aliases(folds: dict[str, Any]) -> dict[int, set[str]]:
    return {int(fold["fold"]): set(str(alias) for alias in fold["participant_aliases"]) for fold in folds["folds"]}


def sanitize_public_row(row: dict[str, Any]) -> dict[str, Any]:
    blocked = {"speaker_id", "session_id", "account_id", "email", "name", "speaker_hash"}
    return {key: value for key, value in row.items() if key not in blocked}


def mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    import statistics

    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}
