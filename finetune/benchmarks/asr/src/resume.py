from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils import append_jsonl, write_json, write_jsonl


def load_prediction_state(path: Path, *, retry_failed: bool = False) -> tuple[dict[str, dict[str, Any]], int]:
    completed: dict[str, dict[str, Any]] = {}
    corrupted = 0
    if not path.exists():
        return completed, corrupted
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                corrupted += 1
                continue
            utt_id = str(row.get("id", ""))
            if not utt_id:
                corrupted += 1
                continue
            if row.get("status") == "success":
                completed[utt_id] = row
            elif not retry_failed and row.get("status") == "failed":
                completed[utt_id] = row
    return completed, corrupted


def append_prediction(path: Path, row: dict[str, Any]) -> None:
    append_jsonl(path, row)


def deduplicate_predictions(path: Path, output_path: Path | None = None) -> dict[str, int]:
    rows: dict[str, dict[str, Any]] = {}
    total = corrupted = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                total += 1
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    corrupted += 1
                    continue
                utt_id = str(row.get("id", ""))
                if not utt_id:
                    corrupted += 1
                    continue
                rows[utt_id] = row
    target = output_path or path
    write_jsonl(target, [rows[key] for key in sorted(rows)])
    return {"input_rows": total, "output_rows": len(rows), "corrupted_rows": corrupted}


def update_progress(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
