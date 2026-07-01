#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from scoring import score_pairs, score_prediction_rows
from utils import ensure_dir, read_jsonl, utc_timestamp, write_json


def _group(rows: list[dict], key: str) -> dict[str, dict]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {name: score_prediction_rows(group, normalized=True) for name, group in sorted(out.items())}


def _write_errors(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    fields = ["id", "split", "reference", "hypothesis", "wer", "substitutions", "deletions", "insertions"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.get("status") != "success":
                continue
            metrics = score_pairs([str(row.get("normalized_reference", ""))], [str(row.get("normalized_hypothesis", ""))])
            errors = int(metrics["substitutions"] or 0) + int(metrics["deletions"] or 0) + int(metrics["insertions"] or 0)
            if errors:
                writer.writerow(
                    {
                        "id": row.get("id"),
                        "split": row.get("split"),
                        "reference": row.get("reference"),
                        "hypothesis": row.get("hypothesis"),
                        "wer": metrics["wer"],
                        "substitutions": metrics["substitutions"],
                        "deletions": metrics["deletions"],
                        "insertions": metrics["insertions"],
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Score ASR predictions with local WER.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    successes = [row for row in rows if row.get("status") == "success"]
    payload = {
        "created_at_utc": utc_timestamp(),
        "predictions": str(args.predictions.resolve()),
        "prediction_rows": len(rows),
        "successful_predictions": len(successes),
        "failed_predictions": len([row for row in rows if row.get("status") == "failed"]),
        "normalized": score_prediction_rows(successes, normalized=True),
        "raw": score_prediction_rows(successes, normalized=False),
        "per_split": _group(successes, "split"),
        "per_speaker": _group(successes, "speaker_id"),
    }
    ensure_dir(args.output_dir)
    write_json(args.output_dir / "metrics.json", payload)
    _write_errors(args.output_dir / "error_analysis.csv", rows)
    (args.output_dir / "FINAL_REPORT.md").write_text(
        "# ASR Score Report\n\n"
        f"Predictions: `{args.predictions}`\n\n"
        f"Successful predictions: {payload['successful_predictions']}\n\n"
        f"Normalized WER: {payload['normalized']['wer']}\n",
        encoding="utf-8",
    )
    return 0 if successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
