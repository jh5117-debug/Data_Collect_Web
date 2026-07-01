#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vigil_participant_cv.aggregation import mean_std
from vigil_participant_cv.utils import ensure_dir, write_json


METRICS = ["precision", "recall", "false_positive_rate", "f1", "balanced_accuracy"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/participant_cv/runs/zero_shot")
    parser.add_argument("--out-dir", default="finetune/experiments/participant_cv/reports")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    rows = []
    results = []
    for path in sorted(run_root.glob("fold_*/zero_shot_result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append(data)
        for method, bundle in data["methods"].items():
            metrics = bundle["metrics"]
            row = {"fold": data["outer_fold"], "method": method}
            for metric in METRICS:
                row[metric] = metrics.get(metric)
            rows.append(row)
    out_dir = ensure_dir(args.out_dir)
    csv_path = out_dir / "zero_shot_fold_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "method", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"status": "ok" if len(results) == 5 else "incomplete", "folds_completed": len(results), "methods": {}}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {metric: mean_std([float(row[metric]) for row in method_rows if row[metric] is not None]) for metric in METRICS}
    write_json(out_dir / "zero_shot_summary.json", summary)
    lines = [
        "# Participant-Disjoint Zero-Shot Five-Fold Evaluation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Folds completed: `{summary['folds_completed']}`",
        "",
        "| Method | Recall mean | Recall std | FPR mean | Precision mean | F1 mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in summary["methods"].items():
        lines.append(
            f"| {method} | {metrics['recall']['mean']} | {metrics['recall']['std']} | "
            f"{metrics['false_positive_rate']['mean']} | {metrics['precision']['mean']} | {metrics['f1']['mean']} |"
        )
    (out_dir / "ZERO_SHOT_5FOLD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
