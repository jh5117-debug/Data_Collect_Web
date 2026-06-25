#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vigil_final.utils import mean_std, read_json, write_csv, write_json


METHODS = ["qwen_exact", "stage1_only", "stage2_bce", "stage2_bce_supcon", "validation_selected"]


def metric(result: dict[str, Any], method: str) -> dict[str, Any]:
    return result["outer_test_methods"][method]["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/vigil_final/runs/nested_v2")
    parser.add_argument("--reports", default="finetune/experiments/vigil_final/reports")
    parser.add_argument("--v1-summary", default="finetune/experiments/participant_cv/reports/zero_shot_summary.json")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    reports = Path(args.reports)
    results = []
    for fold in range(5):
        path = run_root / f"outer_{fold}" / "nested_outer_result.json"
        if path.exists():
            results.append(read_json(path))
    if len(results) != 5:
        write_json(reports / "nested_zero_shot_summary.json", {"status": "incomplete", "completed_folds": len(results)})
        return 2

    fold_rows = []
    summary: dict[str, Any] = {"status": "ok", "version": "STRICT NESTED PARTICIPANT-DISJOINT FIVE-FOLD V2", "methods": {}}
    for method in METHODS:
        per_fold = []
        for result in results:
            m = metric(result, method)
            row = {"fold": result["outer_fold"], "method": method, **{k: m.get(k) for k in ("precision", "recall", "false_positive_rate", "f1", "balanced_accuracy")}}
            fold_rows.append(row)
            per_fold.append(row)
        summary["methods"][method] = {
            key: mean_std([float(row[key]) for row in per_fold if row.get(key) is not None])
            for key in ("precision", "recall", "false_positive_rate", "f1", "balanced_accuracy")
        }
    write_json(reports / "nested_zero_shot_summary.json", summary)
    write_csv(reports / "nested_fold_results.csv", fold_rows)
    v1 = read_json(args.v1_summary) if Path(args.v1_summary).exists() else {}
    lines = [
        "# Strict Nested Zero-Shot Five-Fold Report",
        "",
        "- Label: `STRICT NESTED PARTICIPANT-DISJOINT FIVE-FOLD V2`",
        "- Outer-test folds are final-evaluation only.",
        "- Development thresholds and model choice use inner-fold OOF predictions.",
        "",
        "| Method | Recall mean | FPR mean | Precision mean | F1 mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary["methods"][method]
        lines.append(
            f"| {method} | {item['recall']['mean']:.6f} | {item['false_positive_rate']['mean']:.6f} | "
            f"{item['precision']['mean']:.6f} | {item['f1']['mean']:.6f} |"
        )
    if v1:
        lines.extend(["", "## V1 Comparison", "", "V1 used one development fold as validation for each outer fold. V2 uses four inner held-out development folds for OOF selection."])
    (reports / "NESTED_ZERO_SHOT_5FOLD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": "ok", "folds": len(results)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
