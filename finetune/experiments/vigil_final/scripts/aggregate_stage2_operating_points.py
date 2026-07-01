#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from vigil_final.utils import mean_std, read_json, write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/vigil_final/runs/nested_v2")
    parser.add_argument("--reports", default="finetune/experiments/vigil_final/reports")
    args = parser.parse_args()
    rows = []
    for fold in range(5):
        result = read_json(Path(args.run_root) / f"outer_{fold}" / "nested_outer_result.json")
        for variant, points in result["stage2_operating_points"].items():
            for point in points:
                metrics = point["metrics"]
                rows.append(
                    {
                        "fold": fold,
                        "variant": f"stage2_{variant}",
                        "recall_target": point["recall_target"],
                        "threshold": point["threshold"],
                        "recall": metrics.get("recall"),
                        "false_positive_rate": metrics.get("false_positive_rate"),
                        "precision": metrics.get("precision"),
                        "f1": metrics.get("f1"),
                    }
                )
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], float(row["recall_target"]))].append(row)
    summary = {"status": "ok", "operating_points": {}}
    for (variant, target), group in sorted(grouped.items()):
        key = f"{variant}_target_{target:.2f}"
        summary["operating_points"][key] = {
            metric: mean_std([float(row[metric]) for row in group if row.get(metric) is not None])
            for metric in ("threshold", "recall", "false_positive_rate", "precision", "f1")
        }
    reports = Path(args.reports)
    write_json(reports / "stage2_operating_points.json", summary)
    write_csv(reports / "stage2_operating_points.csv", rows)
    lines = [
        "# Stage 2 Operating-Point Report",
        "",
        "Development OOF predictions only. Outer-test folds are not used to choose thresholds.",
        "",
        "| Variant | Recall target | Recall mean | FPR mean | Precision mean | F1 mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary["operating_points"].items():
        variant, target = key.rsplit("_target_", 1)
        lines.append(
            f"| {variant} | {float(target):.2f} | {item['recall']['mean']:.6f} | "
            f"{item['false_positive_rate']['mean']:.6f} | {item['precision']['mean']:.6f} | {item['f1']['mean']:.6f} |"
        )
    (reports / "STAGE2_OPERATING_POINT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": "ok", "rows": len(rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
