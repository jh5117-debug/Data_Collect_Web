#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPORTS = Path("finetune/experiments/fewshot_ablation/reports")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, default=REPORTS)
    args = parser.parse_args()
    summary_path = args.reports / "fewshot_ablation_summary.json"
    method_path = args.reports / "fewshot_ablation_per_method.csv"
    if not summary_path.exists() or not method_path.exists():
        raise SystemExit("run_target_doctor_fewshot_ablation.py must be run first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    methods = read_csv(method_path)
    best = max(
        [row for row in methods if row["method"] != "zero_shot"],
        key=lambda row: (
            float(row.get("delta_f1") or 0.0),
            float(row.get("delta_recall") or 0.0),
        ),
    )
    summary["aggregate_check"] = {
        "method_rows": len(methods),
        "best_by_csv": {
            "shot": int(best["shot"]),
            "method": best["method"],
            "delta_f1": float(best["delta_f1"]),
            "f1": float(best["f1"]) if best["f1"] else None,
            "recall": float(best["recall"]) if best["recall"] else None,
            "fpr": float(best["fpr"]) if best["fpr"] else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate_check"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
