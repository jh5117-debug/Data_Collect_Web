#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from vigil_participant_cv.error_analysis import classify_stage_error
from vigil_participant_cv.utils import ensure_dir, read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/participant_cv/runs/zero_shot")
    parser.add_argument("--variant", default="stage2_bce")
    parser.add_argument("--out-dir", default="finetune/experiments/participant_cv/reports")
    args = parser.parse_args()
    out_dir = ensure_dir(args.out_dir)
    errors = []
    all_rows = []
    for path in sorted(Path(args.run_root).glob(f"fold_*/{args.variant}_cascade_test_clip_predictions.jsonl")):
        fold = int(path.parent.name.split("_")[-1])
        for row in read_jsonl(path):
            item = dict(row)
            item["fold"] = fold
            category = classify_stage_error(item)
            item["error_category"] = category
            all_rows.append(item)
            if category:
                errors.append(item)
    jsonl_path = out_dir / "stage_error_predictions.jsonl"
    write_jsonl(jsonl_path, errors)
    summary_counts = Counter(row["error_category"] for row in errors)
    by_prompt = Counter((row["error_category"], row.get("prompt_group")) for row in errors)
    by_participant = Counter((row["error_category"], row.get("participant_alias") or row.get("speaker_id")) for row in errors)
    false_rejects = [row for row in errors if int(row["label"]) == 1]
    stage1_miss = sum(row["error_category"] == "STAGE1_MISS" for row in false_rejects)
    stage2_reject = sum(row["error_category"] == "STAGE2_REJECT" for row in false_rejects)
    summary_rows = [
        {"error_category": key, "count": value}
        for key, value in sorted(summary_counts.items())
    ]
    csv_path = out_dir / "stage_error_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["error_category", "count"])
        writer.writeheader()
        writer.writerows(summary_rows)
    report = [
        "# Stage Error Analysis",
        "",
        f"- Variant: `{args.variant}`",
        f"- Evaluated clips: `{len(all_rows)}`",
        f"- Error records: `{len(errors)}`",
        f"- False rejects: `{len(false_rejects)}`",
        f"- Stage 1 miss false rejects: `{stage1_miss}`",
        f"- Stage 2 reject false rejects: `{stage2_reject}`",
        f"- False-reject percentage caused by Stage 1: `{stage1_miss / len(false_rejects) if false_rejects else None}`",
        f"- False-reject percentage caused by Stage 2: `{stage2_reject / len(false_rejects) if false_rejects else None}`",
        "",
        "## Error Counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for row in summary_rows:
        report.append(f"| {row['error_category']} | {row['count']} |")
    report.extend(["", "## Errors By Prompt", "", "| Category | Prompt | Count |", "|---|---|---:|"])
    for (category, prompt), count in sorted(by_prompt.items()):
        report.append(f"| {category} | {prompt} | {count} |")
    report.extend(["", "## Interpretation", "", "Outer-test errors are reporting-only. Few-shot recipe selection must use development pseudo-targets, not these errors."])
    (out_dir / "STAGE_ERROR_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"errors": len(errors), "stage1_miss": stage1_miss, "stage2_reject": stage2_reject}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
