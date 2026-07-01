#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

from vigil_latest_opt.cascade import apply_threshold, clip_score_rows
from vigil_latest_opt.utils import read_json, read_jsonl, write_csv, write_json


def main() -> int:
    reports = Path("finetune/experiments/latest_data_optimization/reports")
    selected = read_json(reports / "latest_opt_stage2_selected_config.json")
    run_root = Path("finetune/experiments/latest_data/runs/nested_zero_shot")
    rows = []
    for fold in range(5):
        root = run_root / f"fold_{fold}"
        clip_rows = clip_score_rows(
            read_jsonl(root / "stage1/test_predictions.jsonl"),
            read_jsonl(root / selected["variant"] / "test_predictions.jsonl"),
            theta1=float(read_json(root / "stage1/threshold.json")["threshold"]),
            top_k=int(selected["top_k"]),
        )
        rows.extend({**row, "fold": fold} for row in apply_threshold(clip_rows, float(selected["thresholds"][fold])))
    errors = []
    for row in rows:
        label = int(row["label"])
        decision = bool(row["decision"])
        if label == 1 and not decision:
            error_type = "STAGE1_MISS" if not row["stage1_candidate"] else "STAGE2_REJECT"
        elif label == 0 and decision:
            error_type = "STAGE2_FALSE_ACCEPT"
        elif label == 0 and row["stage1_candidate"]:
            error_type = "STAGE1_FALSE_CANDIDATE_REJECTED"
        else:
            continue
        errors.append(
            {
                "fold": row["fold"],
                "clip_id": row["clip_id"],
                "label": label,
                "prompt_group": row.get("prompt_group"),
                "error_type": error_type,
                "stage1_candidate": row["stage1_candidate"],
                "stage1_clip_score": row["stage1_clip_score"],
                "stage2_candidate_score": row["stage2_candidate_score"],
            }
        )
    counts = Counter(row["error_type"] for row in errors)
    prompt_counts = Counter((row["error_type"], row.get("prompt_group")) for row in errors)
    summary = {
        "status": "ok",
        "selected_variant": selected["variant"],
        "top_k": selected["top_k"],
        "total_error_records": len(errors),
        "counts": dict(sorted(counts.items())),
        "prompt_counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(prompt_counts.items())},
    }
    write_json(reports / "latest_opt_stage_error_summary.json", summary)
    write_csv(reports / "latest_opt_stage_error_summary.csv", [{"error_type": key, "count": value} for key, value in sorted(counts.items())])
    write_csv(reports / "latest_opt_stage_error_examples.csv", errors[:100])
    lines = [
        "# Latest Optimized Stage Error Analysis",
        "",
        f"- Status: `{summary['status']}`",
        f"- Selected variant/top_k: `{selected['variant']}` / `{selected['top_k']}`",
        "",
        "| Error type | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    (reports / "LATEST_OPT_STAGE_ERROR_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
