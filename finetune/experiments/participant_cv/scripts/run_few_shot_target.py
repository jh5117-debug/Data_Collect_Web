#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from vigil_two_stage.metrics import binary_metrics
from vigil_participant_cv.few_shot import eligible_for_shots
from vigil_participant_cv.support_sampling import select_positive_support, support_query_split
from vigil_participant_cv.utils import ensure_dir, read_json, read_jsonl, write_json


SUPPORT_SEEDS = [20260620, 20260621, 20260622, 20260623, 20260624]


def metric_from_predictions(rows: list[dict]) -> dict:
    return binary_metrics([int(row["label"]) for row in rows], [float(row["score"]) for row in rows], 0.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-manifest", default="finetune/experiments/participant_cv/shared/balanced_max100_manifest.jsonl")
    parser.add_argument("--folds", default="finetune/experiments/participant_cv/shared/participant_folds_5fold.json")
    parser.add_argument("--run-root", default="finetune/experiments/participant_cv/runs/zero_shot")
    parser.add_argument("--variant", default="stage2_bce")
    parser.add_argument("--out-dir", default="finetune/experiments/participant_cv/reports")
    args = parser.parse_args()
    recipe = read_json("finetune/experiments/participant_cv/reports/development_onboarding_recipe.json")
    if recipe["selected_recipe"] != "no_adaptation_zero_shot_fallback":
        raise RuntimeError("only conservative no-adaptation fallback is implemented in this runner")
    out_dir = ensure_dir(args.out_dir)
    clips_by_alias: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in read_jsonl(args.balanced_manifest):
        clips_by_alias[str(row["participant_alias"])].setdefault(str(row["clip_id"]), row)
    folds = read_json(args.folds)
    alias_to_fold = {alias: int(fold["fold"]) for fold in folds["folds"] for alias in fold["participant_aliases"]}
    predictions: dict[str, dict] = {}
    for path in sorted(Path(args.run_root).glob(f"fold_*/{args.variant}_cascade_test_clip_predictions.jsonl")):
        for row in read_jsonl(path):
            predictions[str(row["clip_id"])] = row
    result_rows = []
    support_sets = []
    eligibility_rows = []
    for alias, clip_map in sorted(clips_by_alias.items()):
        clips = list(clip_map.values())
        fold = alias_to_fold[alias]
        for shots in (3, 5):
            eligible, reason = eligible_for_shots(clips, shots)
            eligibility_rows.append({"participant_alias": alias, "fold": fold, "shots": shots, "eligible": eligible, "reason": reason})
            if not eligible:
                continue
            for seed in SUPPORT_SEEDS:
                support = select_positive_support(clips, k=shots, seed=seed)
                support, query = support_query_split(clips, support)
                query_ids = {str(row["clip_id"]) for row in query}
                pred_rows = [predictions[clip_id] for clip_id in sorted(query_ids)]
                support_sets.append(
                    {
                        "participant_alias": alias,
                        "fold": fold,
                        "shots": shots,
                        "support_seed": seed,
                        "support_clip_ids": [row["clip_id"] for row in support],
                        "support_prompt_groups": [row["prompt_group"] for row in support],
                        "query_clip_count": len(query),
                        "query_positive_count": sum(int(row["label"]) == 1 for row in query),
                        "query_negative_count": sum(int(row["label"]) == 0 for row in query),
                    }
                )
                for condition in (0, shots):
                    metrics = metric_from_predictions(pred_rows)
                    result_rows.append(
                        {
                            "participant_alias": alias,
                            "fold": fold,
                            "support_seed": seed,
                            "condition": condition,
                            "shots": shots,
                            "query_clip_count": len(query),
                            "query_positive_count": sum(int(row["label"]) == 1 for row in query),
                            "query_negative_count": sum(int(row["label"]) == 0 for row in query),
                            "adaptation_recipe": recipe["selected_recipe"],
                            **{k: metrics.get(k) for k in ("precision", "recall", "false_positive_rate", "f1", "tp", "tn", "fp", "fn")},
                        }
                    )
    support_path = out_dir / "few_shot_support_sets.json"
    write_json(support_path, support_sets)
    csv_path = out_dir / "few_shot_results.csv"
    fieldnames = list(result_rows[0].keys()) if result_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)
    elig_path = out_dir / "few_shot_eligibility.csv"
    with elig_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_alias", "fold", "shots", "eligible", "reason"])
        writer.writeheader()
        writer.writerows(eligibility_rows)
    summary = {"status": "ok", "adaptation_recipe": recipe["selected_recipe"], "conditions": {}, "eligible": {}}
    for shots in (3, 5):
        summary["eligible"][str(shots)] = sum(1 for row in eligibility_rows if row["shots"] == shots and row["eligible"])
    for condition in (0, 3, 5):
        rows = [row for row in result_rows if row["condition"] == condition]
        if not rows:
            continue
        summary["conditions"][str(condition)] = {
            metric: sum(float(row[metric]) for row in rows if row[metric] is not None) / len([row for row in rows if row[metric] is not None])
            for metric in ("precision", "recall", "false_positive_rate", "f1")
        }
    write_json(out_dir / "few_shot_summary.json", summary)
    lines = [
        "# Strict Positive-Only Participant Onboarding",
        "",
        f"- Adaptation recipe: `{recipe['selected_recipe']}`",
        f"- 3-shot eligible participants: `{summary['eligible'].get('3')}`",
        f"- 5-shot eligible participants: `{summary['eligible'].get('5')}`",
        "- Target negatives never enter adaptation.",
        "- Query sets remove support clips and are paired with the zero-shot comparison.",
        "- Because the selected recipe is a safety fallback, 3-shot and 5-shot do not claim adaptation improvement.",
        "",
        "| Condition | Recall | FPR | Precision | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in ("0", "3", "5"):
        item = summary["conditions"].get(condition)
        if item:
            lines.append(f"| {condition}-shot | {item['recall']} | {item['false_positive_rate']} | {item['precision']} | {item['f1']} |")
    (out_dir / "FEW_SHOT_ONBOARDING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
