#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from vigil_final.utils import read_json, read_jsonl, sha256_file, write_json


def main() -> int:
    out = {
        "branch_expected": "research/vigil-fewshot-cost-nested-20260625",
        "balanced_manifest_rows": len(read_jsonl("finetune/experiments/participant_cv/shared/balanced_max100_manifest.jsonl")),
        "balanced_manifest_sha256": sha256_file("finetune/experiments/participant_cv/shared/balanced_max100_manifest.jsonl"),
        "fold_sha256": sha256_file("finetune/experiments/participant_cv/shared/participant_folds_5fold.json"),
        "qwen_transcript_cache_rows": len(read_jsonl("finetune/experiments/participant_cv/shared/qwen_transcript_cache_balanced_max100.jsonl")),
        "feature_coverage": read_json("finetune/experiments/participant_cv/reports/feature_coverage_report.json"),
        "old_few_shot_recipe": read_json("finetune/experiments/participant_cv/reports/development_onboarding_recipe.json"),
    }
    Path("finetune/experiments/vigil_final/reports").mkdir(parents=True, exist_ok=True)
    write_json("finetune/experiments/vigil_final/reports/current_result_audit.json", out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
