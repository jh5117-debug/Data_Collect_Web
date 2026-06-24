#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from vigil_participant_cv.folds import build_folds
from vigil_participant_cv.privacy import assert_public_text_is_sanitized
from vigil_participant_cv.protocol import validate_no_duplicate_hash_crosses_folds, validate_unique_fold_membership
from vigil_participant_cv.utils import read_jsonl, sha256_file, write_json


def clip_rows(rows: list[dict]) -> list[dict]:
    seen = {}
    for row in rows:
        clip_id = str(row["clip_id"])
        if clip_id not in seen:
            seen[clip_id] = dict(row)
    return [seen[k] for k in sorted(seen)]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="finetune/experiments/participant_cv/shared/balanced_max100_manifest.jsonl")
    parser.add_argument("--config", default="finetune/experiments/participant_cv/configs/protocol.yaml")
    parser.add_argument("--out-dir", default="finetune/experiments/participant_cv")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = read_jsonl(args.manifest)
    clips = clip_rows(rows)
    folds = build_folds(
        clips,
        fold_count=int(config["fold_count"]),
        seed=int(config["seed"]),
        weights={k: float(v) for k, v in config["fold_objective_weights"].items()},
    )
    validate_unique_fold_membership(folds)
    validate_no_duplicate_hash_crosses_folds(clips, folds)

    out_dir = Path(args.out_dir)
    shared = out_dir / "shared"
    reports = out_dir / "reports"
    shared.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    fold_path = shared / "participant_folds_5fold.json"
    write_json(fold_path, folds)
    fold_sha = sha256_file(fold_path)
    alias_to_fold = {alias: fold["fold"] for fold in folds["folds"] for alias in fold["participant_aliases"]}
    per_fold_counts = []
    for fold in folds["folds"]:
        fold_rows = [clip for clip in clips if alias_to_fold[clip["participant_alias"]] == fold["fold"]]
        per_fold_counts.append(
            {
                "fold": fold["fold"],
                "participant_aliases": fold["participant_aliases"],
                "participants": len(fold["participant_aliases"]),
                "clips": len(fold_rows),
                "positive": sum(int(c["label"]) == 1 for c in fold_rows),
                "negative": sum(int(c["label"]) == 0 for c in fold_rows),
                "prompt_counts": dict(sorted(Counter(str(c["prompt_group"]) for c in fold_rows).items())),
            }
        )
    protocol = {
        "protocol_version": config["protocol_version"],
        "dataset_fingerprint": config["dataset_fingerprint"],
        "balanced_manifest_sha256": sha256_file(args.manifest),
        "fold_definition_sha256": fold_sha,
        "cap_rule": {"max_clips_per_participant": config["max_clips_per_participant"], "unit": "clip_id"},
        "fold_seed": config["seed"],
        "fold_generation_algorithm": folds["algorithm"],
        "alias_generation_rule": config["privacy"]["alias_rule"],
        "code_git_commit": git_commit(),
        "folds": per_fold_counts,
    }
    write_json(shared / "shared_experiment_protocol.json", protocol)
    report = [
        "# Fold Balance Report",
        "",
        f"- Fold definition SHA-256: `{fold_sha}`",
        f"- Objective: `{folds['objective']}`",
        f"- Algorithm: `{folds['algorithm']}`",
        "",
        "| Fold | Participants | Clips | Pos | Neg | P1 | P2 | P3 | P4 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_fold_counts:
        prompt = defaultdict(int, row["prompt_counts"])
        report.append(
            f"| {row['fold']} | {row['participants']} | {row['clips']} | {row['positive']} | {row['negative']} | "
            f"{prompt['P1_vigil_only']} | {prompt['P2_phrase_plus_vigil']} | {prompt['P3_vigil_plus_phrase']} | {prompt['P4_negative']} |"
        )
    text = "\n".join(report) + "\n"
    assert_public_text_is_sanitized(text)
    (reports / "FOLD_BALANCE_REPORT.md").write_text(text, encoding="utf-8")
    print(json.dumps({"fold_sha256": fold_sha, "objective": folds["objective"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
