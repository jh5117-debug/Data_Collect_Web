#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from vigil_two_stage.metrics import binary_metrics
from vigil_two_stage.utils import read_json, read_jsonl, write_json, write_jsonl
from finetune.evaluation.aggregate_clip_predictions import aggregate_clip_cascade_predictions, aggregate_stage_clip_predictions


def key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["clip_id"]), int(row.get("window_index", 0)), str(row.get("window_audio_sha256"))


def fold_maps(folds: dict[str, Any]) -> dict[str, int]:
    return {alias: int(fold["fold"]) for fold in folds["folds"] for alias in fold["participant_aliases"]}


def split_for_alias(alias: str, alias_to_fold: dict[str, int], outer_fold: int) -> str:
    fold = alias_to_fold[alias]
    if fold == outer_fold:
        return "test"
    dev_folds = [i for i in range(5) if i != outer_fold]
    val_fold = dev_folds[0]
    return "val" if fold == val_fold else "train"


def make_feature_manifests(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    balanced = read_jsonl(args.balanced_manifest)
    folds = read_json(args.folds)
    alias_to_fold = fold_maps(folds)
    selected = {key(row): row for row in balanced}
    stage1_source = {key(row): row for row in read_jsonl(args.stage1_manifest)}
    qwen_source = {key(row): row for row in read_jsonl(args.qwen_manifest)}
    stage1_rows = []
    qwen_rows = []
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    alias_roles: dict[str, str] = {}
    for k, public in sorted(selected.items()):
        split = split_for_alias(str(public["participant_alias"]), alias_to_fold, int(args.outer_fold))
        alias_roles[str(public["participant_alias"])] = split
        split_counts[split] += 1
        for source, target in ((stage1_source, stage1_rows), (qwen_source, qwen_rows)):
            row = dict(source[k])
            row["split"] = split
            row["participant_alias"] = public["participant_alias"]
            target.append(row)
    write_jsonl(run_dir / "stage1_features_manifest.jsonl", stage1_rows)
    qwen_dir = run_dir / "stage2_qwen_features"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(qwen_dir / "qwen_features_manifest.jsonl", qwen_rows)
    integrity_src = Path(args.run_source_dir) / "stage2_qwen_features" / "frozen_qwen_integrity.json"
    if integrity_src.exists():
        write_json(qwen_dir / "frozen_qwen_integrity.json", read_json(integrity_src))
    return {"split_counts_windows": split_counts, "alias_roles": alias_roles}


def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}")


def score_clip_rows(rows: list[dict[str, Any]], score_key: str, threshold: float) -> dict[str, Any]:
    return binary_metrics([int(row["label"]) for row in rows], [float(row[score_key]) for row in rows], threshold)


def prompt_metrics(rows: list[dict[str, Any]], score_key: str, threshold: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("prompt_group")), []).append(row)
    return {name: score_clip_rows(group, score_key, threshold) for name, group in sorted(groups.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--balanced-manifest", default="finetune/experiments/participant_cv/shared/balanced_max100_manifest.jsonl")
    parser.add_argument("--folds", default="finetune/experiments/participant_cv/shared/participant_folds_5fold.json")
    parser.add_argument("--stage1-manifest", default="finetune/runs/20260624_075127_0fad4c7828149099_full/stage1/features_manifest.jsonl")
    parser.add_argument("--qwen-manifest", default="finetune/runs/20260624_075127_0fad4c7828149099_full/stage2_qwen_features/qwen_features_manifest.jsonl")
    parser.add_argument("--qwen-cache", default="finetune/experiments/participant_cv/shared/qwen_transcript_cache_balanced_max100.jsonl")
    parser.add_argument("--run-source-dir", default="finetune/runs/20260624_075127_0fad4c7828149099_full")
    parser.add_argument("--config", default="finetune/configs/full.yaml")
    parser.add_argument("--run-root", default="finetune/experiments/participant_cv/runs/zero_shot")
    args = parser.parse_args()
    run_dir = Path(args.run_root) / f"fold_{args.outer_fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = make_feature_manifests(args, run_dir)
    log_dir = Path("finetune/experiments/participant_cv/logs")
    py = sys.executable
    run_command([py, "finetune/scripts/train_stage1.py", "--features-manifest", str(run_dir / "stage1_features_manifest.jsonl"), "--config", args.config, "--run-dir", str(run_dir)], log_dir / f"zero_fold{args.outer_fold}_stage1.log")
    for variant in ("bce", "bce_supcon"):
        run_command([py, "finetune/scripts/train_stage2.py", "--dataset-dir", "finetune/data/processed/0fad4c7828149099", "--config", args.config, "--run-dir", str(run_dir), "--variant", variant], log_dir / f"zero_fold{args.outer_fold}_stage2_{variant}.log")

    theta1 = float(read_json(run_dir / "stage1" / "threshold.json")["threshold"])
    stage1_test = read_jsonl(run_dir / "stage1" / "test_predictions.jsonl")
    stage1_clip = aggregate_stage_clip_predictions(stage1_test, theta1)
    stage1_metrics = score_clip_rows(stage1_clip, "score", theta1)

    qwen_rows = [row for row in read_jsonl(args.qwen_cache)]
    folds = read_json(args.folds)
    alias_to_fold = fold_maps(folds)
    qwen_test = [row for row in qwen_rows if alias_to_fold[str(row["participant_alias"])] == int(args.outer_fold)]
    qwen_metrics = binary_metrics([int(row["label"]) for row in qwen_test], [float(row["exact_trigger_decision"]) for row in qwen_test], 0.5)

    methods: dict[str, Any] = {
        "qwen_exact": {"metrics": qwen_metrics, "per_prompt": prompt_metrics(qwen_test, "exact_trigger_decision", 0.5)},
        "stage1_only": {"metrics": stage1_metrics, "per_prompt": prompt_metrics(stage1_clip, "score", theta1), "theta_1": theta1},
    }
    dev_selection: dict[str, Any] = {}
    for variant, dirname in (("stage2_bce", "stage2_bce"), ("stage2_bce_supcon", "stage2_bce_supcon")):
        theta2 = float(read_json(run_dir / dirname / "threshold.json")["threshold"])
        stage2_test = read_jsonl(run_dir / dirname / "test_predictions.jsonl")
        cascade = aggregate_clip_cascade_predictions(stage1_test, stage2_test, theta1, theta2, top_k=3)
        metrics = binary_metrics([int(row["label"]) for row in cascade], [float(row["score"]) for row in cascade], 0.5)
        methods[variant] = {"metrics": metrics, "per_prompt": prompt_metrics(cascade, "score", 0.5), "theta_1": theta1, "theta_2": theta2}
        val_metrics = read_json(run_dir / dirname / "metrics.json").get("threshold_selection", {}).get("metrics", {})
        dev_selection[variant] = val_metrics
        write_jsonl(run_dir / f"{variant}_cascade_test_clip_predictions.jsonl", cascade)
    selected = max(
        dev_selection.items(),
        key=lambda item: (
            float(item[1].get("recall") or 0.0) >= 0.9,
            -float(item[1].get("false_positive_rate") or 1.0),
            float(item[1].get("precision") or 0.0),
            float(item[1].get("f1") or 0.0),
        ),
    )[0]
    methods["validation_selected"] = methods[selected]
    result = {
        "status": "ok",
        "outer_fold": int(args.outer_fold),
        "protocol_note": "participant_disjoint_outer_fold_with_single_development_validation_fold_v1",
        "split_counts_windows": meta["split_counts_windows"],
        "methods": methods,
        "validation_selected_variant": selected,
    }
    write_json(run_dir / "zero_shot_result.json", result)
    print(json.dumps({"outer_fold": args.outer_fold, "selected": selected, "stage1_recall": stage1_metrics.get("recall")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
