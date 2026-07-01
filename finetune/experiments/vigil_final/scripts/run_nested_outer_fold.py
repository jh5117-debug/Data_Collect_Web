#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from finetune.evaluation.aggregate_clip_predictions import aggregate_clip_cascade_predictions, aggregate_stage_clip_predictions
from vigil_final.calibration import choose_variant, operating_points, threshold_for_recall_target
from vigil_final.metrics import metric_from_decisions
from vigil_final.nested_cv import Context, RoleSeparatedDataset, build_outer_plan, validate_inner_coverage
from vigil_final.refit import median_epoch_policy
from vigil_final.utils import alias_to_fold, read_json, read_jsonl, window_key, write_json, write_jsonl
from vigil_two_stage.metrics import binary_metrics


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}")


def map_feature_rows(public_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    source_by_key = {window_key(row): row for row in source_rows}
    out = []
    missing = []
    for public in public_rows:
        src = source_by_key.get(window_key(public))
        if src is None:
            missing.append(window_key(public))
            continue
        row = dict(src)
        row["split"] = split
        row["participant_alias"] = public["participant_alias"]
        out.append(row)
    if missing:
        raise RuntimeError(f"missing feature rows: {missing[:3]}")
    return out


def write_run_manifests(
    run_dir: Path,
    *,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    stage1_source: list[dict[str, Any]],
    qwen_source: list[dict[str, Any]],
    source_run_dir: Path,
) -> None:
    stage1_rows = (
        map_feature_rows(train_rows, stage1_source, "train")
        + map_feature_rows(val_rows, stage1_source, "val")
        + map_feature_rows(test_rows, stage1_source, "test")
    )
    qwen_rows = (
        map_feature_rows(train_rows, qwen_source, "train")
        + map_feature_rows(val_rows, qwen_source, "val")
        + map_feature_rows(test_rows, qwen_source, "test")
    )
    write_jsonl(run_dir / "stage1_features_manifest.jsonl", stage1_rows)
    qwen_dir = run_dir / "stage2_qwen_features"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(qwen_dir / "qwen_features_manifest.jsonl", qwen_rows)
    integrity = source_run_dir / "stage2_qwen_features" / "frozen_qwen_integrity.json"
    if integrity.exists():
        write_json(qwen_dir / "frozen_qwen_integrity.json", read_json(integrity))


def maybe_train(run_dir: Path, config: dict[str, Any], *, logs: Path, env: dict[str, str]) -> None:
    py = sys.executable
    base_config = str(config["base_train_config"])
    if not (run_dir / "stage1" / "test_predictions.jsonl").exists() and not (run_dir / "stage1" / "val_predictions.jsonl").exists():
        run_command(
            [py, "finetune/scripts/train_stage1.py", "--features-manifest", str(run_dir / "stage1_features_manifest.jsonl"), "--config", base_config, "--run-dir", str(run_dir)],
            logs / f"{run_dir.name}_stage1.log",
            env=env,
        )
    for variant in config["variants"]:
        out_dir = run_dir / f"stage2_{variant}"
        if not (out_dir / "test_predictions.jsonl").exists() and not (out_dir / "val_predictions.jsonl").exists():
            run_command(
                [py, "finetune/scripts/train_stage2.py", "--dataset-dir", "finetune/data/processed/0fad4c7828149099", "--config", base_config, "--run-dir", str(run_dir), "--variant", variant],
                logs / f"{run_dir.name}_stage2_{variant}.log",
                env=env,
            )


def epoch_from_history(path: Path) -> int | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) <= 1:
        return None
    return len(lines) - 1


def cascade_scores(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]], theta1: float, top_k: int) -> list[dict[str, Any]]:
    rows = aggregate_clip_cascade_predictions(stage1_rows, stage2_rows, theta1, 1.000001, top_k=top_k)
    for row in rows:
        row["selection_score"] = float(row["stage2_candidate_score"]) if row["stage1_candidate"] else 0.0
    return rows


def qwen_outer_metrics(qwen_cache: list[dict[str, Any]], folds: dict[str, Any], outer_fold: int) -> dict[str, Any]:
    a2f = alias_to_fold(folds)
    rows = [row for row in qwen_cache if a2f[str(row["participant_alias"])] == outer_fold]
    return binary_metrics([int(row["label"]) for row in rows], [float(row["exact_trigger_decision"]) for row in rows], 0.5)


def evaluate_refit(
    refit_dir: Path,
    *,
    config: dict[str, Any],
    theta1: float,
    theta2_by_variant: dict[str, float],
    qwen_metrics: dict[str, Any],
) -> dict[str, Any]:
    stage1_test = read_jsonl(refit_dir / "stage1" / "test_predictions.jsonl")
    stage1_clip = aggregate_stage_clip_predictions(stage1_test, theta1)
    methods: dict[str, Any] = {
        "qwen_exact": {"metrics": qwen_metrics},
        "stage1_only": {
            "metrics": binary_metrics([int(row["label"]) for row in stage1_clip], [float(row["score"]) for row in stage1_clip], theta1),
            "theta_1": theta1,
        },
    }
    for variant in config["variants"]:
        name = f"stage2_{variant}"
        stage2_test = read_jsonl(refit_dir / name / "test_predictions.jsonl")
        cascade = aggregate_clip_cascade_predictions(stage1_test, stage2_test, theta1, theta2_by_variant[variant], top_k=int(config["top_k"]))
        write_jsonl(refit_dir / f"{name}_cascade_test_clip_predictions.jsonl", cascade)
        methods[name] = {
            "metrics": binary_metrics([int(row["label"]) for row in cascade], [float(row["score"]) for row in cascade], 0.5),
            "theta_1": theta1,
            "theta_2": theta2_by_variant[variant],
        }
    return methods


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--config", default="finetune/experiments/vigil_final/configs/nested_cv.yaml")
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    outer_fold = int(args.outer_fold)
    run_root = Path(config["run_root"])
    outer_dir = run_root / f"outer_{outer_fold}"
    outer_dir.mkdir(parents=True, exist_ok=True)
    logs = Path("finetune/experiments/vigil_final/logs")
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    balanced = read_jsonl(config["balanced_manifest"])
    folds = read_json(config["folds"])
    plan = build_outer_plan(outer_fold)
    coverage = validate_inner_coverage(plan, folds)
    guarded = RoleSeparatedDataset(balanced, folds, outer_fold)
    stage1_source = read_jsonl(config["source_stage1_manifest"])
    qwen_source = read_jsonl(config["source_qwen_manifest"])
    source_run_dir = Path(config["source_run_dir"])

    all_oof_stage1 = []
    oof_stage2: dict[str, list[dict[str, Any]]] = {str(variant): [] for variant in config["variants"]}
    best_epochs: dict[str, list[int]] = {"stage1": [], **{str(variant): [] for variant in config["variants"]}}
    inner_meta = []
    for inner in plan.inner_folds:
        inner_dir = outer_dir / f"inner_val_{inner.inner_validation_fold}"
        train_rows = guarded.read_inner_train(inner, Context.INNER_SELECTION)
        val_rows = guarded.read_inner_validation(inner, Context.INNER_SELECTION)
        write_run_manifests(
            inner_dir,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=[],
            stage1_source=stage1_source,
            qwen_source=qwen_source,
            source_run_dir=source_run_dir,
        )
        maybe_train(inner_dir, config, logs=logs, env=env)
        stage1_val = read_jsonl(inner_dir / "stage1" / "val_predictions.jsonl")
        all_oof_stage1.extend(stage1_val)
        epoch = epoch_from_history(inner_dir / "stage1" / "train_history.csv")
        if epoch is not None:
            best_epochs["stage1"].append(epoch)
        for variant in config["variants"]:
            stage2_val = read_jsonl(inner_dir / f"stage2_{variant}" / "val_predictions.jsonl")
            oof_stage2[str(variant)].extend(stage2_val)
            epoch = epoch_from_history(inner_dir / f"stage2_{variant}" / "train_history.csv")
            if epoch is not None:
                best_epochs[str(variant)].append(epoch)
        inner_meta.append({"inner_validation_fold": inner.inner_validation_fold, "train_rows": len(train_rows), "validation_rows": len(val_rows)})

    stage1_clip = aggregate_stage_clip_predictions(all_oof_stage1, 0.5)
    theta1_sel = threshold_for_recall_target(
        [int(row["label"]) for row in stage1_clip],
        [float(row["score"]) for row in stage1_clip],
        float(config["stage1_recall_target"]),
    )
    theta1 = float(theta1_sel["threshold"])
    stage2_selections: dict[str, Any] = {}
    theta2_by_variant: dict[str, float] = {}
    dev_metrics: dict[str, dict[str, Any]] = {}
    operating: dict[str, list[dict[str, Any]]] = {}
    for variant in config["variants"]:
        scores = cascade_scores(all_oof_stage1, oof_stage2[str(variant)], theta1, int(config["top_k"]))
        labels = [int(row["label"]) for row in scores]
        selection_scores = [float(row["selection_score"]) for row in scores]
        selected = threshold_for_recall_target(labels, selection_scores, float(config["stage2_recall_target"]))
        theta2 = float(selected["threshold"])
        theta2_by_variant[str(variant)] = theta2
        cascade = aggregate_clip_cascade_predictions(all_oof_stage1, oof_stage2[str(variant)], theta1, theta2, top_k=int(config["top_k"]))
        metrics = binary_metrics([int(row["label"]) for row in cascade], [float(row["score"]) for row in cascade], 0.5)
        dev_metrics[f"stage2_{variant}"] = metrics
        stage2_selections[str(variant)] = {"threshold_selection": selected, "oof_metrics": metrics}
        operating[str(variant)] = operating_points(labels, selection_scores, [float(x) for x in config["stage2_operating_recall_targets"]])
    selected_name = choose_variant(dev_metrics, recall_target=float(config["stage2_recall_target"]))
    selected_variant = selected_name.replace("stage2_", "")

    refit_rows = guarded.read_refit_train(Context.REFIT)
    test_rows = guarded.read_outer_test(Context.FINAL_EVALUATION)
    refit_dir = outer_dir / "refit"
    write_run_manifests(
        refit_dir,
        train_rows=refit_rows,
        val_rows=[],
        test_rows=test_rows,
        stage1_source=stage1_source,
        qwen_source=qwen_source,
        source_run_dir=source_run_dir,
    )
    maybe_train(refit_dir, config, logs=logs, env=env)
    qwen_metrics = qwen_outer_metrics(read_jsonl(config["qwen_transcript_cache"]), folds, outer_fold)
    methods = evaluate_refit(refit_dir, config=config, theta1=theta1, theta2_by_variant=theta2_by_variant, qwen_metrics=qwen_metrics)
    methods["validation_selected"] = methods[f"stage2_{selected_variant}"]
    result = {
        "status": "ok",
        "version": config["version"],
        "outer_fold": outer_fold,
        "coverage": coverage,
        "inner_runs": inner_meta,
        "access_log": guarded.access_log.records,
        "oof_rows": {"stage1_windows": len(all_oof_stage1), **{f"stage2_{k}_windows": len(v) for k, v in oof_stage2.items()}},
        "theta_1_selection": theta1_sel,
        "stage2_selections": stage2_selections,
        "stage2_operating_points": operating,
        "selected_variant": selected_variant,
        "selected_method": f"stage2_{selected_variant}",
        "epoch_policy": {
            "stage1": median_epoch_policy(best_epochs["stage1"]),
            **{variant: median_epoch_policy(values) for variant, values in best_epochs.items() if variant != "stage1"},
        },
        "outer_test_methods": methods,
    }
    write_json(outer_dir / "nested_outer_result.json", result)
    print(json.dumps({"outer_fold": outer_fold, "selected_variant": selected_variant, "theta1": theta1}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
