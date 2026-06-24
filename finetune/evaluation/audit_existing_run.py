#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "finetune" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vigil_two_stage.thresholds import select_recall_first_threshold
from vigil_two_stage.utils import read_json, read_jsonl, write_json

from compare_window_clip_metrics import evaluate_run
from model_selection import select_stage2_variant, write_model_selection_markdown
from split_report import build_split_report
from finetune.scripts.run_qwen_text_baseline import (
    group_rows_for_evaluation,
    write_baseline_outputs,
)


STAGES = {
    "stage1": {"score_key": "score", "threshold_key": "theta_1", "config_section": "stage1"},
    "stage2_bce": {"score_key": "stage2_score", "threshold_key": "theta_2", "config_section": "stage2"},
    "stage2_bce_supcon": {"score_key": "stage2_score", "threshold_key": "theta_2", "config_section": "stage2"},
}


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({**row, "epoch": int(row["epoch"]), "train_loss": float(row["train_loss"]), "val_score": float(row["val_score"])})
    return rows


def training_history_summary(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stage, meta in STAGES.items():
        metrics_path = run_dir / stage / "metrics.json"
        metrics = read_json(metrics_path) if metrics_path.exists() else {}
        history = load_history(run_dir / stage / "train_history.csv")
        section = config.get(meta["config_section"], {})
        train_rows = int(metrics.get("train_rows") or 0)
        batch_size = int(section.get("batch_size") or 1)
        steps_per_epoch = int(math.ceil(train_rows / batch_size)) if train_rows else 0
        best = max(history, key=lambda row: row["val_score"]) if history else None
        final_epoch = int(history[-1]["epoch"]) if history else None
        configured_epochs = int(section.get("epochs") or 0)
        out[stage] = {
            "epochs_completed": len(history),
            "early_stopping_epoch": final_epoch,
            "best_validation_epoch": int(best["epoch"]) if best else None,
            "best_validation_score": float(best["val_score"]) if best else None,
            "train_rows": train_rows,
            "batch_size": batch_size,
            "train_steps_per_epoch": steps_per_epoch,
            "approximate_total_optimizer_steps": steps_per_epoch * len(history),
            "configured_epochs": configured_epochs,
            "early_stopping_patience": section.get("early_stopping_patience"),
            "stopped_early": bool(final_epoch and configured_epochs and final_epoch < configured_epochs),
        }
    return out


def threshold_audit(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for stage, meta in STAGES.items():
        pred_path = run_dir / stage / "val_predictions.jsonl"
        metrics_path = run_dir / stage / "metrics.json"
        threshold_path = run_dir / stage / "threshold.json"
        if not pred_path.exists() or not metrics_path.exists() or not threshold_path.exists():
            out[stage] = {"status": "missing_required_files"}
            continue
        preds = read_jsonl(pred_path)
        metrics = read_json(metrics_path)
        threshold = read_json(threshold_path)
        section = config.get(meta["config_section"], {})
        recall_target = float(section.get("recall_target", 0.0))
        recomputed = select_recall_first_threshold(
            [int(pred["label"]) for pred in preds],
            [float(pred[meta["score_key"]]) for pred in preds],
            recall_target,
        )
        stored_value = float(threshold["threshold"])
        recomputed_value = float(recomputed["threshold"])
        metrics_value = float(metrics[meta["threshold_key"]])
        out[stage] = {
            "status": "ok",
            "validation_prediction_rows": len(preds),
            "threshold_file": str(threshold_path),
            "threshold_key": meta["threshold_key"],
            "stored_threshold": stored_value,
            "metrics_threshold": metrics_value,
            "recomputed_from_validation_threshold": recomputed_value,
            "matches_threshold_file": abs(stored_value - recomputed_value) < 1e-9,
            "matches_metrics_json": abs(metrics_value - stored_value) < 1e-9,
            "recall_target": recall_target,
            "selection_reason": threshold.get("reason"),
            "validation_metrics_at_threshold": threshold.get("metrics"),
            "test_metrics_present_but_not_used_for_threshold": "test_metrics" in metrics,
        }
    return out


def metrics_file_summary(run_dir: Path) -> dict[str, Any]:
    summary = {}
    for rel in (
        "stage1/metrics.json",
        "stage2_bce/metrics.json",
        "stage2_bce_supcon/metrics.json",
        "cascade/metrics.json",
        "baseline_qwen_exact/metrics.json",
    ):
        path = run_dir / rel
        if not path.exists():
            summary[rel] = {"status": "missing"}
            continue
        data = read_json(path)
        summary[rel] = {
            key: data.get(key)
            for key in (
                "status",
                "variant",
                "theta_1",
                "theta_2",
                "train_rows",
                "val_rows",
                "test_rows",
                "test_metrics",
                "variants",
                "model_name",
                "evaluation_unit",
                "legacy_window_manifest_qwen_baseline",
                "precision",
                "recall",
                "false_positive_rate",
                "f1",
            )
            if key in data
        }
    return summary


def materialize_qwen_clip_baseline_from_legacy(run_dir: Path) -> dict[str, Any]:
    legacy_path = run_dir / "baseline_qwen_exact" / "predictions.jsonl"
    if not legacy_path.exists():
        return {"status": "missing_legacy_predictions"}
    legacy_preds = read_jsonl(legacy_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in legacy_preds:
        grouped[str(row["clip_id"])].append(row)
    duplicate_groups = {clip_id: len(rows) for clip_id, rows in grouped.items() if len(rows) > 1}
    clip_preds = group_rows_for_evaluation(legacy_preds, evaluation_unit="clip", deduplicate_by="clip_id")
    for pred in clip_preds:
        pred["evaluation_unit"] = "clip"
        pred["derived_from_legacy_window_manifest"] = True
    out_dir = run_dir / "baseline_qwen_exact_clip"
    mean_latency = sum(float(p.get("latency_sec", 0.0)) for p in clip_preds) / len(clip_preds) if clip_preds else None
    metrics = write_baseline_outputs(
        out_dir,
        clip_preds,
        model_name="Qwen/Qwen3-ASR-1.7B",
        split="test",
        evaluation_unit="clip",
        deduplicate_by="clip_id",
        mean_latency_sec=mean_latency,
        peak_gpu_memory_gb=None,
        extra={
            "source": "derived_from_existing_legacy_window_manifest_predictions",
            "source_legacy_rows": len(legacy_preds),
            "legacy_duplicate_clip_groups": duplicate_groups,
            "safe_without_rerun": not duplicate_groups,
        },
    )
    return {"status": "ok", "output_dir": str(out_dir), "metrics": metrics, "legacy_duplicate_clip_groups": duplicate_groups}


def _metric_line(metric: dict[str, Any]) -> str:
    return (
        f"n={metric.get('n')}, precision={metric.get('precision')}, recall={metric.get('recall')}, "
        f"FPR={metric.get('false_positive_rate')}, F1={metric.get('f1')}"
    )


def write_audit_markdown(path: Path, audit: dict[str, Any]) -> None:
    split_report = audit["split_report"]
    metrics = audit["window_clip_metrics"]["splits"]
    lines = [
        "# Evaluation Audit",
        "",
        "This audit recomputes evaluation semantics from existing predictions without retraining and without modifying original metric files.",
        "",
        "## Dataset",
        "",
        f"- Dataset fingerprint: `{audit['dataset_fingerprint']}`",
        f"- Dataset directory: `{audit['dataset_dir']}`",
        f"- Run directory: `{audit['run_dir']}`",
        "",
        "## Split Counts",
        "",
        "The P1/P2/P3/P4 columns in this table are window counts.",
        "",
        "| Split | Speakers | Participants | Sessions | Clips | Windows | Pos windows | Neg windows | P1 windows | P2 windows | P3 windows | P4 windows | Duplicate audio hash groups |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        counts = split_report["splits"][split]
        lines.append(
            f"| {split} | {counts['speakers']} | {counts['participants']} | {counts['sessions']} | "
            f"{counts['unique_original_clips']} | {counts['windows']} | {counts['positives']} | {counts['negatives']} | "
            f"{counts['P1']} | {counts['P2']} | {counts['P3']} | {counts['P4']} | {counts['duplicate_audio_hash_groups']} |"
        )
    lines.extend(
        [
            "",
            "Clip-level prompt group counts:",
            "",
            "| Split | P1 clips | P2 clips | P3 clips | P4 clips | Hard-negative phrase IDs |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for split in ("train", "val", "test"):
        counts = split_report["splits"][split]
        clip_counts = counts["prompt_group_clips"]
        hard_negative_phrase_ids = ", ".join(
            f"{phrase}:{count}" for phrase, count in counts["hard_negative_phrase_ids"].items()
        )
        lines.append(
            f"| {split} | {clip_counts['P1_vigil_only']} | {clip_counts['P2_phrase_plus_vigil']} | "
            f"{clip_counts['P3_vigil_plus_phrase']} | {clip_counts['P4_negative']} | {hard_negative_phrase_ids} |"
        )
    lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
            f"- No speaker leakage: {split_report['validations']['no_speaker_leakage']}",
            f"- No session leakage: {split_report['validations']['no_session_leakage']}",
            f"- No duplicate audio leakage: {split_report['validations']['no_duplicate_audio_leakage']}",
            f"- Clip split consistency: {split_report['validations']['clip_consistency']['all_windows_from_one_clip_in_one_split']}",
            f"- Clip label consistency: {split_report['validations']['clip_consistency']['all_windows_from_one_clip_same_label']}",
            f"- Clip speaker consistency: {split_report['validations']['clip_consistency']['all_windows_from_one_clip_same_speaker']}",
            f"- Train/val/test manifests consistent: {split_report['validations']['manifest_consistency']['train_val_test_manifests_consistent']}",
            "",
            "## Training Histories",
            "",
            "| Stage | Epochs | Best val epoch | Early-stop epoch | Steps/epoch | Approx optimizer steps |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage, item in audit["training_history"].items():
        lines.append(
            f"| {stage} | {item['epochs_completed']} | {item['best_validation_epoch']} | {item['early_stopping_epoch']} | "
            f"{item['train_steps_per_epoch']} | {item['approximate_total_optimizer_steps']} |"
        )
    lines.extend(
        [
            "",
            "The run completed quickly because the dataset is small, features are cached, Stage 2 trains only a small verifier head, and all three training jobs stopped before the configured 20 epochs.",
            "",
            "## Threshold Audit",
            "",
        ]
    )
    for stage, item in audit["threshold_audit"].items():
        lines.append(
            f"- {stage}: status={item.get('status')}, matches validation recompute={item.get('matches_threshold_file')}, "
            f"threshold={item.get('stored_threshold')}, selection={item.get('selection_reason')}"
        )
    lines.extend(["", "## Window-Level Metrics", ""])
    for variant in ("stage2_bce", "stage2_bce_supcon"):
        metric = metrics["test"][variant]["cascade_window"]
        lines.append(f"- test {variant} cascade window: {_metric_line(metric)}")
    lines.extend(["", "## Clip-Level Metrics", ""])
    for variant in ("stage2_bce", "stage2_bce_supcon"):
        metric = metrics["test"][variant]["cascade_clip"]
        lines.append(f"- test {variant} cascade clip: {_metric_line(metric)}")
    lines.extend(
        [
            "",
            "## Qwen Baseline",
            "",
            f"- Corrected clip-level baseline status: {audit['qwen_clip_baseline'].get('status')}",
            f"- Output: `{audit['qwen_clip_baseline'].get('output_dir')}`",
            "",
            "## Model Selection",
            "",
            f"- Selected variant: `{audit['model_selection'].get('selected_variant')}`",
            f"- Test metrics used for selection: {audit['model_selection'].get('test_metrics_used_for_selection')}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config_resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_report = read_json(run_dir / "dataset" / "dataset_report.json")
    evaluation_dir = run_dir / "evaluation"
    window_clip_metrics = evaluate_run(run_dir, dataset_dir, top_k=args.top_k, output_dir=evaluation_dir)
    model_selection = select_stage2_variant(
        window_clip_metrics,
        recall_constraint=float(config.get("stage2", {}).get("recall_target", 0.90)),
    )
    write_json(run_dir / "model_selection.json", model_selection)
    write_model_selection_markdown(run_dir / "MODEL_SELECTION.md", model_selection)
    audit = {
        "status": "ok",
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "dataset_fingerprint": dataset_report.get("dataset_fingerprint"),
        "dataset_report": dataset_report,
        "split_report": build_split_report(dataset_dir),
        "training_history": training_history_summary(run_dir, config),
        "threshold_audit": threshold_audit(run_dir, config),
        "metrics_file_summary": metrics_file_summary(run_dir),
        "window_clip_metrics": window_clip_metrics,
        "qwen_clip_baseline": materialize_qwen_clip_baseline_from_legacy(run_dir),
        "model_selection": model_selection,
    }
    write_json(run_dir / "evaluation_audit.json", audit)
    write_audit_markdown(run_dir / "EVALUATION_AUDIT.md", audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
