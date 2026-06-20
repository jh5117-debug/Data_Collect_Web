#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

from vigil_two_stage.metrics import binary_metrics
from vigil_two_stage.utils import ensure_dir, read_json, read_jsonl, write_json


def run_text(cmd: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return None


def write_predictions_csv(path: Path, preds: list[dict]) -> None:
    fields = [
        "clip_id",
        "speaker_id",
        "session_id",
        "prompt_group",
        "transcript",
        "label",
        "phrase_id",
        "split",
        "window_index",
        "score",
        "stage1_score",
        "stage2_score",
        "candidate",
        "final_trigger",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in preds:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_reproducibility(path: Path, project_root: Path, dataset_report: dict, run_dir: Path) -> None:
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}
    reproducibility = {
        "project_git_commit": run_text(["git", "rev-parse", "HEAD"], project_root),
        "project_git_status_short": run_text(["git", "status", "--short"], project_root),
        "archive_sha256": dataset_report.get("zip_sha256"),
        "dataset_fingerprint": dataset_report.get("dataset_fingerprint"),
        "openwakeword": {
            "package_version": package_version("openwakeword"),
            "feature_status": read_json(run_dir / "stage1" / "feature_status.json")
            if (run_dir / "stage1" / "feature_status.json").exists()
            else None,
        },
        "qwen3_asr": {
            "model_name": "Qwen/Qwen3-ASR-1.7B",
            "transformers_version": package_version("transformers"),
            "modelscope_version": package_version("modelscope"),
            "feature_status": read_json(run_dir / "stage2_qwen_features" / "feature_status.json")
            if (run_dir / "stage2_qwen_features" / "feature_status.json").exists()
            else None,
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "torch": torch_info,
        "config_resolved": (run_dir / "config_resolved.yaml").read_text(encoding="utf-8")
        if (run_dir / "config_resolved.yaml").exists()
        else None,
        "executed_commands": [
            "bash finetune/scripts/run_official_smoke_local_3090.sh <physical_gpu_index> /home/hj/Data_Collect_Web/finetune/data/vigil_dataset_export_20260620_020617.zip"
        ],
    }
    write_json(path, reproducibility)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    cascade_dir = ensure_dir(run_dir / "cascade")
    ensure_dir(run_dir / "plots")
    stage1_metrics_path = run_dir / "stage1" / "metrics.json"
    stage1_predictions_path = run_dir / "stage1" / "test_predictions.jsonl"
    dataset_report = read_json(Path(args.dataset_dir) / "dataset_report.json")
    project_root = Path(__file__).resolve().parents[2]
    summary = {
        "dataset_fingerprint": dataset_report["dataset_fingerprint"],
        "dataset": {
            "manifest_windows": dataset_report["manifest_windows"],
            "prompt_group_counts": dataset_report["prompt_group_counts"],
            "label_counts": dataset_report["label_counts"],
            "split_report": dataset_report["split_report"],
        },
        "stage1": {"status": "missing"},
        "stage2_bce": {"status": "missing"},
        "stage2_bce_supcon": {"status": "missing"},
        "cascade": {"status": "not_run"},
    }
    if stage1_metrics_path.exists() and stage1_predictions_path.exists():
        stage1_metrics = read_json(stage1_metrics_path)
        preds = read_jsonl(stage1_predictions_path)
        write_predictions_csv(run_dir / "predictions.csv", preds)
        theta = float(stage1_metrics.get("theta_1", 0.5))
        summary["stage1"] = stage1_metrics
        summary["cascade"] = {
            "status": "stage1_only",
            "reason": "Stage 2 Qwen verifier features were unavailable, so final two-stage cascade was not executed.",
            "stage1_only_test_metrics": binary_metrics([p["label"] for p in preds], [p["score"] for p in preds], theta),
            "theta_1": theta,
            "theta_2": None,
        }
    for name in ("stage2_bce", "stage2_bce_supcon"):
        path = run_dir / name / "metrics.json"
        if path.exists():
            summary[name] = read_json(path)
    if stage1_predictions_path.exists():
        stage1_preds = read_jsonl(stage1_predictions_path)
        stage1_metrics = read_json(stage1_metrics_path) if stage1_metrics_path.exists() else {}
        theta_1 = float(stage1_metrics.get("theta_1", 0.5))
        cascade_results = {}
        for name in ("stage2_bce", "stage2_bce_supcon"):
            metrics_path = run_dir / name / "metrics.json"
            preds_path = run_dir / name / "test_predictions.jsonl"
            if not metrics_path.exists() or not preds_path.exists():
                continue
            metrics = read_json(metrics_path)
            if metrics.get("status") != "ok":
                continue
            theta_2 = float(metrics.get("theta_2", 0.5))
            stage2_by_key = {(p["clip_id"], p.get("window_index", 0)): p for p in read_jsonl(preds_path)}
            cascade_preds = []
            labels = []
            final_scores = []
            for p in stage1_preds:
                key = (p["clip_id"], p.get("window_index", 0))
                s2 = stage2_by_key.get(key)
                if not s2:
                    continue
                candidate = float(p["score"]) >= theta_1
                final_trigger = candidate and float(s2["stage2_score"]) >= theta_2
                row = dict(p)
                row["stage1_score"] = float(p["score"])
                row["stage2_score"] = float(s2["stage2_score"])
                row["candidate"] = candidate
                row["final_trigger"] = final_trigger
                cascade_preds.append(row)
                labels.append(int(p["label"]))
                final_scores.append(1.0 if final_trigger else 0.0)
            cascade_metrics = binary_metrics(labels, final_scores, 0.5)
            cascade_metrics.update(
                {
                    "status": "ok",
                    "variant": name,
                    "theta_1": theta_1,
                    "theta_2": theta_2,
                    "candidate_rate": sum(1 for p in stage1_preds if float(p["score"]) >= theta_1) / len(stage1_preds)
                    if stage1_preds
                    else None,
                    "stage2_rows_matched": len(cascade_preds),
                }
            )
            cascade_results[name] = cascade_metrics
            write_predictions_csv(cascade_dir / f"{name}_predictions.csv", cascade_preds)
        if cascade_results:
            summary["cascade"] = {
                "status": "ok",
                "variants": cascade_results,
                "theta_1": theta_1,
            }
    write_json(cascade_dir / "metrics.json", summary["cascade"])
    write_json(run_dir / "summary.json", summary)
    write_reproducibility(run_dir / "reproducibility.json", project_root, dataset_report, run_dir)
    report = [
        "# FINAL REPORT",
        "",
        "This is an engineering smoke report for the current incomplete dataset.",
        "",
        "## Dataset",
        f"- Fingerprint: `{summary['dataset_fingerprint']}`",
        f"- Windows: {summary['dataset']['manifest_windows']}",
        f"- Prompt groups: `{summary['dataset']['prompt_group_counts']}`",
        f"- Split mode: `{summary['dataset']['split_report']['split_mode']}`",
        "",
        "## Stage 1",
        f"- Status: `{summary['stage1'].get('status')}`",
        f"- Feature backend: `{summary['stage1'].get('feature_backend')}`",
        f"- Official openWakeWord used: {summary['stage1'].get('official_openwakeword_used')}",
        f"- Theta 1: {summary['stage1'].get('theta_1')}",
        f"- Test metrics: `{summary['stage1'].get('test_metrics')}`",
        "",
        "## Stage 2",
        f"- BCE verifier: `{summary['stage2_bce'].get('status')}`",
        f"- BCE + SupCon verifier: `{summary['stage2_bce_supcon'].get('status')}`",
        "",
        "## Cascade",
        f"- Status: `{summary['cascade'].get('status')}`",
        f"- Reason: {summary['cascade'].get('reason')}",
    ]
    (run_dir / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
