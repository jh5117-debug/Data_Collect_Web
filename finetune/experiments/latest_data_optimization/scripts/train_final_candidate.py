#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from vigil_latest_opt.utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl


def duplicate_for_train_val(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for split in ("train", "val"):
        for row in rows:
            copied = dict(row)
            copied["split"] = split
            out.append(copied)
    return out


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(cmd)}")


def checkpoint_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path) if path.exists() else None, "exists": path.exists(), "committed": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", default="")
    parser.add_argument("--config", default="finetune/configs/full.yaml")
    parser.add_argument("--dataset-dir", default="finetune/data/processed/2b78e211183d47fb")
    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bundle = Path(args.bundle_root or f"finetune/model_bundles/vigil_latest_optimized_{timestamp}")
    bundle.mkdir(parents=True, exist_ok=True)
    reports = Path("finetune/experiments/latest_data_optimization/reports")
    selected = read_json(reports / "latest_opt_stage2_selected_config.json")
    balanced_rows = read_jsonl("finetune/experiments/latest_data/shared/balanced_max100_latest_manifest.jsonl")
    balanced_ids = {row["clip_id"] for row in balanced_rows}
    stage1_rows = [row for row in read_jsonl("finetune/experiments/latest_data/runs/latest_feature_cache_2b78e211183d47fb/stage1/features_manifest.jsonl") if row["clip_id"] in balanced_ids]
    qwen_rows = [row for row in read_jsonl("finetune/experiments/latest_data/runs/latest_feature_cache_2b78e211183d47fb/stage2_qwen_features/qwen_features_manifest.jsonl") if row["clip_id"] in balanced_ids]
    stage1_manifest = bundle / "stage1_features_manifest.jsonl"
    stage2_dir = bundle / "stage2_qwen_features"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage2_manifest = stage2_dir / "qwen_features_manifest.jsonl"
    write_jsonl(stage1_manifest, duplicate_for_train_val(stage1_rows))
    write_jsonl(stage2_manifest, duplicate_for_train_val(qwen_rows))
    integrity = Path("finetune/experiments/latest_data/runs/latest_feature_cache_2b78e211183d47fb/stage2_qwen_features/frozen_qwen_integrity.json")
    if integrity.exists():
        shutil.copy2(integrity, stage2_dir / "frozen_qwen_integrity.json")
    run(["python", "finetune/scripts/train_stage1.py", "--features-manifest", str(stage1_manifest), "--config", args.config, "--run-dir", str(bundle)])
    run(["python", "finetune/scripts/train_stage2.py", "--dataset-dir", args.dataset_dir, "--config", args.config, "--run-dir", str(bundle), "--variant", "bce_supcon"])
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    manifest = {
        "status": "trained_deployment_candidate_not_scientific_test",
        "bundle_dir": str(bundle),
        "code_commit": commit,
        "include_qwen_weights": False,
        "dataset_fingerprint": "2b78e211183d47fb",
        "dataset_zip_sha256": "e2e38518d6725449653138e0ee484c4b5903467e418e8968d4b98ada5fd41701",
        "balanced_manifest_sha256": "549134e307f21470cb942acd44c2c27d2b29fcaa8527b9e7f8e2722e3232b58e",
        "fold_sha256": "7c1c65da28f87922f111ee1549b61c053323fc876d2cd26346544de0b37b2a5e",
        "selected_config": {
            "variant": selected["variant"],
            "top_k": selected["top_k"],
            "fewshot_recipe": "no_adaptation_zero_shot_fallback",
        },
        "training_policy": "All balanced windows duplicated as train and val for deployment-only final fit/calibration; not a held-out scientific estimate.",
        "stage1_checkpoint": checkpoint_entry(bundle / "stage1/checkpoint_best.pt"),
        "stage2_checkpoint": checkpoint_entry(bundle / "stage2_bce_supcon/checkpoint_best.pt"),
        "stage1_threshold": read_json(bundle / "stage1/threshold.json"),
        "stage2_threshold": read_json(bundle / "stage2_bce_supcon/threshold.json"),
        "stage1_metrics": read_json(bundle / "stage1/metrics.json"),
        "stage2_metrics": read_json(bundle / "stage2_bce_supcon/metrics.json"),
    }
    write_json(bundle / "PUBLIC_MANIFEST.json", manifest)
    public_manifest = dict(manifest)
    public_manifest["stage1_checkpoint"] = {k: manifest["stage1_checkpoint"][k] for k in ("sha256", "exists", "committed")}
    public_manifest["stage2_checkpoint"] = {k: manifest["stage2_checkpoint"][k] for k in ("sha256", "exists", "committed")}
    write_json(reports / "latest_opt_final_model_manifest.json", public_manifest)
    lines = [
        "# Latest Optimized Final Model Report",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Bundle dir: `{bundle}`",
        "- Qwen weights included: `False`",
        f"- Stage1 checkpoint exists/committed: `{manifest['stage1_checkpoint']['exists']}` / `False`",
        f"- Stage2 checkpoint exists/committed: `{manifest['stage2_checkpoint']['exists']}` / `False`",
        f"- Training policy: {manifest['training_policy']}",
        f"- Stage1 theta: `{manifest['stage1_threshold'].get('threshold')}`",
        f"- Stage2 theta: `{manifest['stage2_threshold'].get('threshold')}`",
    ]
    (reports / "LATEST_OPT_FINAL_MODEL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": manifest["status"], "bundle_dir": str(bundle)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
