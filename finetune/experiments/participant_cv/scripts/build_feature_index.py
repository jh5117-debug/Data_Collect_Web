#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vigil_participant_cv.utils import read_jsonl, write_json


def key(row: dict) -> tuple[str, int, str]:
    return str(row["clip_id"]), int(row.get("window_index", 0)), str(row.get("window_audio_sha256"))


def validate_npz(path: str) -> dict:
    arr = np.load(path)
    name = "features" if "features" in arr else arr.files[0]
    data = arr[name]
    return {
        "shape": list(data.shape),
        "finite": bool(np.isfinite(data).all()),
        "has_nan": bool(np.isnan(data).any()),
        "has_inf": bool(np.isinf(data).any()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-manifest", default="finetune/experiments/participant_cv/shared/balanced_max100_manifest.jsonl")
    parser.add_argument("--stage1-manifest", default="finetune/runs/20260624_075127_0fad4c7828149099_full/stage1/features_manifest.jsonl")
    parser.add_argument("--qwen-manifest", default="finetune/runs/20260624_075127_0fad4c7828149099_full/stage2_qwen_features/qwen_features_manifest.jsonl")
    parser.add_argument("--out", default="finetune/experiments/participant_cv/reports/feature_coverage_report.json")
    args = parser.parse_args()
    balanced = read_jsonl(args.balanced_manifest)
    stage1 = {key(row): row for row in read_jsonl(args.stage1_manifest)}
    qwen = {key(row): row for row in read_jsonl(args.qwen_manifest)}
    missing_stage1 = []
    missing_qwen = []
    checked = []
    for row in balanced:
        k = key(row)
        if k not in stage1:
            missing_stage1.append(row["clip_id"])
            continue
        if k not in qwen:
            missing_qwen.append(row["clip_id"])
            continue
        checked.append((stage1[k], qwen[k]))
    sample_stage1 = validate_npz(checked[0][0]["feature_path"]) if checked else None
    sample_qwen = validate_npz(checked[0][1]["feature_path"]) if checked else None
    result = {
        "status": "ok" if not missing_stage1 and not missing_qwen else "missing_features",
        "balanced_windows": len(balanced),
        "stage1_covered": len(balanced) - len(missing_stage1),
        "qwen_covered": len(balanced) - len(missing_qwen),
        "missing_stage1_clip_ids": sorted(set(missing_stage1)),
        "missing_qwen_clip_ids": sorted(set(missing_qwen)),
        "stage1_feature_backend": checked[0][0].get("feature_backend") if checked else None,
        "qwen_feature_backend": checked[0][1].get("feature_backend") if checked else None,
        "qwen_model_name": checked[0][1].get("qwen_model_name") if checked else None,
        "stage1_sample": sample_stage1,
        "qwen_sample": sample_qwen,
    }
    write_json(args.out, result)
    md = [
        "# Feature Coverage Report",
        "",
        f"- Status: `{result['status']}`",
        f"- Balanced windows: `{result['balanced_windows']}`",
        f"- openWakeWord covered: `{result['stage1_covered']}`",
        f"- Qwen audio-encoder covered: `{result['qwen_covered']}`",
        f"- Stage 1 backend: `{result['stage1_feature_backend']}`",
        f"- Qwen backend: `{result['qwen_feature_backend']}`",
        f"- Qwen model: `{result['qwen_model_name']}`",
    ]
    Path(args.out).with_name("FEATURE_COVERAGE_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)[:2000])
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
