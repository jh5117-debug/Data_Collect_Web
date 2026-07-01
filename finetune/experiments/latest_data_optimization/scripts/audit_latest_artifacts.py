#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

from vigil_latest_opt.utils import read_json, read_jsonl, sha256_file, write_json


ROOT = Path("finetune/experiments/latest_data")
REPORTS = Path("finetune/experiments/latest_data_optimization/reports")


def line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def report_identity_scan(paths: list[Path]) -> dict[str, object]:
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
    suspicious = []
    for path in paths:
        if not path.exists() or path.suffix.lower() not in {".md", ".json", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        emails = sorted(set(email_re.findall(text)))
        if emails:
            suspicious.append({"path": str(path), "emails": emails[:5]})
    return {"email_like_values_found": bool(suspicious), "hits": suspicious}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    zip_path = Path("finetune/data/vigil_dataset_export_latest_readonly_20260626_042151.zip")
    balanced = ROOT / "shared/balanced_max100_latest_manifest.jsonl"
    folds = ROOT / "shared/latest_participant_folds_5fold.json"
    transcript = ROOT / "shared/qwen_transcript_cache_balanced_max100_latest.jsonl"
    stage1_features = ROOT / "runs/latest_feature_cache_2b78e211183d47fb/stage1/features_manifest.jsonl"
    qwen_features = ROOT / "runs/latest_feature_cache_2b78e211183d47fb/stage2_qwen_features/qwen_features_manifest.jsonl"
    processed_manifest = Path("finetune/data/processed/2b78e211183d47fb/manifest_all.jsonl")
    dataset = read_json(ROOT / "reports/latest_dataset_preparation.json")
    audio_rejections = read_json(ROOT / "reports/latest_audio_rejection_summary.json")
    balanced_summary = read_json(ROOT / "shared/latest_balanced_summary.json")
    protocol = read_json(ROOT / "reports/latest_protocol_validation.json")
    feature = read_json(ROOT / "reports/latest_feature_coverage_report.json")
    transcript_summary = read_json(ROOT / "shared/qwen_transcript_cache_balanced_max100_latest.summary.json")
    nested = read_json(ROOT / "reports/latest_nested_zero_shot_summary.json")
    fewshot = read_json(ROOT / "reports/latest_real_few_shot_summary.json")
    cost = read_json(ROOT / "reports/latest_compute_accuracy_tradeoff.json")

    source_reports = [
        ROOT / "reports/LATEST_PROFESSOR_MEETING_REPORT.md",
        ROOT / "reports/LATEST_NESTED_ZERO_SHOT_5FOLD_REPORT.md",
        ROOT / "reports/LATEST_STAGE2_OPERATING_POINT_REPORT.md",
        ROOT / "reports/LATEST_REAL_FEW_SHOT_ONBOARDING_REPORT.md",
        ROOT / "reports/LATEST_COMPUTE_ACCURACY_TRADEOFF.md",
        ROOT / "reports/LATEST_STAGE_ERROR_ANALYSIS.md",
        ROOT / "reports/latest_nested_zero_shot_summary.json",
        ROOT / "reports/latest_real_few_shot_summary.json",
        ROOT / "reports/latest_compute_accuracy_tradeoff.json",
        ROOT / "reports/latest_stage_error_summary.csv",
    ]
    audit = {
        "status": "ok",
        "zip_present": zip_path.exists(),
        "zip_sha256": sha256_file(zip_path) if zip_path.exists() else None,
        "raw_canonical_clips": dataset["dataset_report"]["canonical_metadata_rows"],
        "valid_unique_clips_after_qc": len({row["clip_id"] for row in read_jsonl(processed_manifest)}),
        "manifest_windows_after_qc": dataset["dataset_report"]["manifest_windows"],
        "audio_qc_rejected": audio_rejections["rejected_silent_count"],
        "balanced_manifest_rows": line_count(balanced),
        "balanced_clips": balanced_summary["clips_after"],
        "balanced_windows": balanced_summary["windows_after_cap"],
        "balanced_participants": balanced_summary["participants"],
        "balanced_manifest_sha256": sha256_file(balanced),
        "fold_sha256": sha256_file(folds),
        "participant_leakage_free": protocol["no_participant_crosses_outer_folds"],
        "duplicate_audio_leakage_free": protocol["no_duplicate_audio_hash_crosses_folds"],
        "qwen_transcript_cache_rows": line_count(transcript),
        "qwen_transcript_summary": transcript_summary,
        "stage1_feature_rows": line_count(stage1_features),
        "qwen_feature_rows": line_count(qwen_features),
        "feature_coverage": feature,
        "nested_status": nested.get("status"),
        "fewshot_status": fewshot.get("status"),
        "compute_status": cost.get("status"),
        "identity_scan": report_identity_scan(source_reports),
    }
    if audit["zip_sha256"] != "e2e38518d6725449653138e0ee484c4b5903467e418e8968d4b98ada5fd41701":
        audit["status"] = "zip_sha_mismatch"
    if audit["balanced_manifest_sha256"] != "549134e307f21470cb942acd44c2c27d2b29fcaa8527b9e7f8e2722e3232b58e":
        audit["status"] = "balanced_manifest_sha_mismatch"
    if audit["fold_sha256"] != "7c1c65da28f87922f111ee1549b61c053323fc876d2cd26346544de0b37b2a5e":
        audit["status"] = "fold_sha_mismatch"
    if audit["identity_scan"]["email_like_values_found"]:
        audit["status"] = "identity_scan_failed"
    write_json(REPORTS / "optimization_start_audit.json", audit)

    lines = [
        "# Optimization Start Audit",
        "",
        f"- Status: `{audit['status']}`",
        f"- Raw production clips: `{audit['raw_canonical_clips']}`",
        f"- Valid clips after audio QC: `{audit['valid_unique_clips_after_qc']}`",
        f"- Manifest windows after audio QC: `{audit['manifest_windows_after_qc']}`",
        f"- Audio QC rejected silent clips: `{audit['audio_qc_rejected']}`",
        f"- Balanced clips/windows/participants: `{audit['balanced_clips']}` / `{audit['balanced_windows']}` / `{audit['balanced_participants']}`",
        f"- Balanced manifest rows: `{audit['balanced_manifest_rows']}`",
        f"- Qwen transcript cache rows: `{audit['qwen_transcript_cache_rows']}`",
        f"- Feature coverage: Stage1 `{feature['stage1_covered']}/{feature['balanced_windows']}`, Qwen `{feature['qwen_covered']}/{feature['balanced_windows']}`",
        f"- Participant leakage free: `{audit['participant_leakage_free']}`",
        f"- Duplicate-audio leakage free: `{audit['duplicate_audio_leakage_free']}`",
        f"- Email-like identities in committed source reports: `{audit['identity_scan']['email_like_values_found']}`",
    ]
    (REPORTS / "OPTIMIZATION_START_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(audit["status"])
    return 0 if audit["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
