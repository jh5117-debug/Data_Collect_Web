#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vigil_participant_cv.balancing import balance_max_clips_per_participant
from vigil_participant_cv.participant_stats import attach_aliases, dedupe_clips, duplicate_audio_groups, load_manifest_rows
from vigil_participant_cv.privacy import assert_public_text_is_sanitized
from vigil_participant_cv.utils import ensure_dir, sha256_file, write_json, write_jsonl


def public_window_rows(selected_clips: list[dict], manifest_rows: list[dict]) -> list[dict]:
    selected_ids = {str(c["clip_id"]) for c in selected_clips}
    alias_by_clip = {str(c["clip_id"]): c["participant_alias"] for c in selected_clips}
    out = []
    for row in sorted(manifest_rows, key=lambda r: (alias_by_clip.get(str(r["clip_id"]), ""), str(r["clip_id"]), int(r.get("window_index", 0)))):
        clip_id = str(row["clip_id"])
        if clip_id not in selected_ids:
            continue
        item = {
            "participant_alias": alias_by_clip[clip_id],
            "clip_id": clip_id,
            "window_index": int(row.get("window_index", 0)),
            "label": int(row["label"]),
            "prompt_group": row.get("prompt_group"),
            "phrase_id": row.get("phrase_id"),
            "transcript": row.get("transcript"),
            "normalized_transcript": row.get("normalized_transcript"),
            "full_wav_path": row.get("full_wav_path"),
            "window_wav_path": row.get("window_wav_path"),
            "full_wav_sha256": row.get("full_wav_sha256"),
            "audio_sha256": row.get("audio_sha256"),
            "window_audio_sha256": row.get("window_audio_sha256"),
            "window_start_sec": row.get("window_start_sec"),
            "window_end_sec": row.get("window_end_sec"),
        }
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", default="finetune/experiments/latest_data")
    parser.add_argument("--max-clips", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    shared = ensure_dir(out_dir / "shared")
    reports = ensure_dir(out_dir / "reports")
    manifest_rows = load_manifest_rows(args.dataset_dir)
    clips, _alias_map = attach_aliases(dedupe_clips(manifest_rows))
    duplicate_groups = duplicate_audio_groups(clips)
    excluded_ids: set[str] = set()
    for group in duplicate_groups:
        if group["cross_participant"]:
            excluded_ids.update(str(clip_id) for clip_id in group["clip_ids"])
        else:
            for clip_id in sorted(str(clip_id) for clip_id in group["clip_ids"])[1:]:
                excluded_ids.add(clip_id)
    formal_clips = [clip for clip in clips if str(clip["clip_id"]) not in excluded_ids]
    full_rows = public_window_rows(formal_clips, manifest_rows)
    write_jsonl(shared / "full_unbalanced_latest_manifest.jsonl", full_rows)
    selected, summary = balance_max_clips_per_participant(formal_clips, max_clips=args.max_clips, seed=args.seed)
    rows = public_window_rows(selected, manifest_rows)
    manifest_path = shared / "balanced_max100_latest_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    summary.update(
        {
            "source_unique_clips": len(clips),
            "formal_valid_clips_after_duplicate_policy": len(formal_clips),
            "excluded_duplicate_clip_ids": sorted(excluded_ids),
            "windows_after_cap": len(rows),
            "balanced_manifest_sha256": sha256_file(manifest_path),
            "full_unbalanced_windows": len(full_rows),
        }
    )
    write_json(shared / "latest_balanced_summary.json", summary)
    prompt = Counter(row["prompt_group"] for row in rows)
    label_rows = {r["clip_id"]: r for r in rows}.values()
    labels = Counter(str(row["label"]) for row in label_rows)
    report = [
        "# Latest Balanced Max-100 Dataset",
        "",
        f"- Source unique clips: `{summary['source_unique_clips']}`",
        f"- Formal valid clips after duplicate policy: `{summary['formal_valid_clips_after_duplicate_policy']}`",
        f"- Clips after cap: `{summary['clips_after']}`",
        f"- Windows after cap: `{summary['windows_after_cap']}`",
        f"- Participants: `{summary['participants']}`",
        f"- Prompt counts after cap: `{dict(sorted(prompt.items()))}`",
        f"- Label counts after cap: `{dict(sorted(labels.items()))}`",
        f"- Manifest SHA-256: `{summary['balanced_manifest_sha256']}`",
    ]
    text = "\n".join(report) + "\n"
    assert_public_text_is_sanitized(text)
    (reports / "LATEST_BALANCED_MAX100_REPORT.md").write_text(text, encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "sha256": summary["balanced_manifest_sha256"], "clips": summary["clips_after"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
