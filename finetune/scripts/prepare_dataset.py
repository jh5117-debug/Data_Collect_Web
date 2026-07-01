#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from vigil_two_stage.audio import (
    detect_speech_bounds,
    ffmpeg_convert_to_wav,
    fixed_window_bounds,
    materialize_window,
    read_wav,
    validate_wav,
    write_wav,
)
from vigil_two_stage.export_parser import canonical_samples, load_export
from vigil_two_stage.manifests import write_split_manifests
from vigil_two_stage.splits import assert_no_duplicate_audio_leakage, assert_no_speaker_leakage, assign_splits
from vigil_two_stage.utils import ensure_dir, sha256_bytes, sha256_file, short_hash, stable_json, write_json, write_jsonl


def dataset_fingerprint(zip_sha: str, config: dict) -> str:
    relevant = {
        "zip_sha256": zip_sha,
        "audio": config.get("audio", {}),
        "data": config.get("data", {}),
        "seed": config.get("seed"),
        "pipeline": "vigil_two_stage_prepare_v1",
    }
    return short_hash(stable_json(relevant), 16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="finetune/data/processed")
    args = parser.parse_args()
    zip_path = Path(args.zip_path)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    zip_sha = sha256_file(zip_path)
    fp = dataset_fingerprint(zip_sha, config)
    dataset_dir = ensure_dir(Path(args.output_root) / fp)
    raw_dir = ensure_dir(dataset_dir / "canonical_raw")
    wav_dir = ensure_dir(dataset_dir / "full_wav")
    window_dir = ensure_dir(dataset_dir / "windows")

    bundle = load_export(zip_path)
    samples, rejected = canonical_samples(bundle, config["data"]["hard_negative_phrases"])
    rows = []
    qc_rows = []
    duplicate_hashes: dict[str, list[str]] = defaultdict(list)
    sample_rate = int(config["audio"]["sample_rate"])
    window_seconds = float(config["audio"]["window_seconds"])

    with zipfile.ZipFile(zip_path) as zf:
        for sample in samples:
            clip_id = sample["clip_id"]
            suffix = Path(sample["canonical_audio_member"]).suffix or ".webm"
            raw_path = raw_dir / f"{clip_id}{suffix}"
            raw_bytes = zf.read(sample["canonical_audio_member"])
            original_sha = sha256_bytes(raw_bytes)
            duplicate_hashes[original_sha].append(clip_id)
            if not raw_path.exists() or sha256_file(raw_path) != original_sha:
                raw_path.write_bytes(raw_bytes)
            full_wav = wav_dir / f"{clip_id}.wav"
            conversion = {"ok": full_wav.exists() and validate_wav(full_wav, sample_rate).get("ok", False), "reused": True}
            if not conversion["ok"]:
                conversion = ffmpeg_convert_to_wav(raw_path, full_wav, sample_rate)
            wav_qc = validate_wav(full_wav, sample_rate)
            qc = {
                "clip_id": clip_id,
                "conversion_ok": bool(conversion["ok"]),
                "wav_ok": bool(wav_qc.get("ok")),
                "conversion_error": conversion.get("stderr", "") if not conversion["ok"] else "",
                **{f"wav_{k}": v for k, v in wav_qc.items() if k != "sha256"},
                "original_audio_sha256": original_sha,
                "full_wav_sha256": wav_qc.get("sha256"),
            }
            qc_rows.append(qc)
            if not conversion["ok"] or not wav_qc.get("ok"):
                rejected.append({"clip_id": clip_id, "reasons": ["audio_conversion_or_validation_failed"], "detail": qc})
                continue
            sr, audio = read_wav(full_wav)
            speech_begin, speech_end = detect_speech_bounds(audio, sr)
            bounds = fixed_window_bounds(sample["prompt_group"], speech_begin, speech_end, audio.shape[0], sr, window_seconds)
            for window_index, (start, end, heuristic) in enumerate(bounds):
                win_audio, left_pad, right_pad = materialize_window(audio, start, end)
                window_path = window_dir / f"{clip_id}_w{window_index:02d}.wav"
                write_wav(window_path, sr, win_audio)
                window_sha = sha256_file(window_path)
                row = {
                    "clip_id": clip_id,
                    "speaker_id": "",
                    "session_id": sample.get("session_id") or "",
                    "prompt_group": sample["prompt_group"],
                    "prompt_title": sample["prompt_title"],
                    "transcript": sample["transcript"],
                    "normalized_transcript": sample["normalized_transcript"],
                    "label": int(sample["label"]),
                    "contains_vigil": bool(sample["contains_vigil"]),
                    "wake_intent": bool(sample["wake_intent"]),
                    "is_negative": bool(sample["is_negative"]),
                    "phrase_id": sample["phrase_id"],
                    "full_wav_path": str(full_wav.resolve()),
                    "window_wav_path": str(window_path.resolve()),
                    "window_index": window_index,
                    "window_start_sec": round(start / sr, 4),
                    "window_end_sec": round(end / sr, 4),
                    "left_padding_samples": int(left_pad),
                    "right_padding_samples": int(right_pad),
                    "window_heuristic": heuristic,
                    "audio_sha256": original_sha,
                    "full_wav_sha256": wav_qc.get("sha256"),
                    "window_audio_sha256": window_sha,
                    "participant_key": sample.get("participant_key", "unknown"),
                }
                rows.append(row)
    rows, split_report = assign_splits(rows, int(config["seed"]))
    for row in rows:
        row.pop("participant_key", None)
    write_split_manifests(dataset_dir, rows)
    write_jsonl(dataset_dir / "qc_report.jsonl", qc_rows)
    write_jsonl(dataset_dir / "rejected_or_inconsistent.jsonl", rejected)
    report = {
        "dataset_fingerprint": fp,
        "zip_sha256": zip_sha,
        "source_zip": str(zip_path),
        "canonical_metadata_rows": len(samples),
        "manifest_windows": len(rows),
        "rejected_or_inconsistent": len(rejected),
        "prompt_group_counts": dict(Counter(r["prompt_group"] for r in rows)),
        "label_counts": dict(Counter(r["label"] for r in rows)),
        "split_report": split_report,
        "speaker_leakage_free": assert_no_speaker_leakage(rows) if split_report["split_mode"] != "SMOKE_ONLY_NO_SPEAKER_GENERALIZATION" else None,
        "duplicate_audio_leakage_free": assert_no_duplicate_audio_leakage(rows),
        "duplicate_original_audio_hashes": {h: ids for h, ids in duplicate_hashes.items() if len(ids) > 1},
        "config": config,
    }
    write_json(dataset_dir / "dataset_report.json", report)
    md = [
        "# Dataset Report",
        "",
        f"- Dataset fingerprint: `{fp}`",
        f"- Source ZIP SHA-256: `{zip_sha}`",
        f"- Canonical clip rows accepted before audio QC: {len(samples)}",
        f"- Manifest windows: {len(rows)}",
        f"- Rejected or inconsistent rows: {len(rejected)}",
        f"- Prompt groups: `{report['prompt_group_counts']}`",
        f"- Labels: `{report['label_counts']}`",
        f"- Split mode: `{split_report['split_mode']}`",
        f"- Duplicate audio leakage free: {report['duplicate_audio_leakage_free']}",
        "",
        "Raw participant identities are not written to generated manifests.",
    ]
    (dataset_dir / "dataset_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(dataset_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
