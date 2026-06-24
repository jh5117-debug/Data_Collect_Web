#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
PROMPT_GROUP_ORDER = (
    "P1_vigil_only",
    "P2_phrase_plus_vigil",
    "P3_vigil_plus_phrase",
    "P4_negative",
)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def load_split_rows(dataset_dir: Path | str, split: str) -> list[dict[str, Any]]:
    path = Path(dataset_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_jsonl(path)


def load_all_rows(dataset_dir: Path | str) -> list[dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    manifest_all = dataset_dir / "manifest_all.jsonl"
    if manifest_all.exists():
        return read_jsonl(manifest_all)
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_path = dataset_dir / f"{split}.jsonl"
        if split_path.exists():
            rows.extend(read_jsonl(split_path))
    return sorted(rows, key=lambda r: (str(r.get("split", "")), str(r.get("clip_id", "")), int(r.get("window_index", 0))))


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row.get("label", 0)) for row in rows)
    return {"positive": int(counts.get(1, 0)), "negative": int(counts.get(0, 0))}


def _clip_representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reps = []
    for clip_id, group in sorted(group_rows(rows, "clip_id").items()):
        rep = dict(sorted(group, key=lambda r: int(r.get("window_index", 0)))[0])
        rep["_clip_window_count"] = len(group)
        rep["_clip_id"] = clip_id
        reps.append(rep)
    return reps


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda x: str(x))}


def duplicate_audio_hash_groups(rows: list[dict[str, Any]], hash_key: str = "audio_sha256") -> dict[str, list[str]]:
    by_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row.get(hash_key)
        if value:
            by_hash[str(value)].add(str(row.get("clip_id", "")))
    return {h: sorted(clips) for h, clips in sorted(by_hash.items()) if len(clips) > 1}


def split_leakage(rows: list[dict[str, Any]], key: str) -> dict[str, list[str]]:
    by_key: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row.get(key)
        if value:
            by_key[str(value)].add(str(row.get("split", "")))
    return {value: sorted(splits) for value, splits in sorted(by_key.items()) if len(splits) > 1}


def _clip_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    inconsistent: dict[str, list[str]] = {"split": [], "label": [], "speaker_id": []}
    for clip_id, group in group_rows(rows, "clip_id").items():
        for key in inconsistent:
            values = {str(row.get(key, "")) for row in group}
            if len(values) > 1:
                inconsistent[key].append(clip_id)
    return {
        "all_windows_from_one_clip_in_one_split": not inconsistent["split"],
        "all_windows_from_one_clip_same_label": not inconsistent["label"],
        "all_windows_from_one_clip_same_speaker": not inconsistent["speaker_id"],
        "inconsistent_clip_ids": {key: sorted(values) for key, values in inconsistent.items()},
    }


def _manifest_consistency(dataset_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_field_mismatches = {
        split: [str(row.get("clip_id", "")) for row in rows if row.get("split") != split]
        for split, rows in rows_by_split.items()
    }
    keys = [(row.get("clip_id"), int(row.get("window_index", 0))) for row in all_rows]
    duplicate_window_keys = [f"{clip_id}:w{window_index}" for (clip_id, window_index), count in Counter(keys).items() if count > 1]
    manifest_all_path = dataset_dir / "manifest_all.jsonl"
    manifest_all_matches_split_union = None
    if manifest_all_path.exists():
        union_keys = sorted((row.get("clip_id"), int(row.get("window_index", 0)), row.get("split")) for rows in rows_by_split.values() for row in rows)
        all_keys = sorted((row.get("clip_id"), int(row.get("window_index", 0)), row.get("split")) for row in all_rows)
        manifest_all_matches_split_union = union_keys == all_keys
    return {
        "split_field_mismatches": {split: values for split, values in split_field_mismatches.items() if values},
        "duplicate_window_keys": duplicate_window_keys,
        "manifest_all_matches_split_union": manifest_all_matches_split_union,
        "train_val_test_manifests_consistent": not any(split_field_mismatches.values())
        and not duplicate_window_keys
        and manifest_all_matches_split_union is not False,
    }


def _counts_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clip_rows = _clip_representatives(rows)
    window_prompt_counts = Counter(row.get("prompt_group", "") for row in rows)
    clip_prompt_counts = Counter(row.get("prompt_group", "") for row in clip_rows)
    hard_negative_phrase_counts = Counter(str(row.get("phrase_id", "")) for row in clip_rows if int(row.get("label", 0)) == 0)
    duplicate_groups = duplicate_audio_hash_groups(rows)
    return {
        "speakers": len({row.get("speaker_id") for row in rows if row.get("speaker_id")}),
        "participants": len({row.get("speaker_id") for row in rows if row.get("speaker_id")}),
        "participants_note": "Raw participant IDs are intentionally dropped; participant count is the unique redacted speaker_id count.",
        "sessions": len({row.get("session_id") for row in rows if row.get("session_id")}),
        "unique_original_clips": len(clip_rows),
        "windows": len(rows),
        "window_labels": _label_counts(rows),
        "clip_labels": _label_counts(clip_rows),
        "positives": _label_counts(rows)["positive"],
        "negatives": _label_counts(rows)["negative"],
        "prompt_group_windows": {name: int(window_prompt_counts.get(name, 0)) for name in PROMPT_GROUP_ORDER},
        "prompt_group_clips": {name: int(clip_prompt_counts.get(name, 0)) for name in PROMPT_GROUP_ORDER},
        "P1": int(window_prompt_counts.get("P1_vigil_only", 0)),
        "P2": int(window_prompt_counts.get("P2_phrase_plus_vigil", 0)),
        "P3": int(window_prompt_counts.get("P3_vigil_plus_phrase", 0)),
        "P4": int(window_prompt_counts.get("P4_negative", 0)),
        "hard_negative_phrase_ids": _counter_dict(hard_negative_phrase_counts),
        "duplicate_audio_hash_groups": len(duplicate_groups),
        "duplicate_audio_hash_clips": int(sum(len(clips) for clips in duplicate_groups.values())),
    }


def build_split_report(dataset_dir: Path | str) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    rows_by_split = {split: load_split_rows(dataset_dir, split) for split in SPLITS}
    all_rows = load_all_rows(dataset_dir)
    all_from_splits = [row for split in SPLITS for row in rows_by_split[split]]
    speaker_leaks = split_leakage(all_from_splits, "speaker_id")
    session_leaks = split_leakage(all_from_splits, "session_id")
    audio_hash_leaks = split_leakage(all_from_splits, "audio_sha256")
    full_wav_hash_leaks = split_leakage(all_from_splits, "full_wav_sha256")
    duplicate_groups_all = duplicate_audio_hash_groups(all_from_splits)
    return {
        "dataset_dir": str(dataset_dir),
        "splits": {split: _counts_for_rows(rows) for split, rows in rows_by_split.items()},
        "all": _counts_for_rows(all_from_splits),
        "validations": {
            "no_speaker_leakage": not speaker_leaks,
            "speaker_leakage": speaker_leaks,
            "no_session_leakage": not session_leaks,
            "session_leakage": session_leaks,
            "no_duplicate_audio_leakage": not audio_hash_leaks and not full_wav_hash_leaks,
            "duplicate_audio_leakage": audio_hash_leaks,
            "duplicate_full_wav_leakage": full_wav_hash_leaks,
            "clip_consistency": _clip_consistency(all_from_splits),
            "manifest_consistency": _manifest_consistency(dataset_dir, rows_by_split, all_rows),
        },
        "duplicate_audio_hash_groups_all": len(duplicate_groups_all),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = build_split_report(args.dataset_dir)
    if args.output:
        write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

