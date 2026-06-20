from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from .utils import short_hash


def speaker_hash(raw_key: str) -> str:
    return "spk_" + short_hash(raw_key, 12)


def assign_splits(rows: list[dict[str, Any]], seed: int = 20260620) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["speaker_id"] = speaker_hash(str(row.get("participant_key") or "unknown"))
        by_speaker[row["speaker_id"]].append(row)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    split_mode = "speaker_disjoint"
    if len(speakers) >= 5:
        n = len(speakers)
        n_train = max(1, round(n * 0.70))
        n_val = max(1, round(n * 0.15))
        train_spk = set(speakers[:n_train])
        val_spk = set(speakers[n_train : n_train + n_val])
        test_spk = set(speakers[n_train + n_val :])
        if not test_spk:
            test_spk.add(val_spk.pop())
    elif len(speakers) >= 3:
        test_spk = {speakers[0]}
        val_spk = {speakers[1]}
        train_spk = set(speakers[2:])
    elif len(speakers) == 2:
        test_spk = {speakers[0]}
        val_spk = set()
        train_spk = {speakers[1]}
        split_mode = "speaker_disjoint_test_only"
    else:
        train_spk = val_spk = test_spk = set()
        split_mode = "SMOKE_ONLY_NO_SPEAKER_GENERALIZATION"
    if split_mode == "SMOKE_ONLY_NO_SPEAKER_GENERALIZATION":
        clip_ids = sorted({r["clip_id"] for r in rows})
        rng.shuffle(clip_ids)
        n = len(clip_ids)
        val_cut = max(1, n // 5)
        test_cut = max(2, (n * 2) // 5)
        val_clips = set(clip_ids[:val_cut])
        test_clips = set(clip_ids[val_cut:test_cut])
        for row in rows:
            row["split"] = "val" if row["clip_id"] in val_clips else "test" if row["clip_id"] in test_clips else "train"
    else:
        for row in rows:
            spk = row["speaker_id"]
            row["split"] = "train" if spk in train_spk else "val" if spk in val_spk else "test"
        if not any(r["split"] == "val" for r in rows):
            train_clip_ids = sorted({r["clip_id"] for r in rows if r["split"] == "train"})
            rng.shuffle(train_clip_ids)
            val_clips = set(train_clip_ids[: max(1, len(train_clip_ids) // 5)])
            for row in rows:
                if row["clip_id"] in val_clips:
                    row["split"] = "val"
    enforce_duplicate_hash_splits(rows)
    report = {
        "split_mode": split_mode,
        "speaker_count": len(speakers),
        "speakers_by_split": {
            split: sorted({r["speaker_id"] for r in rows if r["split"] == split}) for split in ("train", "val", "test")
        },
        "clips_by_split": dict(Counter(r["split"] for r in rows)),
        "labels_by_split": {
            split: dict(Counter(r["label"] for r in rows if r["split"] == split)) for split in ("train", "val", "test")
        },
    }
    return rows, report


def enforce_duplicate_hash_splits(rows: list[dict[str, Any]]) -> None:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        h = row.get("audio_sha256")
        if h:
            by_hash[h].append(row)
    priority = {"train": 0, "val": 1, "test": 2}
    for group in by_hash.values():
        splits = {r.get("split") for r in group}
        if len(splits) <= 1:
            continue
        target = max((r["split"] for r in group), key=lambda s: priority.get(s, -1))
        for row in group:
            row["split"] = target


def assert_no_speaker_leakage(rows: list[dict[str, Any]]) -> bool:
    by_speaker: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_speaker[row["speaker_id"]].add(row["split"])
    return all(len(splits) <= 1 for splits in by_speaker.values())


def assert_no_duplicate_audio_leakage(rows: list[dict[str, Any]]) -> bool:
    by_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("audio_sha256"):
            by_hash[row["audio_sha256"]].add(row["split"])
    return all(len(splits) <= 1 for splits in by_hash.values())
