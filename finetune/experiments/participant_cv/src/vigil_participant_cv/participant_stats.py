from __future__ import annotations

import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .privacy import build_alias_map
from .utils import prompt_order, read_jsonl


CLIP_FIELDS = (
    "clip_id",
    "speaker_id",
    "session_id",
    "label",
    "prompt_group",
    "phrase_id",
    "transcript",
    "audio_sha256",
    "full_wav_sha256",
    "full_wav_path",
)


def load_manifest_rows(dataset_dir: Path | str) -> list[dict[str, Any]]:
    return read_jsonl(Path(dataset_dir) / "manifest_all.jsonl")


def dedupe_clips(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["clip_id"])].append(row)
    clips: list[dict[str, Any]] = []
    for clip_id, group in sorted(grouped.items()):
        first = sorted(group, key=lambda r: int(r.get("window_index", 0)))[0]
        clip = {key: first.get(key) for key in CLIP_FIELDS if key in first}
        clip["clip_id"] = clip_id
        clip["window_count"] = len(group)
        clip["window_rows"] = sorted(
            [
                {
                    "clip_id": row["clip_id"],
                    "window_index": int(row.get("window_index", 0)),
                    "window_audio_sha256": row.get("window_audio_sha256"),
                    "window_start_sec": row.get("window_start_sec"),
                    "window_end_sec": row.get("window_end_sec"),
                    "window_wav_path": row.get("window_wav_path"),
                }
                for row in group
            ],
            key=lambda r: int(r["window_index"]),
        )
        clips.append(clip)
    return clips


def wav_duration(path: str | None) -> float | None:
    if not path:
        return None
    try:
        with wave.open(path, "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return float(frames / rate) if rate else None
    except Exception:
        return None


def attach_aliases(clips: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    alias_map = build_alias_map(str(c["speaker_id"]) for c in clips)
    out: list[dict[str, Any]] = []
    for clip in clips:
        item = dict(clip)
        item["participant_alias"] = alias_map[str(clip["speaker_id"])]
        item["duration_sec"] = wav_duration(str(clip.get("full_wav_path") or ""))
        out.append(item)
    return out, alias_map


def participant_statistics(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_alias: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        by_alias[str(clip["participant_alias"])].append(clip)
    rows: list[dict[str, Any]] = []
    for alias, items in sorted(by_alias.items()):
        prompt = Counter(str(c.get("prompt_group")) for c in items)
        phrase = Counter(str(c.get("phrase_id")) for c in items if str(c.get("prompt_group")) == "P4_negative")
        positives = sum(int(c.get("label", 0)) == 1 for c in items)
        negatives = sum(int(c.get("label", 0)) == 0 for c in items)
        duplicate_hashes = sum(count - 1 for count in Counter(str(c.get("full_wav_sha256") or c.get("audio_sha256")) for c in items).values() if count > 1)
        missing = sum(
            1
            for c in items
            if not c.get("clip_id") or not c.get("speaker_id") or c.get("label") not in (0, 1) or not c.get("prompt_group")
        )
        rows.append(
            {
                "participant_alias": alias,
                "total_unique_clips": len(items),
                "total_windows": sum(int(c.get("window_count", 1)) for c in items),
                "positive_clips": positives,
                "negative_clips": negatives,
                "P1_vigil_only": prompt["P1_vigil_only"],
                "P2_phrase_plus_vigil": prompt["P2_phrase_plus_vigil"],
                "P3_vigil_plus_phrase": prompt["P3_vigil_plus_phrase"],
                "P4_negative": prompt["P4_negative"],
                "hard_negative_phrase_counts": dict(sorted(phrase.items())),
                "sessions": len({str(c.get("session_id")) for c in items}),
                "audio_duration_sec": round(sum(float(c.get("duration_sec") or 0.0) for c in items), 3),
                "duplicate_audio_hash_count": duplicate_hashes,
                "missing_or_invalid_metadata": missing,
                "eligible_3_shot": positives >= 4 and negatives >= 1,
                "eligible_5_shot": positives >= 6 and negatives >= 1,
                "both_positive_and_negative_query_examples": positives >= 1 and negatives >= 1,
            }
        )
    return rows


def duplicate_audio_groups(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        key = str(clip.get("full_wav_sha256") or clip.get("audio_sha256") or "")
        if key:
            grouped[key].append(clip)
    groups: list[dict[str, Any]] = []
    for audio_hash, items in sorted(grouped.items()):
        if len(items) <= 1:
            continue
        aliases = sorted({str(c["participant_alias"]) for c in items})
        groups.append(
            {
                "audio_hash": audio_hash,
                "clip_ids": sorted(str(c["clip_id"]) for c in items),
                "participant_aliases": aliases,
                "participant_count": len(aliases),
                "clip_count": len(items),
                "cross_participant": len(aliases) > 1,
            }
        )
    return groups


def sanitized_clip_rows(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for clip in sorted(clips, key=lambda c: (str(c["participant_alias"]), prompt_order(str(c.get("prompt_group"))), str(c["clip_id"]))):
        row = {
            "participant_alias": clip["participant_alias"],
            "clip_id": clip["clip_id"],
            "label": int(clip["label"]),
            "prompt_group": clip.get("prompt_group"),
            "phrase_id": clip.get("phrase_id"),
            "window_count": int(clip.get("window_count", 1)),
            "full_wav_sha256": clip.get("full_wav_sha256"),
            "audio_sha256": clip.get("audio_sha256"),
            "duration_sec": clip.get("duration_sec"),
        }
        out.append(row)
    return out
