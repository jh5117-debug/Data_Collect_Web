from __future__ import annotations

from collections import defaultdict
from typing import Any


class ProtocolError(ValueError):
    pass


def validate_unique_fold_membership(folds: dict[str, Any]) -> None:
    seen: dict[str, int] = {}
    for fold in folds["folds"]:
        fold_id = int(fold["fold"])
        for alias in fold["participant_aliases"]:
            if alias in seen:
                raise ProtocolError(f"participant {alias} appears in folds {seen[alias]} and {fold_id}")
            seen[alias] = fold_id


def validate_no_duplicate_hash_crosses_folds(clips: list[dict[str, Any]], folds: dict[str, Any]) -> None:
    alias_to_fold = {alias: int(fold["fold"]) for fold in folds["folds"] for alias in fold["participant_aliases"]}
    hashes: dict[str, set[int]] = defaultdict(set)
    for clip in clips:
        key = str(clip.get("full_wav_sha256") or clip.get("audio_sha256") or "")
        if key:
            hashes[key].add(alias_to_fold[str(clip["participant_alias"])])
    crossing = {key: sorted(value) for key, value in hashes.items() if len(value) > 1}
    if crossing:
        raise ProtocolError(f"duplicate audio hashes cross folds: {list(crossing.items())[:5]}")
