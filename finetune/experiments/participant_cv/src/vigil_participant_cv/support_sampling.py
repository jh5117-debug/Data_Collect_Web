from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


def support_sort_key(seed: int, clip: dict[str, Any]) -> str:
    text = f"{seed}:{clip.get('participant_alias')}:{clip.get('prompt_group')}:{clip.get('clip_id')}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_positive_support(clips: list[dict[str, Any]], *, k: int, seed: int) -> list[dict[str, Any]]:
    positives = [clip for clip in clips if int(clip.get("label", 0)) == 1]
    if len(positives) < k:
        raise ValueError(f"need {k} positive support clips, got {len(positives)}")
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in positives:
        by_prompt[str(clip.get("prompt_group"))].append(clip)
    for group in by_prompt.values():
        group.sort(key=lambda clip: support_sort_key(seed, clip))
    chosen: list[dict[str, Any]] = []
    for prompt in ("P1_vigil_only", "P2_phrase_plus_vigil", "P3_vigil_plus_phrase"):
        if by_prompt.get(prompt) and len(chosen) < k:
            chosen.append(by_prompt[prompt].pop(0))
    remaining = sorted([clip for group in by_prompt.values() for clip in group], key=lambda clip: support_sort_key(seed, clip))
    chosen.extend(remaining[: max(0, k - len(chosen))])
    return sorted(chosen[:k], key=lambda clip: str(clip["clip_id"]))


def support_query_split(clips: list[dict[str, Any]], support: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    support_ids = {str(clip["clip_id"]) for clip in support}
    query = [clip for clip in clips if str(clip["clip_id"]) not in support_ids]
    support_hashes = {str(clip.get("full_wav_sha256") or clip.get("audio_sha256")) for clip in support}
    query_hashes = {str(clip.get("full_wav_sha256") or clip.get("audio_sha256")) for clip in query}
    if support_ids & {str(clip["clip_id"]) for clip in query}:
        raise ValueError("support/query clip overlap")
    if support_hashes & query_hashes:
        raise ValueError("support/query duplicate audio hash overlap")
    return support, query
