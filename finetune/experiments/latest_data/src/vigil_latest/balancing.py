from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any


def prompt_order(prompt_group: str) -> int:
    return {"P1_vigil_only": 1, "P2_phrase_plus_vigil": 2, "P3_vigil_plus_phrase": 3, "P4_negative": 4}.get(prompt_group, 99)


def stable_sample_key(seed: int, clip: dict[str, Any]) -> str:
    text = f"{seed}:{clip.get('participant_alias')}:{clip.get('prompt_group')}:{clip.get('phrase_id')}:{clip.get('clip_id')}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def allocation_by_largest_remainder(counts: dict[str, int], cap: int) -> dict[str, int]:
    total = sum(counts.values())
    if total <= cap:
        return dict(counts)
    raw = {key: cap * value / total for key, value in counts.items()}
    alloc = {key: min(counts[key], int(math.floor(value))) for key, value in raw.items()}
    for key, value in counts.items():
        if value > 0 and alloc[key] == 0 and sum(alloc.values()) < cap:
            alloc[key] = 1
    remaining = cap - sum(alloc.values())
    order = sorted(counts, key=lambda key: (raw[key] - math.floor(raw[key]), counts[key], key), reverse=True)
    while remaining > 0:
        progressed = False
        for key in order:
            if remaining <= 0:
                break
            if alloc[key] < counts[key]:
                alloc[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def balance_max_clips_per_participant(clips: list[dict[str, Any]], *, max_clips: int = 100, seed: int = 20260620) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_alias: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        by_alias[str(clip["participant_alias"])].append(clip)
    selected: list[dict[str, Any]] = []
    participant_summary: list[dict[str, Any]] = []
    for alias, items in sorted(by_alias.items()):
        items = sorted(items, key=lambda c: (prompt_order(str(c.get("prompt_group"))), str(c.get("phrase_id")), str(c.get("clip_id"))))
        if len(items) <= max_clips:
            chosen = items
        else:
            strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for item in items:
                strata[(str(item.get("prompt_group")), str(item.get("phrase_id")))].append(item)
            counts = {f"{key[0]}||{key[1]}": len(value) for key, value in strata.items()}
            alloc = allocation_by_largest_remainder(counts, max_clips)
            chosen = []
            for key, group in strata.items():
                label = f"{key[0]}||{key[1]}"
                ordered = sorted(group, key=lambda c: stable_sample_key(seed, c))
                chosen.extend(ordered[: alloc[label]])
            chosen = sorted(chosen, key=lambda c: (prompt_order(str(c.get("prompt_group"))), str(c.get("phrase_id")), str(c.get("clip_id"))))
        selected.extend(chosen)
        before_prompt = Counter(str(c.get("prompt_group")) for c in items)
        after_prompt = Counter(str(c.get("prompt_group")) for c in chosen)
        participant_summary.append(
            {
                "participant_alias": alias,
                "clips_before": len(items),
                "clips_after": len(chosen),
                "clips_removed": len(items) - len(chosen),
                "prompt_counts_before": dict(sorted(before_prompt.items())),
                "prompt_counts_after": dict(sorted(after_prompt.items())),
                "positive_after": sum(int(c.get("label", 0)) == 1 for c in chosen),
                "negative_after": sum(int(c.get("label", 0)) == 0 for c in chosen),
            }
        )
    summary = {
        "max_clips_per_participant": max_clips,
        "seed": seed,
        "participants": len(by_alias),
        "clips_before": len(clips),
        "clips_after": len(selected),
        "participant_summary": participant_summary,
        "prompt_counts_before": dict(sorted(Counter(str(c.get("prompt_group")) for c in clips).items())),
        "prompt_counts_after": dict(sorted(Counter(str(c.get("prompt_group")) for c in selected).items())),
        "label_counts_after": dict(sorted(Counter(str(int(c.get("label", 0))) for c in selected).items())),
    }
    return sorted(selected, key=lambda c: (str(c["participant_alias"]), str(c["clip_id"]))), summary
