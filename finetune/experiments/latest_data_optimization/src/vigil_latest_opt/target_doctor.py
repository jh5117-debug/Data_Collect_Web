from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .metrics import binary_metrics


SUPPORT_SEEDS = [20260620, 20260621, 20260622, 20260623, 20260624]


def clip_key(row: dict[str, Any]) -> str:
    return str(row["clip_id"])


def group_by_clip(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clip_key(row)].append(row)
    return dict(grouped)


def one_row_per_clip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sorted(group, key=lambda row: int(row.get("window_index", 0)))[0] for _, group in sorted(group_by_clip(rows).items())]


def target_clip_rows(rows: list[dict[str, Any]], alias: str) -> list[dict[str, Any]]:
    return [row for row in one_row_per_clip(rows) if str(row.get("participant_alias")) == str(alias)]


def eligibility(rows: list[dict[str, Any]], shots: int) -> dict[str, Any]:
    clips = one_row_per_clip(rows)
    positives = sum(int(row["label"]) == 1 for row in clips)
    negatives = sum(int(row["label"]) == 0 for row in clips)
    return {
        "eligible": positives >= shots + 1 and negatives >= 1,
        "stable_fpr": negatives >= 5,
        "positive_clips": positives,
        "negative_clips": negatives,
        "required_positive_clips": shots + 1,
    }


def _prompt_cover_order(positives: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        by_prompt[str(row.get("prompt_group", ""))].append(row)
    for group in by_prompt.values():
        group.sort(key=lambda row: clip_key(row))
        rng.shuffle(group)
    ordered = []
    for prompt in ("P1_vigil_only", "P2_phrase_plus_vigil", "P3_vigil_plus_phrase"):
        if by_prompt.get(prompt):
            ordered.append(by_prompt[prompt].pop(0))
    rest = [row for group in by_prompt.values() for row in group]
    rest.sort(key=lambda row: clip_key(row))
    rng.shuffle(rest)
    ordered.extend(rest)
    seen = set()
    unique = []
    for row in ordered:
        key = clip_key(row)
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique


def choose_supports_for_seed(rows: list[dict[str, Any]], seed: int) -> dict[int, list[dict[str, Any]]]:
    clips = one_row_per_clip(rows)
    positives = [row for row in clips if int(row["label"]) == 1]
    negatives = [row for row in clips if int(row["label"]) == 0]
    if len(positives) < 4 or not negatives:
        raise ValueError("target participant is not eligible for 3-shot target-doctor onboarding")
    ordered = _prompt_cover_order(positives, random.Random(seed))
    supports: dict[int, list[dict[str, Any]]] = {3: ordered[:3]}
    if len(positives) >= 6:
        supports[5] = ordered[:5]
    return supports


def split_support_query(rows: list[dict[str, Any]], support: list[dict[str, Any]], alias: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = [row for row in one_row_per_clip(rows) if str(row.get("participant_alias")) == str(alias)]
    support_ids = {clip_key(row) for row in support}
    if len(support_ids) != len(support):
        raise ValueError("support contains duplicate clip IDs")
    if any(str(row.get("participant_alias")) != str(alias) for row in support):
        raise ValueError("support must contain only the target doctor")
    if any(int(row["label"]) != 1 for row in support):
        raise ValueError("target negatives are not allowed in support")
    target_ids = {clip_key(row) for row in target}
    if not support_ids <= target_ids:
        raise ValueError("support contains clips outside target doctor")
    query = [row for row in target if clip_key(row) not in support_ids]
    query_ids = {clip_key(row) for row in query}
    if support_ids & query_ids:
        raise ValueError("support/query overlap")
    if any(str(row.get("participant_alias")) != str(alias) for row in query):
        raise ValueError("query contains non-target doctor clips")
    return list(support), query


def assert_paired_query(base_query: list[dict[str, Any]], adapted_query: list[dict[str, Any]]) -> None:
    base_ids = {clip_key(row) for row in base_query}
    adapted_ids = {clip_key(row) for row in adapted_query}
    if base_ids != adapted_ids:
        raise ValueError("zero-shot and few-shot query sets are not paired")


def metrics_for_decision(rows: list[dict[str, Any]], decision_key: str) -> dict[str, Any]:
    return binary_metrics([int(row["label"]) for row in rows], [bool(row[decision_key]) for row in rows])


def improvement_counts(per_doctor: list[dict[str, Any]], shot_key: str, eps: float = 1e-12) -> dict[str, int]:
    improved = degraded = unchanged = 0
    for row in per_doctor:
        delta = row.get(f"{shot_key}_delta_f1")
        if delta is None:
            continue
        if float(delta) > eps:
            improved += 1
        elif float(delta) < -eps:
            degraded += 1
        else:
            unchanged += 1
    return {"improved": improved, "degraded": degraded, "unchanged": unchanged}
