from __future__ import annotations

import random
from typing import Any


def choose_positive_support(rows: list[dict[str, Any]], *, shots: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives = [row for row in rows if int(row["label"]) == 1]
    if len(positives) < shots + 1:
        raise ValueError(f"participant needs at least {shots + 1} positive clips for {shots}-shot support plus query")
    rng = random.Random(seed)
    ranked = sorted(positives, key=lambda row: (str(row.get("prompt_group")), str(row["clip_id"])))
    support = rng.sample(ranked, shots)
    support_ids = {str(row["clip_id"]) for row in support}
    query = [row for row in rows if str(row["clip_id"]) not in support_ids]
    return sorted(support, key=lambda row: str(row["clip_id"])), query


def choose_nested_supports(rows: list[dict[str, Any]], *, seed: int) -> dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    positives = [row for row in rows if int(row["label"]) == 1]
    if len(positives) < 6:
        raise ValueError("participant is not 5-shot eligible")
    rng = random.Random(seed)
    ranked = sorted(positives, key=lambda row: (str(row.get("prompt_group")), str(row["clip_id"])))
    support5 = rng.sample(ranked, 5)
    support3 = support5[:3]
    out = {}
    for shots, support in ((3, support3), (5, support5)):
        support_ids = {str(row["clip_id"]) for row in support}
        out[shots] = (
            sorted(support, key=lambda row: str(row["clip_id"])),
            [row for row in rows if str(row["clip_id"]) not in support_ids],
        )
    return out


def assert_no_target_negatives(rows: list[dict[str, Any]]) -> None:
    if any(int(row["label"]) == 0 for row in rows):
        raise ValueError("target negatives are not allowed in strict adaptation")
