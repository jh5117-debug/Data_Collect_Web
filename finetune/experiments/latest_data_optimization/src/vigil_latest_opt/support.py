from __future__ import annotations

import random
from typing import Any


def choose_positive_support(rows: list[dict[str, Any]], *, shots: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives = [row for row in rows if int(row["label"]) == 1]
    negatives = [row for row in rows if int(row["label"]) == 0]
    if len(positives) < shots + 1 or not negatives:
        raise ValueError(f"participant is not eligible for {shots}-shot support plus positive/negative query")
    rng = random.Random(seed)
    ranked = sorted(positives, key=lambda row: (str(row.get("prompt_group")), str(row["clip_id"])))
    support = rng.sample(ranked, shots)
    support_ids = {str(row["clip_id"]) for row in support}
    query = [row for row in rows if str(row["clip_id"]) not in support_ids]
    if any(int(row["label"]) == 0 for row in support):
        raise ValueError("target negatives are not allowed in support")
    if support_ids & {str(row["clip_id"]) for row in query}:
        raise ValueError("support/query overlap")
    return sorted(support, key=lambda row: str(row["clip_id"])), query


def choose_nested_positive_supports(rows: list[dict[str, Any]], *, seed: int) -> dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    return {
        3: choose_positive_support(rows, shots=3, seed=seed),
        5: choose_positive_support(rows, shots=5, seed=seed),
    }
