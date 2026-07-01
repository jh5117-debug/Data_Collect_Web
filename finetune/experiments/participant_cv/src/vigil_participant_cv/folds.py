from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any


def participant_vectors(clips: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_alias: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        by_alias[str(clip["participant_alias"])].append(clip)
    vectors: dict[str, dict[str, int]] = {}
    for alias, items in by_alias.items():
        counts = Counter(str(c.get("prompt_group")) for c in items)
        vectors[alias] = {
            "participants": 1,
            "clips": len(items),
            "positive": sum(int(c.get("label", 0)) == 1 for c in items),
            "negative": sum(int(c.get("label", 0)) == 0 for c in items),
            "P1_vigil_only": counts["P1_vigil_only"],
            "P2_phrase_plus_vigil": counts["P2_phrase_plus_vigil"],
            "P3_vigil_plus_phrase": counts["P3_vigil_plus_phrase"],
            "P4_negative": counts["P4_negative"],
        }
    return vectors


def fold_objective(fold_stats: list[dict[str, int]], weights: dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        values = [float(fold.get(key, 0)) for fold in fold_stats]
        mean = sum(values) / len(values)
        if mean == 0:
            continue
        score += float(weight) * sum(((value - mean) / mean) ** 2 for value in values)
    return float(score)


def build_folds(
    clips: list[dict[str, Any]],
    *,
    fold_count: int = 5,
    seed: int = 20260620,
    starts: int = 200,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    weights = weights or {
        "participants": 5.0,
        "clips": 3.0,
        "positive": 2.0,
        "negative": 2.0,
        "P1_vigil_only": 1.0,
        "P2_phrase_plus_vigil": 1.0,
        "P3_vigil_plus_phrase": 1.0,
        "P4_negative": 1.0,
    }
    vectors = participant_vectors(clips)
    aliases = sorted(vectors)
    best: tuple[float, list[list[str]], list[dict[str, int]]] | None = None
    for start in range(starts):
        rng = random.Random(seed + start)
        order = sorted(aliases, key=lambda a: (vectors[a]["clips"], vectors[a]["positive"], rng.random()), reverse=True)
        folds: list[list[str]] = [[] for _ in range(fold_count)]
        stats: list[dict[str, int]] = [defaultdict(int) for _ in range(fold_count)]  # type: ignore[list-item]
        for alias in order:
            candidate_scores = []
            for i in range(fold_count):
                tmp = [dict(s) for s in stats]
                for key, value in vectors[alias].items():
                    tmp[i][key] = tmp[i].get(key, 0) + value
                candidate_scores.append((fold_objective(tmp, weights), len(folds[i]), i))
            _, _, chosen = min(candidate_scores)
            folds[chosen].append(alias)
            for key, value in vectors[alias].items():
                stats[chosen][key] += value
        normalized_folds = [sorted(fold) for fold in folds]
        normalized_stats = [dict(s) for s in stats]
        objective = fold_objective(normalized_stats, weights)
        if best is None or objective < best[0]:
            best = (objective, normalized_folds, normalized_stats)
    assert best is not None
    return {
        "fold_count": fold_count,
        "seed": seed,
        "algorithm": "deterministic_multistart_greedy_v1",
        "objective": best[0],
        "objective_weights": weights,
        "folds": [
            {"fold": i, "participant_aliases": fold, "stats": best[2][i]}
            for i, fold in enumerate(best[1])
        ],
    }
