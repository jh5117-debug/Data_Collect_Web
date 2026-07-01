from __future__ import annotations


def weighted_score(base: float, personalization: float, weight: float) -> float:
    weight = max(0.0, min(1.0, float(weight)))
    return (1.0 - weight) * float(base) + weight * float(personalization)
