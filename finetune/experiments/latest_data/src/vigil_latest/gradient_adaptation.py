from __future__ import annotations


def adaptation_changes_score(base_score: float, adapted_score: float) -> bool:
    return abs(float(base_score) - float(adapted_score)) > 1e-12
