from __future__ import annotations


def base_plus_prototype(base_logit: float, cosine_similarity: float, *, alpha: float, beta: float) -> float:
    return float(base_logit + alpha * (cosine_similarity - beta))


def decision_from_score(score: float, threshold: float) -> bool:
    return bool(float(score) >= float(threshold))
