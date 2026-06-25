from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class PrototypeError(ValueError):
    pass


@dataclass(frozen=True)
class PrototypeRecipe:
    method: str
    alpha: float = 0.0
    beta: float = 0.0
    threshold: float = 0.5
    top_k: int = 3


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0:
        raise PrototypeError("cannot normalize empty or non-finite vector")
    return vec / norm


def build_prototype(support_embeddings: list[np.ndarray]) -> np.ndarray:
    if not support_embeddings:
        raise PrototypeError("support embeddings are required")
    arr = np.stack([l2_normalize(np.asarray(vec, dtype=np.float32)) for vec in support_embeddings])
    return l2_normalize(arr.mean(axis=0))


def cosine_similarity(embedding: np.ndarray, prototype: np.ndarray) -> float:
    return float(np.dot(l2_normalize(np.asarray(embedding, dtype=np.float32)), l2_normalize(np.asarray(prototype, dtype=np.float32))))


def fused_logit(base_logit: float, similarity: float, alpha: float, beta: float) -> float:
    return float(base_logit + alpha * (similarity - beta))


def validate_support_rows(support: list[dict[str, Any]], query: list[dict[str, Any]], *, shots: int) -> None:
    if len(support) != shots:
        raise PrototypeError(f"expected {shots} support clips, got {len(support)}")
    if any(int(row["label"]) != 1 for row in support):
        raise PrototypeError("prototype support must be positive-only")
    support_ids = {str(row["clip_id"]) for row in support}
    query_ids = {str(row["clip_id"]) for row in query}
    overlap = support_ids & query_ids
    if overlap:
        raise PrototypeError(f"support/query overlap: {sorted(overlap)}")


def apply_recipe(base_logit: float, similarity: float, recipe: PrototypeRecipe) -> tuple[float, bool]:
    if recipe.method == "prototype_only":
        score = similarity
    elif recipe.method == "base_plus_prototype":
        score = fused_logit(base_logit, similarity, recipe.alpha, recipe.beta)
    else:
        raise PrototypeError(f"unsupported prototype method: {recipe.method}")
    return float(score), bool(score >= recipe.threshold)
