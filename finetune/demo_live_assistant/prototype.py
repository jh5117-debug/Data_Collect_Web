from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class PrototypeCalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class PrototypeCalibration:
    embedding: np.ndarray
    support_count: int
    pairwise_mean_similarity: float | None
    pairwise_min_similarity: float | None


def l2_normalize(vector: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise PrototypeCalibrationError(f"expected 1D embedding, got shape {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise PrototypeCalibrationError("cannot normalize an empty or non-finite embedding")
    return arr / norm


def cosine_similarity(embedding: Iterable[float] | np.ndarray, prototype: Iterable[float] | np.ndarray) -> float:
    left = l2_normalize(embedding)
    right = l2_normalize(prototype)
    if left.shape != right.shape:
        raise PrototypeCalibrationError(f"embedding shape mismatch: {left.shape} vs {right.shape}")
    return float(np.dot(left, right))


def build_prototype(support_embeddings: list[Iterable[float] | np.ndarray]) -> PrototypeCalibration:
    if not support_embeddings:
        raise PrototypeCalibrationError("support embeddings are required")
    normalized = [l2_normalize(embedding) for embedding in support_embeddings]
    dims = {tuple(embedding.shape) for embedding in normalized}
    if len(dims) != 1:
        raise PrototypeCalibrationError(f"support embedding dimensions do not match: {sorted(dims)}")
    prototype = l2_normalize(np.stack(normalized, axis=0).mean(axis=0))
    pairwise = [
        cosine_similarity(left, right)
        for index, left in enumerate(normalized)
        for right in normalized[index + 1 :]
    ]
    return PrototypeCalibration(
        embedding=prototype,
        support_count=len(normalized),
        pairwise_mean_similarity=float(np.mean(pairwise)) if pairwise else None,
        pairwise_min_similarity=float(np.min(pairwise)) if pairwise else None,
    )
