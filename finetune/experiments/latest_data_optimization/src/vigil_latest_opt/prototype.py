from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("cannot normalize empty or non-finite vector")
    return arr / norm


def build_prototype(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("support embeddings are required")
    arr = np.stack([l2_normalize(vec) for vec in embeddings])
    return l2_normalize(arr.mean(axis=0))


def cosine_similarity(embedding: np.ndarray, prototype: np.ndarray) -> float:
    return float(np.dot(l2_normalize(embedding), l2_normalize(prototype)))
