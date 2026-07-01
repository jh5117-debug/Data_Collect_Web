from __future__ import annotations

from typing import Iterable


def choose_threshold_for_recall(labels: Iterable[int], scores: Iterable[float], recall_target: float) -> float:
    pairs = sorted(zip(scores, labels), reverse=True)
    positives = sum(1 for _, label in pairs if int(label) == 1)
    if positives == 0:
        return 1.0
    seen_pos = 0
    for score, label in pairs:
        if int(label) == 1:
            seen_pos += 1
        if seen_pos / positives >= recall_target:
            return float(score)
    return float(pairs[-1][0]) if pairs else 1.0
