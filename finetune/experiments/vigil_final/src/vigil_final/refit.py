from __future__ import annotations


def median_epoch_policy(best_epochs: list[int], *, minimum: int = 1, maximum: int = 20) -> int:
    if not best_epochs:
        return minimum
    ordered = sorted(int(epoch) for epoch in best_epochs)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        value = ordered[mid]
    else:
        value = round((ordered[mid - 1] + ordered[mid]) / 2)
    return max(minimum, min(maximum, int(value)))
