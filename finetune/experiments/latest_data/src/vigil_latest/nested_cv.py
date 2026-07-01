from __future__ import annotations


def development_folds_for_outer(outer_fold: int, fold_count: int = 5) -> list[int]:
    return [fold for fold in range(fold_count) if fold != outer_fold]


def validation_fold_for_outer(outer_fold: int, fold_count: int = 5) -> int:
    return development_folds_for_outer(outer_fold, fold_count)[0]
