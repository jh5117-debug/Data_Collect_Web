from __future__ import annotations


def inner_validation_plan(development_folds: list[int]) -> list[dict[str, list[int] | int]]:
    plan = []
    for heldout in development_folds:
        train = [fold for fold in development_folds if fold != heldout]
        plan.append({"inner_validation_fold": heldout, "inner_train_folds": train})
    return plan


def assert_outer_test_not_in_development(development_folds: list[int], outer_test_fold: int) -> None:
    if outer_test_fold in development_folds:
        raise ValueError("outer test fold appears in development folds")
