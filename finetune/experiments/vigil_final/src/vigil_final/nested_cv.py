from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .utils import alias_to_fold, fold_to_aliases


class Role(str, Enum):
    INNER_TRAIN = "inner_train"
    INNER_VALIDATION = "inner_validation"
    REFIT_TRAIN = "refit_train"
    OUTER_TEST = "outer_test"


class Context(str, Enum):
    INNER_SELECTION = "inner_selection"
    REFIT = "refit"
    FINAL_EVALUATION = "final_evaluation"


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class InnerFoldPlan:
    outer_fold: int
    inner_validation_fold: int
    inner_train_folds: tuple[int, ...]


@dataclass(frozen=True)
class OuterFoldPlan:
    outer_fold: int
    development_folds: tuple[int, ...]
    inner_folds: tuple[InnerFoldPlan, ...]


@dataclass
class AccessLog:
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, context: Context, role: Role, rows: list[dict[str, Any]]) -> None:
        self.records.append(
            {
                "context": context.value,
                "role": role.value,
                "rows": len(rows),
                "participants": sorted({str(row["participant_alias"]) for row in rows}),
            }
        )


def build_outer_plan(outer_fold: int, n_folds: int = 5) -> OuterFoldPlan:
    development = tuple(fold for fold in range(n_folds) if fold != outer_fold)
    inner = tuple(
        InnerFoldPlan(
            outer_fold=outer_fold,
            inner_validation_fold=heldout,
            inner_train_folds=tuple(fold for fold in development if fold != heldout),
        )
        for heldout in development
    )
    return OuterFoldPlan(outer_fold=outer_fold, development_folds=development, inner_folds=inner)


def build_all_outer_plans(n_folds: int = 5) -> list[OuterFoldPlan]:
    return [build_outer_plan(fold, n_folds=n_folds) for fold in range(n_folds)]


def assert_disjoint(*sets: set[str]) -> None:
    for i, left in enumerate(sets):
        for j, right in enumerate(sets):
            if i >= j:
                continue
            overlap = left & right
            if overlap:
                raise LeakageError(f"participant role overlap: {sorted(overlap)}")


class RoleSeparatedDataset:
    def __init__(self, rows: list[dict[str, Any]], folds: dict[str, Any], outer_fold: int):
        self.rows = list(rows)
        self.folds = folds
        self.outer_fold = int(outer_fold)
        self.alias_to_fold = alias_to_fold(folds)
        self.fold_aliases = fold_to_aliases(folds)
        self.access_log = AccessLog()

    def _rows_for_folds(self, folds: set[int]) -> list[dict[str, Any]]:
        return [row for row in self.rows if self.alias_to_fold[str(row["participant_alias"])] in folds]

    def read_inner_train(self, plan: InnerFoldPlan, context: Context) -> list[dict[str, Any]]:
        if context != Context.INNER_SELECTION:
            raise LeakageError("inner train rows may only be read in inner-selection context")
        if plan.outer_fold != self.outer_fold:
            raise LeakageError("inner plan outer fold mismatch")
        folds = set(plan.inner_train_folds)
        if self.outer_fold in folds or plan.inner_validation_fold in folds:
            raise LeakageError("inner train folds include held-out validation or outer test")
        rows = self._rows_for_folds(folds)
        self.access_log.record(context, Role.INNER_TRAIN, rows)
        return rows

    def read_inner_validation(self, plan: InnerFoldPlan, context: Context) -> list[dict[str, Any]]:
        if context != Context.INNER_SELECTION:
            raise LeakageError("inner validation rows may only be read in inner-selection context")
        if plan.outer_fold != self.outer_fold:
            raise LeakageError("inner plan outer fold mismatch")
        if plan.inner_validation_fold == self.outer_fold:
            raise LeakageError("outer test fold used as inner validation")
        rows = self._rows_for_folds({plan.inner_validation_fold})
        self.access_log.record(context, Role.INNER_VALIDATION, rows)
        return rows

    def read_refit_train(self, context: Context) -> list[dict[str, Any]]:
        if context != Context.REFIT:
            raise LeakageError("refit train rows may only be read in refit context")
        rows = self._rows_for_folds(set(range(len(self.fold_aliases))) - {self.outer_fold})
        self.access_log.record(context, Role.REFIT_TRAIN, rows)
        return rows

    def read_outer_test(self, context: Context) -> list[dict[str, Any]]:
        if context != Context.FINAL_EVALUATION:
            raise LeakageError("outer-test rows are only available in final-evaluation context")
        rows = self._rows_for_folds({self.outer_fold})
        self.access_log.record(context, Role.OUTER_TEST, rows)
        return rows


def validate_inner_coverage(plan: OuterFoldPlan, folds: dict[str, Any]) -> dict[str, Any]:
    aliases = fold_to_aliases(folds)
    development_aliases = set().union(*(aliases[fold] for fold in plan.development_folds))
    seen_validation: set[str] = set()
    for inner in plan.inner_folds:
        train_aliases = set().union(*(aliases[fold] for fold in inner.inner_train_folds))
        validation_aliases = set(aliases[inner.inner_validation_fold])
        outer_aliases = set(aliases[plan.outer_fold])
        assert_disjoint(train_aliases, validation_aliases, outer_aliases)
        if seen_validation & validation_aliases:
            raise LeakageError("inner validation participants repeated")
        seen_validation |= validation_aliases
    if seen_validation != development_aliases:
        raise LeakageError("inner validation folds do not cover development participants exactly once")
    return {
        "outer_fold": plan.outer_fold,
        "development_participants": len(development_aliases),
        "outer_test_participants": len(aliases[plan.outer_fold]),
        "inner_runs": len(plan.inner_folds),
    }
