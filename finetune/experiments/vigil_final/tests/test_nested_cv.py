from __future__ import annotations

import pytest

from vigil_final.nested_cv import Context, LeakageError, RoleSeparatedDataset, build_outer_plan, validate_inner_coverage
from vigil_final.refit import median_epoch_policy


def folds():
    return {"folds": [{"fold": i, "participant_aliases": [f"P{i}{j}"]} for i in range(5) for j in range(2)]}


def compact_folds():
    return {"folds": [{"fold": i, "participant_aliases": [f"P{i}"]} for i in range(5)]}


def rows():
    return [{"participant_alias": f"P{i}", "clip_id": f"C{i}", "label": i % 2} for i in range(5)]


def test_four_inner_validation_folds_cover_development_once():
    f = compact_folds()
    plan = build_outer_plan(2)
    info = validate_inner_coverage(plan, f)
    assert info["development_participants"] == 4
    assert len(plan.inner_folds) == 4
    assert {inner.inner_validation_fold for inner in plan.inner_folds} == {0, 1, 3, 4}


def test_outer_test_rows_cannot_be_read_during_inner_selection():
    ds = RoleSeparatedDataset(rows(), compact_folds(), outer_fold=2)
    with pytest.raises(LeakageError):
        ds.read_outer_test(Context.INNER_SELECTION)


def test_inner_training_and_validation_roles_are_disjoint():
    ds = RoleSeparatedDataset(rows(), compact_folds(), outer_fold=2)
    inner = build_outer_plan(2).inner_folds[0]
    train = ds.read_inner_train(inner, Context.INNER_SELECTION)
    val = ds.read_inner_validation(inner, Context.INNER_SELECTION)
    assert {r["participant_alias"] for r in train}.isdisjoint({r["participant_alias"] for r in val})
    assert "P2" not in {r["participant_alias"] for r in train + val}


def test_refit_uses_development_and_no_outer_test():
    ds = RoleSeparatedDataset(rows(), compact_folds(), outer_fold=2)
    refit = ds.read_refit_train(Context.REFIT)
    assert {r["participant_alias"] for r in refit} == {"P0", "P1", "P3", "P4"}


def test_final_evaluation_is_first_outer_test_access():
    ds = RoleSeparatedDataset(rows(), compact_folds(), outer_fold=2)
    inner = build_outer_plan(2).inner_folds[0]
    ds.read_inner_train(inner, Context.INNER_SELECTION)
    ds.read_inner_validation(inner, Context.INNER_SELECTION)
    assert all(record["role"] != "outer_test" for record in ds.access_log.records)
    test = ds.read_outer_test(Context.FINAL_EVALUATION)
    assert {r["participant_alias"] for r in test} == {"P2"}


def test_final_epoch_policy_is_deterministic():
    assert median_epoch_policy([3, 9, 5, 5]) == 5
    assert median_epoch_policy([]) == 1
    assert median_epoch_policy([100], maximum=20) == 20
