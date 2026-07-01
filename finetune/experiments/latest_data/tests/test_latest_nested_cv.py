from __future__ import annotations

from vigil_latest.nested_cv import development_folds_for_outer, validation_fold_for_outer


def test_nested_cv_development_excludes_outer_test_fold() -> None:
    dev = development_folds_for_outer(2, 5)
    assert 2 not in dev
    assert validation_fold_for_outer(2, 5) in dev
