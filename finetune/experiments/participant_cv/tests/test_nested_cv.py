from __future__ import annotations

import pytest

from vigil_participant_cv.nested_cv import assert_outer_test_not_in_development, inner_validation_plan


def test_inner_oof_validation_never_contains_inner_training_participants():
    plan = inner_validation_plan([1, 2, 3, 4])
    for item in plan:
        assert item["inner_validation_fold"] not in item["inner_train_folds"]


def test_every_outer_test_fold_is_untouched():
    assert_outer_test_not_in_development([1, 2, 3, 4], 0)
    with pytest.raises(ValueError):
        assert_outer_test_not_in_development([0, 1, 2, 3], 0)
