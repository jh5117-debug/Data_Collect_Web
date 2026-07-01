from __future__ import annotations

import numpy as np
import pytest

from vigil_final.prototype import PrototypeError, PrototypeRecipe, apply_recipe, build_prototype, cosine_similarity, validate_support_rows
from vigil_final.support import assert_no_target_negatives, choose_nested_supports, choose_positive_support


def test_prototype_is_normalized_and_uses_support_only():
    proto = build_prototype([np.array([1.0, 0.0]), np.array([1.0, 1.0])])
    assert np.isclose(np.linalg.norm(proto), 1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), proto) > 0.7


def test_support_query_overlap_zero_and_positive_only():
    support = [{"clip_id": f"S{i}", "label": 1} for i in range(3)]
    query = [{"clip_id": "Q", "label": 0}]
    validate_support_rows(support, query, shots=3)
    with pytest.raises(PrototypeError):
        validate_support_rows(support, support, shots=3)
    with pytest.raises(PrototypeError):
        validate_support_rows([{"clip_id": "N", "label": 0}], query, shots=1)


def test_choose_support_has_exact_shots_and_removes_from_query():
    rows = [{"clip_id": f"P{i}", "label": 1, "prompt_group": "P1"} for i in range(6)] + [{"clip_id": "N", "label": 0}]
    support, query = choose_positive_support(rows, shots=3, seed=7)
    assert len(support) == 3
    assert all(row["label"] == 1 for row in support)
    assert {r["clip_id"] for r in support}.isdisjoint({r["clip_id"] for r in query})


def test_three_shot_is_subset_of_five_when_possible():
    rows = [{"clip_id": f"P{i}", "label": 1, "prompt_group": "P1"} for i in range(6)] + [{"clip_id": "N", "label": 0}]
    chosen = choose_nested_supports(rows, seed=11)
    assert len(chosen[3][0]) == 3
    assert len(chosen[5][0]) == 5
    assert {r["clip_id"] for r in chosen[3][0]} <= {r["clip_id"] for r in chosen[5][0]}


def test_target_negatives_absent_from_adaptation():
    assert_no_target_negatives([{"label": 1}])
    with pytest.raises(ValueError):
        assert_no_target_negatives([{"label": 0}])


def test_prototype_fusion_does_not_need_query_label():
    score, decision = apply_recipe(0.2, 0.8, PrototypeRecipe(method="base_plus_prototype", alpha=0.5, beta=0.0, threshold=0.5))
    assert score == 0.6000000000000001
    assert decision is True
