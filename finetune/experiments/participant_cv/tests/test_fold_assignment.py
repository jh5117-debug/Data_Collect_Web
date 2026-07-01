from __future__ import annotations

from vigil_participant_cv.folds import build_folds, fold_objective
from vigil_participant_cv.protocol import validate_unique_fold_membership
from vigil_participant_cv.utils import stable_json_dumps


def _clips():
    rows = []
    for p in range(10):
        for i in range(4):
            rows.append({"participant_alias": f"P{p+1:03d}", "clip_id": f"C{p:02d}_{i}", "label": i % 2, "prompt_group": ["P1_vigil_only", "P2_phrase_plus_vigil", "P3_vigil_plus_phrase", "P4_negative"][i]})
    return rows


def test_fold_assignment_is_byte_deterministic():
    assert stable_json_dumps(build_folds(_clips(), starts=5)) == stable_json_dumps(build_folds(_clips(), starts=5))


def test_every_participant_is_in_exactly_one_fold():
    folds = build_folds(_clips(), starts=5)
    validate_unique_fold_membership(folds)
    assert sum(len(f["participant_aliases"]) for f in folds["folds"]) == 10


def test_fold_balance_objective_is_finite():
    folds = build_folds(_clips(), starts=5)
    assert folds["objective"] >= 0
    assert fold_objective([fold["stats"] for fold in folds["folds"]], folds["objective_weights"]) >= 0
