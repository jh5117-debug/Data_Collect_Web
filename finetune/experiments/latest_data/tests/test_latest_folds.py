from __future__ import annotations

from vigil_latest.folds import build_folds
from vigil_latest.protocol import validate_unique_fold_membership


def test_latest_folds_have_unique_participant_membership() -> None:
    clips = [
        {"participant_alias": f"P{i:03d}", "clip_id": f"c{i}", "label": i % 2, "prompt_group": "P1_vigil_only", "full_wav_sha256": f"h{i}"}
        for i in range(10)
    ]
    folds = build_folds(clips, fold_count=5, starts=5)
    validate_unique_fold_membership(folds)
    assert sum(len(fold["participant_aliases"]) for fold in folds["folds"]) == 10
