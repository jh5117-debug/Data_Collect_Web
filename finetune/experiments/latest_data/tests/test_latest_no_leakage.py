from __future__ import annotations

import pytest

from vigil_latest.protocol import ProtocolError, validate_no_duplicate_hash_crosses_folds


def test_duplicate_audio_hash_cross_fold_is_rejected() -> None:
    folds = {
        "folds": [
            {"fold": 0, "participant_aliases": ["P001"]},
            {"fold": 1, "participant_aliases": ["P002"]},
        ]
    }
    clips = [
        {"participant_alias": "P001", "full_wav_sha256": "same"},
        {"participant_alias": "P002", "full_wav_sha256": "same"},
    ]
    with pytest.raises(ProtocolError):
        validate_no_duplicate_hash_crosses_folds(clips, folds)
