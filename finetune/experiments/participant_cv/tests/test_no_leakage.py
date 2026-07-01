from __future__ import annotations

import pytest

from vigil_participant_cv.protocol import ProtocolError, validate_no_duplicate_hash_crosses_folds, validate_outer_split


def test_no_participant_crosses_outer_roles():
    validate_outer_split({"P001"}, {"P002"}, {"P003"})
    with pytest.raises(ProtocolError):
        validate_outer_split({"P001"}, set(), {"P001"})


def test_no_duplicate_audio_hash_crosses_folds():
    clips = [
        {"participant_alias": "P001", "clip_id": "C1", "full_wav_sha256": "h"},
        {"participant_alias": "P002", "clip_id": "C2", "full_wav_sha256": "h"},
    ]
    folds = {"folds": [{"fold": 0, "participant_aliases": ["P001"]}, {"fold": 1, "participant_aliases": ["P002"]}]}
    with pytest.raises(ProtocolError):
        validate_no_duplicate_hash_crosses_folds(clips, folds)
