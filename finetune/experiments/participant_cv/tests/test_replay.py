from __future__ import annotations

import pytest

from vigil_participant_cv.replay import validate_source_replay


def test_source_replay_contains_only_development_participants():
    validate_source_replay([{"participant_alias": "P001"}], {"P001"}, "P002")
    with pytest.raises(ValueError):
        validate_source_replay([{"participant_alias": "P002"}], {"P001"}, "P002")
