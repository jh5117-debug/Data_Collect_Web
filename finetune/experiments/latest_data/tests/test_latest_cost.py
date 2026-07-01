from __future__ import annotations

import pytest

from vigil_latest.cost import candidate_rate, false_accepts_per_hour


def test_cost_helpers() -> None:
    assert false_accepts_per_hour(2, 3600.0) == 2.0
    assert candidate_rate(5, 100) == 0.05
    with pytest.raises(ValueError):
        false_accepts_per_hour(1, 0.0)
