from __future__ import annotations

from vigil_latest.onboarding import fpr_safety_gate, support_query_disjoint


def test_support_query_disjointness_and_fpr_gate() -> None:
    assert support_query_disjoint({"a"}, {"b"})
    assert not support_query_disjoint({"a"}, {"a", "b"})
    assert fpr_safety_gate(0.01, 0.025)
    assert not fpr_safety_gate(0.01, 0.04)
