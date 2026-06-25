from __future__ import annotations

from vigil_final.metrics import paired_delta
from vigil_final.safety import fpr_safety_gate, select_safe_recipe


def test_fpr_safety_gate_and_fallback():
    safe = fpr_safety_gate({"false_positive_rate": 0.0}, {"false_positive_rate": 0.01})
    unsafe = fpr_safety_gate({"false_positive_rate": 0.0}, {"false_positive_rate": 0.05})
    assert safe["passed"] is True
    assert unsafe["passed"] is False
    selected = select_safe_recipe([{"safety": unsafe}])
    assert selected["selected_recipe"] == "no_adaptation_zero_shot_fallback"


def test_paired_query_delta_uses_same_rows():
    rows = [
        {"participant_alias": "P1", "label": 1, "zero": True, "adapted": True},
        {"participant_alias": "P1", "label": 0, "zero": True, "adapted": False},
    ]
    out = paired_delta(rows, "zero", "adapted")
    assert out["n"] == 1
    assert out["improved"] == 1
