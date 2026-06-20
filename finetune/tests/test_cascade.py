from vigil_two_stage.cascade import cascade_decision


def test_cascade_decision_requires_both_thresholds():
    assert cascade_decision(0.9, 0.5, 0.9, 0.5)
    assert not cascade_decision(0.4, 0.5, 0.9, 0.5)
    assert not cascade_decision(0.9, 0.5, 0.4, 0.5)
    assert not cascade_decision(0.9, 0.5, None, None)
