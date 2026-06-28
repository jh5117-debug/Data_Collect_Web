from __future__ import annotations

from trigger import TriggerState, bounded_positive_bias


def test_trigger_state_and_cooldown() -> None:
    trigger = TriggerState(cooldown_seconds=5.0)
    first = trigger.update(candidate=True, trigger_detected=True, now=10.0)
    assert first["trigger_accepted"] is True
    assert trigger.state == "ASSISTANT_STATE"
    second = trigger.update(candidate=True, trigger_detected=True, now=12.0)
    assert second["cooldown_active"] is True
    assert second["trigger_accepted"] is False


def test_bounded_positive_bias() -> None:
    bias = bounded_positive_bias([0.2, 0.3, 0.4], 0.9, max_bias=0.5)
    assert 0.0 <= bias <= 0.5
