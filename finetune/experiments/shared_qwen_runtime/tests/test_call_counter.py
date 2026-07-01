from __future__ import annotations

from call_counter import MethodCallCounter, RuntimeCallCounts, can_claim_verified_one_encoder_forward, require_blocker_text


class FakeModel:
    def __init__(self) -> None:
        self.value = 0

    def generate(self) -> str:
        self.value += 1
        return "ok"

    def get_audio_features(self) -> list[int]:
        self.value += 10
        return [1, 2, 3]


def test_call_counter_counts_and_restores_fake_model() -> None:
    model = FakeModel()
    original = model.generate
    counter = MethodCallCounter()
    assert counter.patch(model, "generate", "generate")
    assert model.generate() == "ok"
    assert model.generate() == "ok"
    assert counter.snapshot().generate_call_count == 2
    counter.restore()
    assert model.generate.__func__ is original.__func__


def test_verified_one_forward_requires_status_and_one_call() -> None:
    assert can_claim_verified_one_encoder_forward("verified_one_encoder_forward", RuntimeCallCounts(get_audio_features_call_count=1))
    assert not can_claim_verified_one_encoder_forward("verified_one_encoder_forward", RuntimeCallCounts(get_audio_features_call_count=2))
    assert not can_claim_verified_one_encoder_forward("blocked_by_runtime_interface", RuntimeCallCounts(get_audio_features_call_count=1))


def test_blocked_status_requires_specific_blocker_text() -> None:
    good = "wrapper does not expose decoder hidden states and does not accept supplied hidden states"
    bad = "it failed"
    assert require_blocker_text("blocked_by_runtime_interface", good)
    assert not require_blocker_text("blocked_by_runtime_interface", bad)
    assert require_blocker_text("verified_one_encoder_forward", "")
