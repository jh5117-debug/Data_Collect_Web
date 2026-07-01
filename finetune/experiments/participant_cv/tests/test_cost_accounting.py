from __future__ import annotations

from vigil_participant_cv.cost import CostAccounting


def test_cost_accounting_distinguishes_model_copies_from_forward_passes():
    cost = CostAccounting(qwen_copies=1, qwen_full_transcript_forwards=1, extra_qwen_encoder_forwards=1, stage1_forwards=10)
    assert cost.as_dict()["qwen_copies"] == 1
    assert cost.as_dict()["extra_qwen_encoder_forwards"] == 1
