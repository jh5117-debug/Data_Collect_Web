from __future__ import annotations

import pytest
import torch

from vigil_participant_cv.adaptation import assert_no_frozen_backbone_parameters


def test_adaptation_optimizer_contains_no_frozen_backbone_parameters():
    frozen = torch.nn.Linear(2, 2)
    head = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    assert_no_frozen_backbone_parameters(opt, {id(p) for p in frozen.parameters()})
    bad = torch.optim.SGD(list(head.parameters()) + list(frozen.parameters()), lr=0.1)
    with pytest.raises(ValueError):
        assert_no_frozen_backbone_parameters(bad, {id(p) for p in frozen.parameters()})


def test_qwen_remains_frozen_placeholder_contract():
    qwen = torch.nn.Linear(2, 2)
    for param in qwen.parameters():
        param.requires_grad = False
    assert sum(p.requires_grad for p in qwen.parameters()) == 0
