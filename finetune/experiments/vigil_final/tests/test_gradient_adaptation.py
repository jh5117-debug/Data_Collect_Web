from __future__ import annotations

import torch
from torch import nn

from vigil_final.gradient_adaptation import GradientRecipe, assert_replay_development_only, build_optimizer, freeze_for_target


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_backbone = nn.Linear(2, 2)
        self.embed = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 1)


def test_gradient_optimizer_excludes_frozen_backbones():
    model = Tiny()
    trainable = freeze_for_target(model, "stage2_classifier")
    assert all("qwen" not in name for name in trainable)
    assert trainable == ["classifier.bias", "classifier.weight"]
    opt = build_optimizer(model, GradientRecipe("stage2_classifier", 1e-5, 5, 1e-3))
    assert opt.param_groups[0]["params"]


def test_source_replay_contains_development_only():
    assert_replay_development_only([{"participant_alias": "P1"}], {"P1"})
    try:
        assert_replay_development_only([{"participant_alias": "P2"}], {"P1"})
    except ValueError as exc:
        assert "non-development" in str(exc)
    else:
        raise AssertionError("expected ValueError")
