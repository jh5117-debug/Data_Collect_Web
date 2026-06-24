from __future__ import annotations

import torch

from model_loader import _infer_stage2_dims


def test_infer_stage2_dims_from_state_dict():
    state = {
        "norm.weight": torch.zeros(2048),
        "proj.weight": torch.zeros(256, 2048),
        "embed.weight": torch.zeros(128, 256),
    }
    assert _infer_stage2_dims(state) == (2048, 256, 128)

