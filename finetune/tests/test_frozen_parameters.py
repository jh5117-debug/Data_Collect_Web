import torch

from vigil_two_stage.qwen_audio_adapter import DummyFrozenEncoder
from vigil_two_stage.stage2_model import QwenVerifierHead


def test_dummy_frozen_encoder_parameters_are_excluded_from_optimizer():
    enc = DummyFrozenEncoder()
    head = QwenVerifierHead(8)
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    opt_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert all(not p.requires_grad for p in enc.parameters())
    assert all(id(p) not in opt_ids for p in enc.parameters())
