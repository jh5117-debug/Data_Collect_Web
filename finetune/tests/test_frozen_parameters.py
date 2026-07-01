import torch

from vigil_two_stage.qwen_audio_adapter import DummyFrozenEncoder, FrozenQwenAudioAdapter
from vigil_two_stage.stage2_model import QwenVerifierHead


def test_dummy_frozen_encoder_parameters_are_excluded_from_optimizer():
    enc = DummyFrozenEncoder()
    head = QwenVerifierHead(8)
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    opt_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert all(not p.requires_grad for p in enc.parameters())
    assert all(id(p) not in opt_ids for p in enc.parameters())


def test_qwen_wrapper_is_unwrapped_for_frozen_integrity():
    class Wrapper:
        def __init__(self):
            self.model = torch.nn.Linear(2, 3)
            self.processor = object()

    wrapper = Wrapper()
    adapter = FrozenQwenAudioAdapter("dummy")
    adapter._set_loaded_model(wrapper)
    integrity = adapter.integrity()

    assert adapter.wrapper is wrapper
    assert adapter.model is wrapper.model
    assert adapter.processor is wrapper.processor
    assert integrity.total_parameters == sum(p.numel() for p in wrapper.model.parameters())
    assert integrity.trainable_parameters == 0
    assert all(not p.requires_grad for p in wrapper.model.parameters())


def test_verifier_optimizer_contains_no_qwen_parameters():
    qwen = torch.nn.Linear(4, 4)
    for param in qwen.parameters():
        param.requires_grad = False
    head = QwenVerifierHead(4)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    opt_ids = {id(p) for group in opt.param_groups for p in group["params"]}

    assert all(id(p) not in opt_ids for p in qwen.parameters())
