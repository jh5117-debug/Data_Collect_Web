from __future__ import annotations

import torch


def trainable_parameter_names(module: torch.nn.Module) -> list[str]:
    return [name for name, param in module.named_parameters() if param.requires_grad]


def assert_no_frozen_backbone_parameters(optimizer: torch.optim.Optimizer, frozen_params: set[int]) -> None:
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) in frozen_params:
                raise ValueError("optimizer contains frozen backbone parameter")
