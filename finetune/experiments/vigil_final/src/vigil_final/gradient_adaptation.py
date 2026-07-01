from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


FROZEN_BACKBONE_KEYWORDS = ("qwen", "audio_encoder", "openwakeword", "backbone", "thinker", "lm", "decoder")


@dataclass(frozen=True)
class GradientRecipe:
    target: str
    learning_rate: float
    steps: int
    l2_to_base: float


def trainable_parameter_names(model: torch.nn.Module, target: str) -> list[str]:
    names = []
    for name, _param in model.named_parameters():
        lowered = name.lower()
        if any(token in lowered for token in FROZEN_BACKBONE_KEYWORDS):
            continue
        if target == "stage2_classifier_bias":
            if "classifier.bias" in lowered:
                names.append(name)
        elif target == "stage2_classifier":
            if "classifier" in lowered:
                names.append(name)
        elif target == "stage2_embedding_and_classifier":
            if "embed" in lowered or "classifier" in lowered:
                names.append(name)
        elif target == "stage1_classifier":
            if "classifier" in lowered:
                names.append(name)
        elif target == "both_final_classifiers":
            if "classifier" in lowered:
                names.append(name)
        else:
            raise ValueError(f"unsupported gradient target: {target}")
    return names


def freeze_for_target(model: torch.nn.Module, target: str) -> list[str]:
    allowed = set(trainable_parameter_names(model, target))
    for name, param in model.named_parameters():
        param.requires_grad = name in allowed
    return sorted(allowed)


def build_optimizer(model: torch.nn.Module, recipe: GradientRecipe) -> torch.optim.Optimizer:
    freeze_for_target(model, recipe.target)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("no trainable parameters selected")
    return torch.optim.AdamW(params, lr=recipe.learning_rate)


def assert_replay_development_only(rows: Iterable[dict], development_aliases: set[str]) -> None:
    bad = sorted({str(row["participant_alias"]) for row in rows if str(row["participant_alias"]) not in development_aliases})
    if bad:
        raise ValueError(f"source replay contains non-development participants: {bad}")
