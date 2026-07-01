from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from vigil_two_stage.strict_runtime import (
    StrictRuntimeError,
    fallback_feature_namespace,
    official_openwakeword_namespace,
    optimizer_contains_any_parameters,
    qwen_trainable_parameter_count,
    require_strict_official_config,
    validate_one_visible_cuda_device,
)


def strict_config() -> dict:
    return {
        "runtime": {
            "require_cuda": True,
            "require_local_gpu": True,
            "allow_slurm": False,
            "max_gpus": 1,
        },
        "stage1": {
            "require_official_openwakeword": True,
            "allow_acoustic_fallback_when_openwakeword_missing": False,
            "allow_fallback_features": False,
        },
        "stage2": {
            "model_name": "Qwen/Qwen3-ASR-1.7B",
            "require_qwen_1_7b": True,
            "allow_skip_without_gpu": False,
            "allow_dummy_encoder": False,
            "allow_cpu_qwen": False,
        },
    }


class FakeCuda:
    def __init__(self, available: bool, count: int, name: str = "NVIDIA GeForce RTX 3090"):
        self._available = available
        self._count = count
        self._name = name

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self._name


class FakeTorch:
    def __init__(self, cuda: FakeCuda):
        self.cuda = cuda


def test_strict_config_accepts_official_settings():
    require_strict_official_config(strict_config())


def test_strict_config_rejects_fallback_features():
    config = deepcopy(strict_config())
    config["stage1"]["allow_fallback_features"] = True
    with pytest.raises(StrictRuntimeError):
        require_strict_official_config(config)


def test_strict_config_rejects_slurm_or_multiple_gpus():
    config = deepcopy(strict_config())
    config["runtime"]["allow_slurm"] = True
    with pytest.raises(StrictRuntimeError):
        require_strict_official_config(config)

    config = deepcopy(strict_config())
    config["runtime"]["max_gpus"] = 2
    with pytest.raises(StrictRuntimeError):
        require_strict_official_config(config)


def test_validate_one_visible_cuda_device_rejects_missing_cuda():
    with pytest.raises(StrictRuntimeError):
        validate_one_visible_cuda_device(FakeTorch(FakeCuda(False, 0)))


def test_validate_one_visible_cuda_device_requires_exactly_one_gpu():
    with pytest.raises(StrictRuntimeError):
        validate_one_visible_cuda_device(FakeTorch(FakeCuda(True, 2)))


def test_validate_one_visible_cuda_device_requires_rtx_3090():
    with pytest.raises(StrictRuntimeError):
        validate_one_visible_cuda_device(FakeTorch(FakeCuda(True, 1, "NVIDIA A100")))

    assert validate_one_visible_cuda_device(FakeTorch(FakeCuda(True, 1))) == "NVIDIA GeForce RTX 3090"


def test_qwen_trainable_parameter_count_zero_when_frozen():
    model = torch.nn.Linear(4, 2)
    for param in model.parameters():
        param.requires_grad = False
    assert qwen_trainable_parameter_count(model) == 0


def test_optimizer_contains_no_qwen_parameters():
    qwen = torch.nn.Linear(4, 4)
    verifier = torch.nn.Linear(4, 1)
    for param in qwen.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=0.001)
    assert not optimizer_contains_any_parameters(optimizer, list(qwen.parameters()))


def test_official_and_fallback_cache_namespaces_are_distinct():
    pre = "preprocessing-v1"
    official = official_openwakeword_namespace("0.6.0", "abc123", pre)
    fallback = fallback_feature_namespace(pre)
    assert "official_openwakeword" in official
    assert "fallback_fft" in fallback
    assert official != fallback
