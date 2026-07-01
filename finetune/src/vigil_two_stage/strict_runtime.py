from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import yaml


class StrictRuntimeError(RuntimeError):
    pass


def require_strict_official_config(config: dict[str, Any]) -> None:
    runtime = config.get("runtime", {})
    stage1 = config.get("stage1", {})
    stage2 = config.get("stage2", {})

    if not runtime.get("require_cuda", False):
        raise StrictRuntimeError("runtime.require_cuda must be true for the official GPU smoke run")
    if not runtime.get("require_local_gpu", False):
        raise StrictRuntimeError("runtime.require_local_gpu must be true for the official GPU smoke run")
    if runtime.get("allow_slurm", True):
        raise StrictRuntimeError("runtime.allow_slurm must be false for the local RTX 3090 smoke run")
    if int(runtime.get("max_gpus", 0)) != 1:
        raise StrictRuntimeError("runtime.max_gpus must be exactly 1")

    if not stage1.get("require_official_openwakeword", False):
        raise StrictRuntimeError("stage1.require_official_openwakeword must be true")
    if stage1.get("allow_acoustic_fallback_when_openwakeword_missing", True):
        raise StrictRuntimeError("stage1 acoustic fallback must be disabled")
    if stage1.get("allow_fallback_features", True):
        raise StrictRuntimeError("stage1.allow_fallback_features must be false")

    if stage2.get("model_name") != "Qwen/Qwen3-ASR-1.7B":
        raise StrictRuntimeError("stage2.model_name must be Qwen/Qwen3-ASR-1.7B")
    if not stage2.get("require_qwen_1_7b", False):
        raise StrictRuntimeError("stage2.require_qwen_1_7b must be true")
    if stage2.get("allow_skip_without_gpu", True):
        raise StrictRuntimeError("stage2.allow_skip_without_gpu must be false")
    if stage2.get("allow_dummy_encoder", True):
        raise StrictRuntimeError("stage2.allow_dummy_encoder must be false")
    if stage2.get("allow_cpu_qwen", True):
        raise StrictRuntimeError("stage2.allow_cpu_qwen must be false")


def require_package(module_name: str, label: str | None = None) -> None:
    if importlib.util.find_spec(module_name) is None:
        display = label or module_name
        raise StrictRuntimeError(f"required package is not importable: {display}")


def official_openwakeword_namespace(version: str, model_checksum: str, preprocessing_fingerprint: str) -> str:
    return f"official_openwakeword/v={version}/model={model_checksum}/pre={preprocessing_fingerprint}"


def fallback_feature_namespace(preprocessing_fingerprint: str) -> str:
    return f"fallback_fft/pre={preprocessing_fingerprint}"


def validate_one_visible_cuda_device(torch_module: Any, required_name_fragment: str = "RTX 3090") -> str:
    if not torch_module.cuda.is_available():
        raise StrictRuntimeError("CUDA is not available")
    count = int(torch_module.cuda.device_count())
    if count != 1:
        raise StrictRuntimeError(f"expected exactly one visible CUDA device, got {count}")
    name = str(torch_module.cuda.get_device_name(0))
    if required_name_fragment not in name:
        raise StrictRuntimeError(f"visible CUDA device is not an {required_name_fragment}: {name}")
    return name


def qwen_trainable_parameter_count(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def optimizer_contains_any_parameters(optimizer: Any, params: list[Any]) -> bool:
    target_ids = {id(param) for param in params}
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if id(param) in target_ids:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-openwakeword", action="store_true")
    parser.add_argument("--require-qwen-asr", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    require_strict_official_config(config)
    if args.require_openwakeword:
        require_package("openwakeword", "official openWakeWord")
    if args.require_qwen_asr:
        require_package("qwen_asr", "official Qwen3-ASR runtime package")
    print("strict official runtime checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
