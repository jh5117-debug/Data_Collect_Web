from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class QwenAdapterUnavailable(RuntimeError):
    pass


@dataclass
class FrozenIntegrity:
    total_parameters: int
    trainable_parameters: int
    representative_checksums: dict[str, float]


def checksum_representative_parameters(model: torch.nn.Module, limit: int = 8) -> FrozenIntegrity:
    total = 0
    trainable = 0
    checksums: dict[str, float] = {}
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
        if len(checksums) < limit:
            checksums[name] = float(param.detach().float().sum().cpu().item())
    return FrozenIntegrity(total, trainable, checksums)


class FrozenQwenAudioAdapter:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Any | None = None
        self.processor: Any | None = None

    def load(self) -> None:
        if not torch.cuda.is_available():
            raise QwenAdapterUnavailable("CUDA is not available; Qwen3-ASR-1.7B encoder extraction was not run.")
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as exc:
            raise QwenAdapterUnavailable(f"transformers import failed: {exc}") from exc
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="cuda:0",
            )
        except Exception as exc:
            raise QwenAdapterUnavailable(f"Qwen model load failed: {exc}") from exc
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def integrity(self) -> FrozenIntegrity:
        if self.model is None:
            raise QwenAdapterUnavailable("Qwen model is not loaded.")
        return checksum_representative_parameters(self.model)

    def extract_audio_features(self, wav_path: str) -> torch.Tensor:
        raise QwenAdapterUnavailable(
            "Qwen3-ASR audio-encoder adapter requires source inspection for this installed version; no transcript decoder features are used."
        )


class DummyFrozenEncoder(torch.nn.Module):
    def __init__(self, input_dim: int = 4, output_dim: int = 8):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.linear(x)
