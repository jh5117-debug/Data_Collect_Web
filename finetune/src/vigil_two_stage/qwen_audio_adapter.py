from __future__ import annotations

from dataclasses import dataclass
import importlib.util
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
        self.wrapper: Any | None = None
        self.model: torch.nn.Module | None = None
        self.processor: Any | None = None
        self.extraction_path: str | None = None
        self.dtype: torch.dtype | None = None

    def load(self) -> None:
        if not torch.cuda.is_available():
            raise QwenAdapterUnavailable("CUDA is not available; Qwen3-ASR-1.7B encoder extraction was not run.")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if importlib.util.find_spec("qwen_asr") is not None:
            try:
                from qwen_asr import Qwen3ASRModel  # type: ignore
            except Exception as exc:
                raise QwenAdapterUnavailable(f"qwen_asr import failed: {exc}") from exc
            try:
                loaded = Qwen3ASRModel.from_pretrained(self.model_name, torch_dtype=self.dtype)
            except Exception as exc:
                raise QwenAdapterUnavailable(f"Qwen ASR model load failed: {exc}") from exc
            self._set_loaded_model(loaded)
            return
        try:
            from transformers import AutoModel, AutoProcessor
        except Exception as exc:
            raise QwenAdapterUnavailable(f"transformers import failed: {exc}") from exc
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            loaded = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            )
        except Exception as exc:
            raise QwenAdapterUnavailable(f"Qwen model load failed: {exc}") from exc
        self._set_loaded_model(loaded, processor=self.processor)

    def _set_loaded_model(self, loaded: Any, processor: Any | None = None) -> None:
        self.wrapper = loaded if hasattr(loaded, "model") and not hasattr(loaded, "named_parameters") else None
        self.model = getattr(loaded, "model", loaded)
        self.processor = processor or getattr(loaded, "processor", self.processor)
        if not hasattr(self.model, "named_parameters"):
            raise QwenAdapterUnavailable("Loaded Qwen object does not expose a torch model for frozen integrity checks.")
        if torch.cuda.is_available() and hasattr(self.model, "to"):
            self.model.to("cuda:0")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def integrity(self) -> FrozenIntegrity:
        if self.model is None:
            raise QwenAdapterUnavailable("Qwen model is not loaded.")
        if not hasattr(self.model, "named_parameters"):
            raise QwenAdapterUnavailable("Loaded Qwen wrapper does not expose named_parameters; frozen integrity cannot be proven.")
        return checksum_representative_parameters(self.model)

    def extract_audio_features(self, wav_path: str) -> torch.Tensor:
        if self.model is None:
            raise QwenAdapterUnavailable("Qwen model is not loaded.")
        errors = []
        with torch.inference_mode():
            try:
                return self._extract_qwen3_asr_thinker_features(wav_path)
            except Exception as exc:  # pragma: no cover - depends on installed Qwen runtime
                errors.append(f"model.thinker.get_audio_features:{type(exc).__name__}:{exc}")
        attempts: list[tuple[str, Any]] = []
        for name in ("extract_audio_features", "get_audio_features", "encode_audio"):
            if hasattr(self.model, name):
                method = getattr(self.model, name)
                attempts.append((name, lambda method=method: method(wav_path)))
                attempts.append((f"{name}_list", lambda method=method: method([wav_path])))
        nested = getattr(self.model, "model", None)
        if nested is not None:
            for name in ("extract_audio_features", "get_audio_features", "encode_audio"):
                if hasattr(nested, name):
                    method = getattr(nested, name)
                    attempts.append((f"model.{name}", lambda method=method: method(wav_path)))
                    attempts.append((f"model.{name}_list", lambda method=method: method([wav_path])))
        with torch.inference_mode():
            for name, attempt in attempts:
                try:
                    output = attempt()
                    tensor = self._coerce_hidden_states(output)
                    self.extraction_path = name
                    return tensor
                except Exception as exc:  # pragma: no cover - depends on installed Qwen runtime
                    errors.append(f"{name}:{type(exc).__name__}:{exc}")
        raise QwenAdapterUnavailable(
            "No supported Qwen3-ASR audio-encoder hidden-state method was found. "
            "Transcript decoder outputs were not used. Attempts: "
            + " | ".join(errors)
        )

    def _extract_qwen3_asr_thinker_features(self, wav_path: str) -> torch.Tensor:
        if self.model is None or self.processor is None:
            raise QwenAdapterUnavailable("Qwen model or processor is not loaded.")
        thinker = getattr(self.model, "thinker", None)
        get_audio_features = getattr(thinker, "get_audio_features", None)
        if get_audio_features is None:
            raise QwenAdapterUnavailable("Loaded Qwen model does not expose thinker.get_audio_features.")
        try:
            from qwen_asr.inference.utils import normalize_audio_input  # type: ignore
        except Exception as exc:
            raise QwenAdapterUnavailable(f"Could not import qwen_asr audio normalization utility: {exc}") from exc
        audio = normalize_audio_input(wav_path)
        audio_token = getattr(self.processor, "audio_token", "<|AUDIO|>")
        inputs = self.processor(text=[audio_token], audio=[audio], return_tensors="pt", padding=True)
        if "input_features" not in inputs:
            raise QwenAdapterUnavailable("Qwen processor did not return input_features for audio encoder extraction.")
        device = next(self.model.parameters()).device
        model_dtype = self.dtype or next(self.model.parameters()).dtype
        input_features = inputs["input_features"].to(device=device, dtype=model_dtype)
        feature_attention_mask = inputs.get("feature_attention_mask")
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(device=device)
        output = get_audio_features(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        self.extraction_path = "model.thinker.get_audio_features"
        return self._coerce_hidden_states(output)

    @staticmethod
    def _coerce_hidden_states(output: Any) -> torch.Tensor:
        if isinstance(output, dict):
            for key in ("hidden_states", "audio_features", "features", "last_hidden_state"):
                if key in output:
                    return FrozenQwenAudioAdapter._coerce_hidden_states(output[key])
        if isinstance(output, (list, tuple)):
            if not output:
                raise QwenAdapterUnavailable("Qwen audio feature output was empty")
            return FrozenQwenAudioAdapter._coerce_hidden_states(output[0])
        if not isinstance(output, torch.Tensor):
            output = torch.as_tensor(output)
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise QwenAdapterUnavailable(f"Qwen audio features must have shape [T, D], got {tuple(output.shape)}")
        output = output.detach()
        if not torch.isfinite(output.float()).all():
            raise QwenAdapterUnavailable("Qwen audio features contain NaN or Inf")
        return output


class DummyFrozenEncoder(torch.nn.Module):
    def __init__(self, input_dim: int = 4, output_dim: int = 8):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.linear(x)
