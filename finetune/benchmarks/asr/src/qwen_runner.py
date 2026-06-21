from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import model_info


class QwenRunnerError(RuntimeError):
    pass


def _extract_text(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "transcript", "prediction", "output", "hypothesis"):
            if key in result:
                return _extract_text(result[key])
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        return _extract_text(result[0])
    return str(result)


def _model_parameter_module(obj: Any) -> torch.nn.Module | None:
    if hasattr(obj, "named_parameters"):
        return obj
    nested = getattr(obj, "model", None)
    if nested is not None and hasattr(nested, "named_parameters"):
        return nested
    return None


@dataclass
class TranscriptionResult:
    hypothesis: str
    latency_sec: float
    peak_gpu_memory_gb: float | None


class QwenASRRunner:
    def __init__(
        self,
        model_name: str,
        *,
        dtype: str = "auto",
        language: str = "auto",
        max_new_tokens: int = 1024,
        backend: str = "transformers",
        require_baseline_model: bool = False,
    ):
        self.model_name = model_name
        self.dtype_arg = dtype
        self.language = None if language in ("auto", "none", "None", "") else language
        self.max_new_tokens = max_new_tokens
        self.backend = backend
        self.require_baseline_model = require_baseline_model
        self.model: Any | None = None
        self.torch_module: torch.nn.Module | None = None
        self.dtype: torch.dtype | None = None
        self.load_time_sec: float | None = None
        self.model_revision: str | None = None

    def load(self) -> None:
        if self.require_baseline_model and self.model_name != "Qwen/Qwen3-ASR-1.7B":
            raise QwenRunnerError(f"baseline run requires Qwen/Qwen3-ASR-1.7B, got {self.model_name}")
        if not torch.cuda.is_available():
            raise QwenRunnerError("CUDA is required; CPU Qwen inference is forbidden for this benchmark")
        if torch.cuda.device_count() != 1:
            raise QwenRunnerError(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")
        self.dtype = self._resolve_dtype()
        if importlib.util.find_spec("qwen_asr") is None:
            raise QwenRunnerError("qwen_asr package is not importable")
        from qwen_asr import Qwen3ASRModel  # type: ignore

        self.model_revision = self._resolve_model_revision()
        started = time.perf_counter()
        try:
            self.model = Qwen3ASRModel.from_pretrained(self.model_name, torch_dtype=self.dtype)
        except TypeError:
            self.model = Qwen3ASRModel.from_pretrained(self.model_name)
        self.load_time_sec = time.perf_counter() - started
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.torch_module = _model_parameter_module(self.model)
        if self.torch_module is not None:
            if hasattr(self.torch_module, "to"):
                self.torch_module.to("cuda:0")
            self.torch_module.eval()
            for param in self.torch_module.parameters():
                param.requires_grad = False
            trainable = sum(param.numel() for param in self.torch_module.parameters() if param.requires_grad)
            if trainable:
                raise QwenRunnerError(f"Qwen must remain frozen, but {trainable} parameters are trainable")

    def _resolve_dtype(self) -> torch.dtype:
        if self.dtype_arg == "auto":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if self.dtype_arg in ("bf16", "bfloat16"):
            return torch.bfloat16
        if self.dtype_arg in ("fp16", "float16"):
            return torch.float16
        if self.dtype_arg in ("fp32", "float32"):
            return torch.float32
        raise QwenRunnerError(f"unsupported dtype: {self.dtype_arg}")

    def _resolve_model_revision(self) -> str | None:
        if Path(self.model_name).exists():
            return None
        try:
            info = model_info(self.model_name)
        except Exception:
            return None
        sha = getattr(info, "sha", None)
        return str(sha) if sha else None

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if self.model is None:
            raise QwenRunnerError("Qwen runner is not loaded")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        attempts = []
        if hasattr(self.model, "transcribe"):
            attempts.extend(
                [
                    lambda: self.model.transcribe(str(audio_path), language=self.language),
                    lambda: self.model.transcribe(str(audio_path)),
                    lambda: self.model.transcribe([str(audio_path)]),
                ]
            )
        if hasattr(self.model, "generate"):
            attempts.extend(
                [
                    lambda: self.model.generate(str(audio_path), max_new_tokens=self.max_new_tokens, do_sample=False),
                    lambda: self.model.generate(str(audio_path)),
                ]
            )
        if callable(self.model):
            attempts.append(lambda: self.model(str(audio_path)))
        errors: list[str] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for attempt in attempts:
                try:
                    hypothesis = _extract_text(attempt()).strip()
                    latency = time.perf_counter() - started
                    peak = float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None
                    return TranscriptionResult(hypothesis=hypothesis, latency_sec=latency, peak_gpu_memory_gb=peak)
                except TypeError as exc:
                    errors.append(f"TypeError:{exc}")
                    continue
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}:{exc}")
                    continue
        raise QwenRunnerError("Qwen transcription failed: " + " | ".join(errors))

    def model_info(self) -> dict[str, Any]:
        total = trainable = None
        if self.torch_module is not None:
            total = sum(param.numel() for param in self.torch_module.parameters())
            trainable = sum(param.numel() for param in self.torch_module.parameters() if param.requires_grad)
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "backend": self.backend,
            "dtype": str(self.dtype),
            "language": self.language,
            "max_new_tokens": self.max_new_tokens,
            "require_baseline_model": self.require_baseline_model,
            "load_time_sec": self.load_time_sec,
            "total_parameters": total,
            "trainable_parameters": trainable,
        }
