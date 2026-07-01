from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from call_counter import MethodCallCounter
from vigil_two_stage.qwen_audio_adapter import FrozenQwenAudioAdapter
from vigil_two_stage.qwen_text_result import extract_qwen_text


BLOCKER = (
    "The current public qwen_asr wrapper does not expose decoder-compatible audio hidden states "
    "and does not accept externally supplied audio hidden states for decoding."
)


@dataclass
class SharedAttempt:
    status: str
    transcript: str | None
    raw_transcribe_result_type: str | None
    audio_features_available: bool
    feature_shape: list[int] | None
    feature_dtype: str | None
    encoder_call_count: int
    decoder_call_count: int
    feature_path: str | None
    transcript_path: str | None
    latency_ms: float
    blocker: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "transcript": self.transcript,
            "raw_transcribe_result_type": self.raw_transcribe_result_type,
            "audio_features_available": self.audio_features_available,
            "feature_shape": self.feature_shape,
            "feature_dtype": self.feature_dtype,
            "encoder_call_count": self.encoder_call_count,
            "decoder_call_count": self.decoder_call_count,
            "feature_path": self.feature_path,
            "transcript_path": self.transcript_path,
            "latency_ms": self.latency_ms,
            "blocker": self.blocker,
        }


class SharedQwenASRRuntime:
    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-ASR-1.7B", dtype: torch.dtype | None = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.wrapper: Any | None = None
        self.model_load_count = 0

    def load(self) -> None:
        if self.wrapper is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Qwen3-ASR runtime diagnostics.")
        from qwen_asr import Qwen3ASRModel

        dtype = self.dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.wrapper = Qwen3ASRModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            max_inference_batch_size=1,
            max_new_tokens=128,
        )
        self.model_load_count += 1
        self.wrapper.model.eval()
        for param in self.wrapper.model.parameters():
            param.requires_grad = False

    def _patch_counter(self) -> MethodCallCounter:
        if self.wrapper is None:
            raise RuntimeError("Qwen runtime is not loaded.")
        counter = MethodCallCounter()
        counter.patch(self.wrapper, "transcribe", "transcribe")
        counter.patch(self.wrapper.model, "generate", "generate")
        thinker = getattr(self.wrapper.model, "thinker", None)
        counter.patch(thinker, "forward", "thinker_forward")
        counter.patch(thinker, "get_audio_features", "get_audio_features")
        return counter

    def public_transcribe(self, audio: str) -> dict[str, Any]:
        self.load()
        counter = self._patch_counter()
        try:
            start = time.perf_counter()
            raw = self.wrapper.transcribe(audio, language=None, return_time_stamps=False)
            latency_ms = (time.perf_counter() - start) * 1000.0
            extracted = extract_qwen_text(raw)
            counts = counter.snapshot(self.model_load_count).as_dict()
        finally:
            counter.restore()
        return {
            "status": "ok",
            "transcript": extracted.text,
            "result_type": extracted.result_type,
            "extraction_path": extracted.extraction_path,
            "latency_ms": latency_ms,
            "counts": counts,
        }

    def separate_stage2_features(self, audio: str) -> dict[str, Any]:
        self.load()
        counter = self._patch_counter()
        try:
            adapter = FrozenQwenAudioAdapter(self.model_name_or_path)
            adapter._set_loaded_model(self.wrapper)
            start = time.perf_counter()
            hidden = adapter.extract_audio_features(audio)
            latency_ms = (time.perf_counter() - start) * 1000.0
            counts = counter.snapshot(self.model_load_count).as_dict()
        finally:
            counter.restore()
        return {
            "status": "ok",
            "feature_shape": list(hidden.shape),
            "feature_dtype": str(hidden.dtype),
            "feature_path": adapter.extraction_path,
            "latency_ms": latency_ms,
            "counts": counts,
        }

    def transcribe_and_get_features(self, audio: str) -> SharedAttempt:
        self.load()
        counter = self._patch_counter()
        start = time.perf_counter()
        try:
            raw = self.wrapper.transcribe(audio, language=None, return_time_stamps=False)
            extracted = extract_qwen_text(raw)
            counts = counter.snapshot(self.model_load_count)
        finally:
            counter.restore()
        latency_ms = (time.perf_counter() - start) * 1000.0
        return SharedAttempt(
            status="blocked_by_runtime_interface",
            transcript=extracted.text,
            raw_transcribe_result_type=extracted.result_type,
            audio_features_available=False,
            feature_shape=None,
            feature_dtype=None,
            encoder_call_count=counts.encoder_call_count,
            decoder_call_count=counts.generate_call_count,
            feature_path=None,
            transcript_path=extracted.extraction_path,
            latency_ms=latency_ms,
            blocker=BLOCKER,
        )
