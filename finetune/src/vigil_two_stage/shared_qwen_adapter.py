from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Callable


@dataclass
class SharedQwenCounters:
    model_load_count: int = 0
    transcribe_call_count: int = 0
    generate_call_count: int = 0
    thinker_forward_call_count: int = 0
    get_audio_features_call_count: int = 0
    decoder_call_count: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def explicit_encoder_call_count(self) -> int:
        return int(self.thinker_forward_call_count) + int(self.get_audio_features_call_count)


class MethodCallCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self._patches: list[tuple[Any, str, Any]] = []

    def patch(self, obj: Any, method_name: str, counter_name: str | None = None) -> bool:
        if obj is None or not hasattr(obj, method_name):
            return False
        original = getattr(obj, method_name)
        name = counter_name or method_name
        self.counts.setdefault(name, 0)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.counts[name] += 1
            return original(*args, **kwargs)

        try:
            setattr(obj, method_name, wrapped)
        except Exception:
            try:
                setattr(obj, method_name, MethodType(wrapped, obj))
            except Exception:
                return False
        self._patches.append((obj, method_name, original))
        return True

    def restore(self) -> None:
        for obj, method_name, original in reversed(self._patches):
            try:
                setattr(obj, method_name, original)
            except Exception:
                pass
        self._patches.clear()


def can_claim_verified_one_encoder_forward(status: str, encoder_call_count: int) -> bool:
    return status == "verified_one_encoder_forward" and int(encoder_call_count) == 1


def blocker_text(public_transcribe_signature: str, feature_path: str) -> str:
    return (
        "The installed qwen_asr public transcribe API accepts raw audio and returns ASRTranscription text, "
        "while the Stage 2 feature path calls "
        f"{feature_path}. The public API does not expose reusable audio encoder hidden states and does not "
        f"accept externally supplied hidden states for decoding. Signature inspected: {public_transcribe_signature}"
    )
