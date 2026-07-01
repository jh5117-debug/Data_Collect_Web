from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeCallCounts:
    model_load_count: int = 0
    transcribe_call_count: int = 0
    generate_call_count: int = 0
    get_audio_features_call_count: int = 0
    thinker_forward_call_count: int = 0
    decoder_call_count: int = 0
    patched_methods: list[str] = field(default_factory=list)

    @property
    def encoder_call_count(self) -> int:
        return int(self.get_audio_features_call_count) + int(self.thinker_forward_call_count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_load_count": self.model_load_count,
            "transcribe_call_count": self.transcribe_call_count,
            "generate_call_count": self.generate_call_count,
            "get_audio_features_call_count": self.get_audio_features_call_count,
            "thinker_forward_call_count": self.thinker_forward_call_count,
            "decoder_call_count": self.decoder_call_count,
            "encoder_call_count": self.encoder_call_count,
            "patched_methods": list(self.patched_methods),
        }


class MethodCallCounter:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._patches: list[tuple[Any, str, Any]] = []

    def patch(self, obj: Any, method_name: str, counter_name: str | None = None) -> bool:
        if obj is None or not hasattr(obj, method_name):
            return False
        original = getattr(obj, method_name)
        name = counter_name or method_name
        self._counts.setdefault(name, 0)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._counts[name] += 1
            return original(*args, **kwargs)

        try:
            setattr(obj, method_name, wrapped)
        except Exception:
            return False
        self._patches.append((obj, method_name, original))
        return True

    def count(self, name: str) -> int:
        return int(self._counts.get(name, 0))

    def snapshot(self, model_load_count: int = 0) -> RuntimeCallCounts:
        return RuntimeCallCounts(
            model_load_count=int(model_load_count),
            transcribe_call_count=self.count("transcribe"),
            generate_call_count=self.count("generate"),
            get_audio_features_call_count=self.count("get_audio_features"),
            thinker_forward_call_count=self.count("thinker_forward"),
            decoder_call_count=self.count("decoder"),
            patched_methods=sorted(self._counts),
        )

    def restore(self) -> None:
        for obj, method_name, original in reversed(self._patches):
            try:
                setattr(obj, method_name, original)
            except Exception:
                pass
        self._patches.clear()


def can_claim_verified_one_encoder_forward(status: str, counts: RuntimeCallCounts | dict[str, Any]) -> bool:
    if isinstance(counts, RuntimeCallCounts):
        encoder_calls = counts.encoder_call_count
    else:
        encoder_calls = int(counts.get("encoder_call_count", -1))
    return status == "verified_one_encoder_forward" and encoder_calls == 1


def require_blocker_text(status: str, blocker: str) -> bool:
    if status not in {"blocked_by_runtime_interface", "not_feasible_with_current_revision"}:
        return True
    required = (
        "does not expose",
        "hidden",
        "does not accept",
    )
    text = blocker.lower()
    return all(part in text for part in required)
