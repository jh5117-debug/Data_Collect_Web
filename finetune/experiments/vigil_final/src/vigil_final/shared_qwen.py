from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SharedQwenDiagnostic:
    status: str
    public_transcribe_path: str | None
    audio_feature_path: str | None
    blocker: str | None
    required_interface: str | None
    call_counts: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "public_transcribe_path": self.public_transcribe_path,
            "audio_feature_path": self.audio_feature_path,
            "blocker": self.blocker,
            "required_interface": self.required_interface,
            "call_counts": self.call_counts,
        }


VALID_SHARED_STATUSES = {
    "verified_one_encoder_forward",
    "partial_internal_prototype",
    "blocked_by_runtime_interface",
    "not_feasible_with_current_revision",
}


def validate_shared_status(status: str) -> None:
    if status not in VALID_SHARED_STATUSES:
        raise ValueError(f"invalid shared-Qwen status: {status}")
