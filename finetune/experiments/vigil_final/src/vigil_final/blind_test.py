from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BlindTestLock:
    code_commit: str
    selected_method: str
    stage1_threshold: float
    stage2_threshold: float
    balanced_dataset_checksum: str
    fold_checksum: str
    onboarding_recipe: dict[str, Any]
    inference_stride: float
    top_k: int
    locked_date: str

    def to_json(self) -> dict[str, Any]:
        return {
            "code_commit": self.code_commit,
            "selected_method": self.selected_method,
            "stage1_threshold": self.stage1_threshold,
            "stage2_threshold": self.stage2_threshold,
            "balanced_dataset_checksum": self.balanced_dataset_checksum,
            "fold_checksum": self.fold_checksum,
            "onboarding_recipe": self.onboarding_recipe,
            "inference_stride": self.inference_stride,
            "top_k": self.top_k,
            "locked_date": self.locked_date,
        }


def reject_known_participants(export_aliases: set[str], known_aliases: set[str]) -> None:
    overlap = sorted(export_aliases & known_aliases)
    if overlap:
        raise ValueError(f"blind-test export contains known development participants: {overlap}")


def validate_lock(lock: dict[str, Any]) -> None:
    required = {
        "code_commit",
        "selected_method",
        "stage1_threshold",
        "stage2_threshold",
        "balanced_dataset_checksum",
        "fold_checksum",
        "onboarding_recipe",
        "inference_stride",
        "top_k",
        "locked_date",
    }
    missing = sorted(required - set(lock))
    if missing:
        raise ValueError(f"blind-test lock missing fields: {missing}")
