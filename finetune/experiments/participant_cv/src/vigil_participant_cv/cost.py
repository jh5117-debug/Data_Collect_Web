from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostAccounting:
    qwen_copies: int
    qwen_full_transcript_forwards: int
    extra_qwen_encoder_forwards: int
    stage1_forwards: int

    def as_dict(self) -> dict[str, int]:
        return {
            "qwen_copies": self.qwen_copies,
            "qwen_full_transcript_forwards": self.qwen_full_transcript_forwards,
            "extra_qwen_encoder_forwards": self.extra_qwen_encoder_forwards,
            "stage1_forwards": self.stage1_forwards,
        }
