from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def normalize_transcript(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def edit_distance(a: str, b: str) -> int:
    aa = a.split()
    bb = b.split()
    prev = list(range(len(bb) + 1))
    for i, token_a in enumerate(aa, 1):
        cur = [i]
        for j, token_b in enumerate(bb, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if token_a == token_b else 1),
                )
            )
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class TranscriptParity:
    exact_match: bool
    normalized_match: bool
    word_edit_distance: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "normalized_match": self.normalized_match,
            "word_edit_distance": self.word_edit_distance,
        }


def compare_transcripts(public_text: str, shared_text: str | None) -> TranscriptParity | None:
    if shared_text is None:
        return None
    public_norm = normalize_transcript(public_text)
    shared_norm = normalize_transcript(shared_text)
    return TranscriptParity(
        exact_match=public_text == shared_text,
        normalized_match=public_norm == shared_norm,
        word_edit_distance=edit_distance(public_norm, shared_norm),
    )


def score_difference(separate_score: float, shared_score: float | None) -> float | None:
    if shared_score is None:
        return None
    return abs(float(separate_score) - float(shared_score))


def score_parity_status(max_abs_diff: float | None, tolerance: float = 1e-4) -> str:
    if max_abs_diff is None:
        return "blocked"
    return "passed" if max_abs_diff <= tolerance else "failed"


def cost_table_row(
    variant: str,
    qwen_weight_copies: int,
    encoder_forwards: str,
    transcript: str,
    stage2_score: str,
    median_latency_ms: float | None,
    status: str,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "qwen_weight_copies": int(qwen_weight_copies),
        "encoder_forwards": encoder_forwards,
        "transcript": transcript,
        "stage2_score": stage2_score,
        "median_latency_ms": median_latency_ms,
        "status": status,
    }
