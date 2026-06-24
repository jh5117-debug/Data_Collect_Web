from __future__ import annotations

import re
from typing import Iterable


ALIAS_RE = re.compile(r"^P\d{3}$")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
SPEAKER_HASH_RE = re.compile(r"\bspk_[0-9a-fA-F]{8,}\b")


def build_alias_map(speaker_ids: Iterable[str]) -> dict[str, str]:
    unique = sorted({str(s) for s in speaker_ids})
    return {speaker_id: f"P{i + 1:03d}" for i, speaker_id in enumerate(unique)}


def assert_public_text_is_sanitized(text: str) -> None:
    if EMAIL_RE.search(text):
        raise ValueError("public artifact contains an email-like string")
    if SPEAKER_HASH_RE.search(text):
        raise ValueError("public artifact contains a raw speaker hash")


def assert_alias(value: str) -> None:
    if not ALIAS_RE.match(value):
        raise ValueError(f"not a privacy-safe participant alias: {value}")
