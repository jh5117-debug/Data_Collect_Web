from __future__ import annotations

import re
import unicodedata

_APOSTROPHES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u02bc": "'",
    "`": "'",
}


def normalize_librispeech_text(text: str) -> str:
    """Project deterministic English normalizer for LibriSpeech WER."""

    text = unicodedata.normalize("NFKC", text)
    for src, dst in _APOSTROPHES.items():
        text = text.replace(src, dst)
    text = text.replace("-", " ")
    text = text.lower()
    # Keep apostrophes only when they are inside a word, e.g. don't.
    text = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_raw_for_scoring(text: str) -> str:
    """Minimal raw-score normalization: Unicode plus whitespace only."""

    text = unicodedata.normalize("NFKC", text)
    for src, dst in _APOSTROPHES.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip()
