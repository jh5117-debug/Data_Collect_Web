from __future__ import annotations

import re
from typing import Any

from vigil_two_stage.qwen_text_result import extract_qwen_text


OBJECT_REPR_RE = re.compile(r"(^[A-Za-z_][A-Za-z0-9_]*\(|language=.*text=|<.* object at 0x[0-9a-fA-F]+>)")


def safe_extract_transcript(result: Any) -> str:
    text = extract_qwen_text(result).text.strip()
    if OBJECT_REPR_RE.search(text):
        raise ValueError("transcript looks like a Python object representation")
    return text


def rolling_append(existing: str, new_text: str, max_chars: int = 1600) -> str:
    combined = " ".join(part for part in [existing.strip(), new_text.strip()] if part)
    if len(combined) <= max_chars:
        return combined
    return combined[-max_chars:].lstrip()
