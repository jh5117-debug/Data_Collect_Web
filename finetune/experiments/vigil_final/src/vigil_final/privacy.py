from __future__ import annotations

import re
from pathlib import Path


PRIVATE_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),
]


def assert_public_report_text(text: str) -> None:
    for pattern in PRIVATE_PATTERNS:
        match = pattern.search(text)
        if match:
            raise ValueError(f"public report contains private-looking token: {match.group(0)[:12]}")


def assert_public_report_file(path: str | Path) -> None:
    assert_public_report_text(Path(path).read_text(encoding="utf-8"))
