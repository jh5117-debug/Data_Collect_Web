from __future__ import annotations

from pathlib import Path


def write_status_report(path: Path | str, title: str, rows: list[str]) -> None:
    Path(path).write_text("# " + title + "\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
