from __future__ import annotations

from typing import Any


def validate_source_replay(rows: list[dict[str, Any]], development_aliases: set[str], target_alias: str) -> None:
    for row in rows:
        alias = str(row["participant_alias"])
        if alias == target_alias:
            raise ValueError("target participant leaked into source replay")
        if alias not in development_aliases:
            raise ValueError("source replay contains non-development participant")
