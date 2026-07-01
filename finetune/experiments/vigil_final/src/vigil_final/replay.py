from __future__ import annotations

from typing import Any


def select_source_replay(rows: list[dict[str, Any]], *, development_aliases: set[str], hard_negative_phrases: set[str]) -> dict[str, list[dict[str, Any]]]:
    dev = [row for row in rows if str(row["participant_alias"]) in development_aliases]
    return {
        "source_positive": [row for row in dev if int(row["label"]) == 1],
        "source_hard_negative": [
            row
            for row in dev
            if int(row["label"]) == 0 and str(row.get("transcript", "")).strip().lower() in hard_negative_phrases
        ],
        "source_general_negative": [
            row
            for row in dev
            if int(row["label"]) == 0 and str(row.get("transcript", "")).strip().lower() not in hard_negative_phrases
        ],
    }
