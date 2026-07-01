from __future__ import annotations

from typing import Any


def classify_stage_error(row: dict[str, Any]) -> str | None:
    label = int(row["label"])
    final = bool(row.get("final_trigger"))
    candidate = bool(row.get("stage1_candidate"))
    if label == 1 and final:
        return None
    if label == 1 and not candidate:
        return "STAGE1_MISS"
    if label == 1 and candidate and not final:
        return "STAGE2_REJECT"
    if label == 0 and final:
        return "FINAL_FALSE_ACCEPT"
    if label == 0 and candidate:
        return "STAGE1_FALSE_CANDIDATE"
    return None
