from __future__ import annotations

from collections import defaultdict
from typing import Any


def group_by_clip(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["clip_id"])].append(row)
    return dict(grouped)


def stage1_clip_score(rows: list[dict[str, Any]], *, score_key: str = "score") -> float:
    return max((float(row[score_key]) for row in rows), default=0.0)


def cascade_clip_decision(
    stage1_rows: list[dict[str, Any]],
    stage2_by_window: dict[tuple[str, int], float],
    *,
    theta_1: float,
    theta_2: float,
    top_k: int = 3,
) -> dict[str, Any]:
    candidates = sorted(
        [row for row in stage1_rows if float(row["score"]) >= theta_1],
        key=lambda row: float(row["score"]),
        reverse=True,
    )[:top_k]
    diagnostics = []
    for row in candidates:
        key = (str(row["clip_id"]), int(row.get("window_index", 0)))
        stage2_score = stage2_by_window.get(key)
        accepted = stage2_score is not None and stage2_score >= theta_2
        diagnostics.append(
            {
                "window_index": key[1],
                "stage1_score": float(row["score"]),
                "stage2_score": stage2_score,
                "accepted": bool(accepted),
            }
        )
    return {
        "stage1_clip_score": stage1_clip_score(stage1_rows),
        "stage1_candidate": bool(candidates),
        "candidate_windows": diagnostics,
        "final_trigger": any(item["accepted"] for item in diagnostics),
    }
