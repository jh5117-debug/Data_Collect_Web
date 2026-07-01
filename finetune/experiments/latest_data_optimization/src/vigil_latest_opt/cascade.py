from __future__ import annotations

from collections import defaultdict
from typing import Any

from .utils import logit


def prediction_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["clip_id"]), int(row.get("window_index", 0))


def group_by_clip(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["clip_id"])].append(row)
    return dict(grouped)


def _merged_window_rows(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage2_by_key = {prediction_key(row): row for row in stage2_rows}
    out = []
    for row in stage1_rows:
        s2 = stage2_by_key.get(prediction_key(row))
        if s2 is None:
            continue
        merged = dict(row)
        merged["stage1_score"] = float(row["score"])
        merged["stage2_score"] = float(s2["stage2_score"])
        out.append(merged)
    return out


def fused_candidate_score(stage1_score: float, stage2_score: float, *, a: float = 1.0, b: float = 0.0, use_logit: bool = False) -> float:
    if use_logit:
        return float(a) * logit(stage2_score) + float(b) * logit(stage1_score)
    return float(stage2_score)


def clip_score_rows(
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    *,
    theta1: float,
    top_k: int,
    fusion_a: float = 1.0,
    fusion_b: float = 0.0,
    use_fusion_logit: bool = False,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    merged = _merged_window_rows(stage1_rows, stage2_rows)
    rows = []
    for clip_id, group in sorted(group_by_clip(merged).items()):
        ranked_all = sorted(group, key=lambda row: float(row["stage1_score"]), reverse=True)
        candidates = [row for row in ranked_all if float(row["stage1_score"]) >= float(theta1)][:top_k]
        best_stage1 = float(ranked_all[0]["stage1_score"]) if ranked_all else 0.0
        best_score = -1.0e9
        best_raw_stage2 = 0.0
        winning_window: dict[str, Any] | None = None
        for candidate in candidates:
            score = fused_candidate_score(
                float(candidate["stage1_score"]),
                float(candidate["stage2_score"]),
                a=fusion_a,
                b=fusion_b,
                use_logit=use_fusion_logit,
            )
            if score > best_score:
                best_score = score
                best_raw_stage2 = float(candidate["stage2_score"])
                winning_window = candidate
        first = ranked_all[0]
        rows.append(
            {
                "clip_id": clip_id,
                "label": int(first["label"]),
                "prompt_group": first.get("prompt_group"),
                "phrase_id": first.get("phrase_id"),
                "speaker_id": first.get("speaker_id"),
                "session_id": first.get("session_id"),
                "split": first.get("split"),
                "transcript": first.get("transcript"),
                "window_count": len(group),
                "stage1_clip_score": best_stage1,
                "stage1_candidate": bool(candidates),
                "evaluated_candidate_count": len(candidates),
                "top_k": int(top_k),
                "score": best_score,
                "stage2_candidate_score": best_raw_stage2,
                "winning_window_index": int(winning_window.get("window_index", 0)) if winning_window else None,
                "theta1": float(theta1),
                "fusion_a": float(fusion_a),
                "fusion_b": float(fusion_b),
                "use_fusion_logit": bool(use_fusion_logit),
            }
        )
    return rows


def apply_threshold(rows: list[dict[str, Any]], threshold: float, decision_key: str = "decision") -> list[dict[str, Any]]:
    return [{**row, decision_key: bool(float(row["score"]) >= float(threshold)), "threshold": float(threshold)} for row in rows]


def stage1_only_rows(stage1_rows: list[dict[str, Any]], theta1: float) -> list[dict[str, Any]]:
    out = []
    for clip_id, group in sorted(group_by_clip(stage1_rows).items()):
        ranked = sorted(group, key=lambda row: float(row["score"]), reverse=True)
        first = ranked[0]
        score = float(first["score"])
        out.append(
            {
                "clip_id": clip_id,
                "label": int(first["label"]),
                "prompt_group": first.get("prompt_group"),
                "speaker_id": first.get("speaker_id"),
                "split": first.get("split"),
                "score": score,
                "decision": score >= float(theta1),
                "window_count": len(group),
            }
        )
    return out
