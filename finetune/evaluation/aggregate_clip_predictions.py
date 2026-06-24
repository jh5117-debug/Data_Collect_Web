#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class PredictionConsistencyError(ValueError):
    pass


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def group_by_clip(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["clip_id"])].append(row)
    return dict(grouped)


def prediction_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["clip_id"]), int(row.get("window_index", 0))


def assert_clip_consistent(group: list[dict[str, Any]], fields: tuple[str, ...] = ("label", "split", "speaker_id")) -> None:
    if not group:
        raise PredictionConsistencyError("empty clip group")
    clip_id = str(group[0].get("clip_id", ""))
    for field in fields:
        values = {str(row.get(field, "")) for row in group}
        if len(values) > 1:
            raise PredictionConsistencyError(f"clip {clip_id} has inconsistent {field}: {sorted(values)}")


def clip_metadata(group: list[dict[str, Any]]) -> dict[str, Any]:
    assert_clip_consistent(group)
    first = sorted(group, key=lambda row: int(row.get("window_index", 0)))[0]
    meta = {
        key: first.get(key)
        for key in (
            "clip_id",
            "speaker_id",
            "session_id",
            "prompt_group",
            "transcript",
            "label",
            "phrase_id",
            "split",
            "full_wav_path",
            "audio_sha256",
            "full_wav_sha256",
        )
        if key in first
    }
    meta["window_count"] = len(group)
    return meta


def enrich_predictions(predictions: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_by_key = {prediction_key(row): row for row in manifest_rows}
    enriched = []
    for row in predictions:
        merged = dict(row)
        manifest = manifest_by_key.get(prediction_key(row))
        if manifest:
            for key in (
                "window_start_sec",
                "window_end_sec",
                "window_wav_path",
                "full_wav_path",
                "audio_sha256",
                "full_wav_sha256",
                "prompt_title",
            ):
                if key in manifest and key not in merged:
                    merged[key] = manifest[key]
        enriched.append(merged)
    return enriched


def aggregate_stage_clip_predictions(
    rows: list[dict[str, Any]],
    threshold: float,
    *,
    score_key: str = "score",
    trigger_key: str = "stage1_candidate",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for clip_id, group in sorted(group_by_clip(rows).items()):
        assert_clip_consistent(group)
        ranked = sorted(group, key=lambda row: float(row[score_key]), reverse=True)
        best = ranked[0]
        score = float(best[score_key])
        row = clip_metadata(group)
        row.update(
            {
                "clip_id": clip_id,
                "score": score,
                "max_window_score": score,
                trigger_key: score >= threshold,
                "candidate_window_index": int(best.get("window_index", 0)),
                "candidate_window_start_sec": best.get("window_start_sec"),
                "candidate_window_end_sec": best.get("window_end_sec"),
                "score_key": score_key,
                "threshold": float(threshold),
            }
        )
        out.append(row)
    return out


def aggregate_window_cascade_predictions(
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    theta_1: float,
    theta_2: float,
) -> list[dict[str, Any]]:
    stage2_by_key = {prediction_key(row): row for row in stage2_rows}
    out = []
    for row in stage1_rows:
        s2 = stage2_by_key.get(prediction_key(row))
        if not s2:
            continue
        stage1_score = float(row["score"])
        stage2_score = float(s2["stage2_score"])
        candidate = stage1_score >= theta_1
        final_trigger = candidate and stage2_score >= theta_2
        merged = dict(row)
        merged.update(
            {
                "stage1_score": stage1_score,
                "stage2_score": stage2_score,
                "candidate": candidate,
                "final_trigger": final_trigger,
                "score": 1.0 if final_trigger else 0.0,
            }
        )
        out.append(merged)
    return out


def aggregate_clip_cascade_predictions(
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    theta_1: float,
    theta_2: float,
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    stage2_by_key = {prediction_key(row): row for row in stage2_rows}
    out: list[dict[str, Any]] = []
    for clip_id, group in sorted(group_by_clip(stage1_rows).items()):
        assert_clip_consistent(group)
        candidates = sorted(
            [row for row in group if float(row["score"]) >= theta_1],
            key=lambda row: float(row["score"]),
            reverse=True,
        )[:top_k]
        diagnostics = []
        final_trigger = False
        winning: dict[str, Any] | None = None
        for candidate in candidates:
            key = prediction_key(candidate)
            s2 = stage2_by_key.get(key)
            stage2_score = float(s2["stage2_score"]) if s2 is not None else None
            accepted = stage2_score is not None and stage2_score >= theta_2
            item = {
                "window_index": int(candidate.get("window_index", 0)),
                "window_start_sec": candidate.get("window_start_sec"),
                "window_end_sec": candidate.get("window_end_sec"),
                "stage1_score": float(candidate["score"]),
                "stage2_score": stage2_score,
                "accepted": bool(accepted),
            }
            diagnostics.append(item)
            if accepted and winning is None:
                final_trigger = True
                winning = item
        best_stage1 = max((float(row["score"]) for row in group), default=0.0)
        best_stage2_candidate = max(
            (float(item["stage2_score"]) for item in diagnostics if item["stage2_score"] is not None),
            default=0.0,
        )
        row = clip_metadata(group)
        row.update(
            {
                "stage1_clip_score": best_stage1,
                "stage1_candidate": bool(candidates),
                "evaluated_candidate_count": len(candidates),
                "top_k": int(top_k),
                "candidate_windows": diagnostics,
                "winning_candidate": winning,
                "stage2_candidate_score": best_stage2_candidate,
                "final_trigger": bool(final_trigger),
                "score": 1.0 if final_trigger else 0.0,
                "theta_1": float(theta_1),
                "theta_2": float(theta_2),
            }
        )
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-predictions", required=True)
    parser.add_argument("--stage2-predictions")
    parser.add_argument("--theta-1", type=float, required=True)
    parser.add_argument("--theta-2", type=float)
    parser.add_argument("--manifest")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stage1 = read_jsonl(args.stage1_predictions)
    if args.manifest:
        stage1 = enrich_predictions(stage1, read_jsonl(args.manifest))
    if args.stage2_predictions:
        if args.theta_2 is None:
            raise SystemExit("--theta-2 is required with --stage2-predictions")
        stage2 = read_jsonl(args.stage2_predictions)
        if args.manifest:
            stage2 = enrich_predictions(stage2, read_jsonl(args.manifest))
        rows = aggregate_clip_cascade_predictions(stage1, stage2, args.theta_1, args.theta_2, top_k=args.top_k)
    else:
        rows = aggregate_stage_clip_predictions(stage1, args.theta_1)
    write_jsonl(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
