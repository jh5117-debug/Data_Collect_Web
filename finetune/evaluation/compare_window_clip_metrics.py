#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "finetune" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vigil_two_stage.metrics import binary_metrics
from vigil_two_stage.utils import read_json, write_json

from aggregate_clip_predictions import (
    aggregate_clip_cascade_predictions,
    aggregate_stage_clip_predictions,
    aggregate_window_cascade_predictions,
    enrich_predictions,
    read_jsonl,
    write_jsonl,
)
from split_report import load_all_rows


STAGE2_VARIANTS = ("stage2_bce", "stage2_bce_supcon")
SPLITS = ("val", "test")
POSITIVE_PROMPTS = ("P1_vigil_only", "P2_phrase_plus_vigil", "P3_vigil_plus_phrase")
NEGATIVE_PROMPTS = ("P4_negative",)


def _metric_inputs(rows: list[dict[str, Any]], *, score_key: str, decision_key: str | None = None) -> tuple[list[int], list[float]]:
    labels = [int(row["label"]) for row in rows]
    if decision_key:
        scores = [1.0 if row.get(decision_key) else 0.0 for row in rows]
    else:
        scores = [float(row[score_key]) for row in rows]
    return labels, scores


def metrics_for_rows(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    score_key: str = "score",
    decision_key: str | None = None,
) -> dict[str, Any]:
    labels, scores = _metric_inputs(rows, score_key=score_key, decision_key=decision_key)
    return binary_metrics(labels, scores, 0.5 if decision_key else threshold)


def _group_metrics(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    threshold: float,
    score_key: str = "score",
    decision_key: str | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    return {
        key: metrics_for_rows(group, threshold=threshold, score_key=score_key, decision_key=decision_key)
        for key, group in sorted(grouped.items())
    }


def metric_breakdowns(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    score_key: str = "score",
    decision_key: str | None = None,
) -> dict[str, Any]:
    per_prompt = _group_metrics(rows, group_key="prompt_group", threshold=threshold, score_key=score_key, decision_key=decision_key)
    hard_negative_rows = [row for row in rows if int(row.get("label", 0)) == 0]
    per_hard_negative_phrase = _group_metrics(
        hard_negative_rows,
        group_key="phrase_id",
        threshold=threshold,
        score_key=score_key,
        decision_key=decision_key,
    )
    positive_recalls = {}
    for prompt in POSITIVE_PROMPTS:
        metric = per_prompt.get(prompt, {})
        positive_recalls[prompt] = metric.get("recall")
    p4_metric = per_prompt.get("P4_negative", {})
    return {
        "per_prompt_metrics": per_prompt,
        "per_hard_negative_phrase_metrics": per_hard_negative_phrase,
        "P1_recall": positive_recalls.get("P1_vigil_only"),
        "P2_recall": positive_recalls.get("P2_phrase_plus_vigil"),
        "P3_recall": positive_recalls.get("P3_vigil_plus_phrase"),
        "P4_false_positive_rate": p4_metric.get("false_positive_rate"),
    }


def metric_bundle(
    rows: list[dict[str, Any]],
    *,
    evaluation_unit: str,
    threshold: float,
    method: str,
    score_key: str = "score",
    decision_key: str | None = None,
    score_source: str | None = None,
) -> dict[str, Any]:
    metrics = metrics_for_rows(rows, threshold=threshold, score_key=score_key, decision_key=decision_key)
    metrics.update(
        {
            "method": method,
            "evaluation_unit": evaluation_unit,
            "score_key": score_key,
            "decision_key": decision_key,
            "score_source": score_source or (decision_key or score_key),
        }
    )
    metrics.update(metric_breakdowns(rows, threshold=threshold, score_key=score_key, decision_key=decision_key))
    return metrics


def _load_predictions(path: Path, manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return enrich_predictions(read_jsonl(path), manifest_rows)


def _write_optional_predictions(output_dir: Path | None, name: str, rows: list[dict[str, Any]]) -> None:
    if output_dir is not None:
        write_jsonl(output_dir / f"{name}.jsonl", rows)


def evaluate_run(
    run_dir: Path | str,
    dataset_dir: Path | str,
    *,
    top_k: int = 3,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_all_rows(dataset_dir)
    manifest_by_split = {split: [row for row in manifest_rows if row.get("split") == split] for split in SPLITS}
    stage1_metrics = read_json(run_dir / "stage1" / "metrics.json")
    theta_1 = float(stage1_metrics["theta_1"])
    result: dict[str, Any] = {
        "top_k": int(top_k),
        "theta_1": theta_1,
        "splits": {},
        "notes": [
            "Window-level and clip-level metrics are computed separately.",
            "Cascade AUROC/AUPRC use the binary final trigger as the score source.",
        ],
    }
    for split in SPLITS:
        split_result: dict[str, Any] = {}
        manifest_split = manifest_by_split[split]
        stage1_pred_path = run_dir / "stage1" / f"{split}_predictions.jsonl"
        if not stage1_pred_path.exists():
            continue
        stage1_rows = _load_predictions(stage1_pred_path, manifest_split)
        stage1_clip = aggregate_stage_clip_predictions(stage1_rows, theta_1, score_key="score", trigger_key="stage1_candidate")
        split_result["stage1"] = {
            "window": metric_bundle(stage1_rows, evaluation_unit="window", threshold=theta_1, method="stage1", score_key="score"),
            "clip": metric_bundle(stage1_clip, evaluation_unit="clip", threshold=theta_1, method="stage1", score_key="score"),
        }
        _write_optional_predictions(output_path, f"{split}_stage1_clip_predictions", stage1_clip)
        for variant in STAGE2_VARIANTS:
            metrics_path = run_dir / variant / "metrics.json"
            pred_path = run_dir / variant / f"{split}_predictions.jsonl"
            if not metrics_path.exists() or not pred_path.exists():
                continue
            stage2_metrics = read_json(metrics_path)
            if stage2_metrics.get("status") != "ok":
                continue
            theta_2 = float(stage2_metrics["theta_2"])
            stage2_rows = _load_predictions(pred_path, manifest_split)
            stage2_clip = aggregate_stage_clip_predictions(stage2_rows, theta_2, score_key="stage2_score", trigger_key="stage2_trigger")
            window_cascade = aggregate_window_cascade_predictions(stage1_rows, stage2_rows, theta_1, theta_2)
            clip_cascade = aggregate_clip_cascade_predictions(stage1_rows, stage2_rows, theta_1, theta_2, top_k=top_k)
            split_result[variant] = {
                "stage2_window": metric_bundle(
                    stage2_rows,
                    evaluation_unit="window",
                    threshold=theta_2,
                    method=f"{variant}_standalone",
                    score_key="stage2_score",
                ),
                "stage2_clip": metric_bundle(
                    stage2_clip,
                    evaluation_unit="clip",
                    threshold=theta_2,
                    method=f"{variant}_standalone",
                    score_key="score",
                ),
                "cascade_window": metric_bundle(
                    window_cascade,
                    evaluation_unit="window",
                    threshold=0.5,
                    method=f"{variant}_cascade",
                    score_key="score",
                    decision_key="final_trigger",
                    score_source="binary_final_trigger",
                ),
                "cascade_clip": metric_bundle(
                    clip_cascade,
                    evaluation_unit="clip",
                    threshold=0.5,
                    method=f"{variant}_cascade",
                    score_key="score",
                    decision_key="final_trigger",
                    score_source="binary_final_trigger",
                ),
                "theta_2": theta_2,
            }
            _write_optional_predictions(output_path, f"{split}_{variant}_stage2_clip_predictions", stage2_clip)
            _write_optional_predictions(output_path, f"{split}_{variant}_cascade_clip_predictions", clip_cascade)
        result["splits"][split] = split_result
    if output_path:
        write_json(output_path / "window_clip_metrics.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = evaluate_run(args.run_dir, args.dataset_dir, top_k=args.top_k, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

