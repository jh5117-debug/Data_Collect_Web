#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = ("stage2_bce", "stage2_bce_supcon")


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _metric(metrics: dict[str, Any], key: str, default: float) -> float:
    value = metrics.get(key)
    return default if value is None else float(value)


def select_stage2_variant(
    window_clip_metrics: dict[str, Any],
    *,
    recall_constraint: float = 0.90,
    latency_by_variant: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    latency_by_variant = latency_by_variant or {}
    candidates = []
    val_metrics_by_variant = {}
    test_metrics_by_variant = {}
    for variant in VARIANTS:
        val_entry = window_clip_metrics.get("splits", {}).get("val", {}).get(variant, {}).get("cascade_clip")
        if not val_entry:
            continue
        test_entry = window_clip_metrics.get("splits", {}).get("test", {}).get(variant, {}).get("cascade_clip")
        val_metrics_by_variant[variant] = val_entry
        if test_entry:
            test_metrics_by_variant[variant] = test_entry
        latency = latency_by_variant.get(variant)
        latency_sort = float(latency) if latency is not None else 0.0
        recall = _metric(val_entry, "recall", -1.0)
        hard_negative_fpr = _metric(val_entry, "P4_false_positive_rate", 1.0)
        precision = _metric(val_entry, "precision", -1.0)
        f1 = _metric(val_entry, "f1", -1.0)
        candidates.append(
            {
                "variant": variant,
                "meets_recall_constraint": recall >= recall_constraint,
                "validation_recall": recall,
                "validation_hard_negative_false_positive_rate": hard_negative_fpr,
                "validation_precision": precision,
                "validation_f1": f1,
                "latency": latency,
                "sort_key": (
                    1 if recall >= recall_constraint else 0,
                    -hard_negative_fpr,
                    precision,
                    f1,
                    -latency_sort,
                ),
            }
        )
    if not candidates:
        return {
            "status": "blocked",
            "reason": "no validation clip-level cascade metrics available",
            "selected_variant": None,
        }
    feasible = [item for item in candidates if item["meets_recall_constraint"]]
    if feasible:
        selected = sorted(candidates, key=lambda item: item["sort_key"], reverse=True)[0]
        fallback_note = ""
    else:
        selected = sorted(
            candidates,
            key=lambda item: (
                item["validation_recall"],
                -item["validation_hard_negative_false_positive_rate"],
                item["validation_precision"],
                item["validation_f1"],
                -(float(item["latency"]) if item["latency"] is not None else 0.0),
            ),
            reverse=True,
        )[0]
        fallback_note = " No variant met the recall constraint, so the closest validation recall was used before the remaining tie-breakers."
    return {
        "status": "ok",
        "selection_rule": [
            "validation recall constraint",
            "validation hard-negative false-positive rate",
            "validation precision",
            "validation F1",
            "latency tie-breaker",
        ],
        "recall_constraint": recall_constraint,
        "selected_variant": selected["variant"],
        "selection_reason": (
            f"Selected {selected['variant']} using validation clip-level cascade metrics: "
            f"recall={selected['validation_recall']}, "
            f"P4_FPR={selected['validation_hard_negative_false_positive_rate']}, "
            f"precision={selected['validation_precision']}, f1={selected['validation_f1']}."
            f"{fallback_note}"
        ),
        "candidates": [{k: v for k, v in item.items() if k != "sort_key"} for item in candidates],
        "validation_metrics": val_metrics_by_variant,
        "test_metrics_after_selection": test_metrics_by_variant,
        "selection_changed_from_previous_reported_default": selected["variant"] != "stage2_bce_supcon",
        "test_metrics_used_for_selection": False,
    }


def write_model_selection_markdown(path: Path | str, selection: dict[str, Any]) -> None:
    lines = [
        "# Model Selection",
        "",
        "Selection uses validation clip-level cascade metrics only. Test metrics are shown only after the variant is selected.",
        "",
        f"- Status: `{selection.get('status')}`",
        f"- Selected variant: `{selection.get('selected_variant')}`",
        f"- Test metrics used for selection: {selection.get('test_metrics_used_for_selection')}",
        f"- Changed from previous reported default: {selection.get('selection_changed_from_previous_reported_default')}",
        "",
        "## Reason",
        "",
        selection.get("selection_reason") or selection.get("reason", ""),
        "",
        "## Candidates",
        "",
    ]
    for item in selection.get("candidates", []):
        lines.append(
            "- "
            f"{item['variant']}: recall={item['validation_recall']}, "
            f"P4_FPR={item['validation_hard_negative_false_positive_rate']}, "
            f"precision={item['validation_precision']}, f1={item['validation_f1']}, "
            f"meets_recall={item['meets_recall_constraint']}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-clip-metrics", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--recall-constraint", type=float, default=0.90)
    args = parser.parse_args()
    metrics = json.loads(Path(args.window_clip_metrics).read_text(encoding="utf-8"))
    selection = select_stage2_variant(metrics, recall_constraint=args.recall_constraint)
    write_json(args.output_json, selection)
    write_model_selection_markdown(args.output_md, selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
