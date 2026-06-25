from __future__ import annotations

from collections import defaultdict
from typing import Any

from .metrics import metric_from_decisions, participant_macro


def aggregate_fold_results(fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        by_method[str(row["method"])].append(row)
    out = {}
    for method, rows in sorted(by_method.items()):
        out[method] = {
            "pooled": metric_from_decisions(rows, "decision"),
            "participant_macro": participant_macro(rows, "decision"),
        }
    return out


def flatten_method_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, item in sorted(summary.items()):
        pooled = item.get("pooled", {})
        macro = item.get("participant_macro", {})
        rows.append(
            {
                "method": method,
                "recall": pooled.get("recall"),
                "false_positive_rate": pooled.get("false_positive_rate"),
                "precision": pooled.get("precision"),
                "f1": pooled.get("f1"),
                "participant_macro_f1": macro.get("f1"),
            }
        )
    return rows
