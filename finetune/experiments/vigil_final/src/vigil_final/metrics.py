from __future__ import annotations

from collections import defaultdict
from typing import Any

from vigil_two_stage.metrics import binary_metrics


def metric_from_decisions(rows: list[dict[str, Any]], decision_key: str = "decision") -> dict[str, Any]:
    return binary_metrics([int(row["label"]) for row in rows], [1.0 if row[decision_key] else 0.0 for row in rows], 0.5)


def metric_from_score(rows: list[dict[str, Any]], score_key: str, threshold: float) -> dict[str, Any]:
    return binary_metrics([int(row["label"]) for row in rows], [float(row[score_key]) for row in rows], threshold)


def group_metrics(rows: list[dict[str, Any]], group_key: str, decision_key: str = "decision") -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, ""))].append(row)
    return {key: metric_from_decisions(value, decision_key) for key, value in sorted(groups.items())}


def participant_macro(rows: list[dict[str, Any]], decision_key: str = "decision") -> dict[str, Any]:
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_participant[str(row["participant_alias"])].append(row)
    metrics = {alias: metric_from_decisions(group, decision_key) for alias, group in sorted(by_participant.items())}
    out: dict[str, Any] = {"participants": len(metrics), "per_participant": metrics}
    for name in ("precision", "recall", "false_positive_rate", "f1"):
        vals = [float(m[name]) for m in metrics.values() if m.get(name) is not None]
        out[name] = sum(vals) / len(vals) if vals else None
    return out


def paired_delta(rows: list[dict[str, Any]], baseline_key: str, adapted_key: str) -> dict[str, Any]:
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_participant[str(row["participant_alias"])].append(row)
    deltas = []
    improved = degraded = unchanged = 0
    for group in by_participant.values():
        base = metric_from_decisions(group, baseline_key).get("f1")
        adapted = metric_from_decisions(group, adapted_key).get("f1")
        if base is None or adapted is None:
            continue
        delta = float(adapted) - float(base)
        deltas.append(delta)
        if delta > 0:
            improved += 1
        elif delta < 0:
            degraded += 1
        else:
            unchanged += 1
    if not deltas:
        return {"n": 0}
    ordered = sorted(deltas)
    return {
        "n": len(deltas),
        "mean_delta_f1": sum(deltas) / len(deltas),
        "median_delta_f1": ordered[len(ordered) // 2],
        "improved": improved,
        "degraded": degraded,
        "unchanged": unchanged,
    }
