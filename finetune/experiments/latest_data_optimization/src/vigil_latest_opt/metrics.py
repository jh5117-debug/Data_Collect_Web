from __future__ import annotations

from collections import defaultdict
from typing import Any


def binary_metrics(labels: list[int], decisions: list[bool]) -> dict[str, Any]:
    if len(labels) != len(decisions):
        raise ValueError("labels and decisions must have equal length")
    tp = tn = fp = fn = 0
    for label, decision in zip(labels, decisions):
        y = int(label)
        pred = bool(decision)
        if pred and y == 1:
            tp += 1
        elif pred and y == 0:
            fp += 1
        elif not pred and y == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "n": len(labels),
        "positive": tp + fn,
        "negative": tn + fp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_reject_rate": fn / (tp + fn) if tp + fn else None,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0 if recall is not None and specificity is not None else None,
    }


def metrics_from_rows(rows: list[dict[str, Any]], decision_key: str = "decision") -> dict[str, Any]:
    return binary_metrics([int(row["label"]) for row in rows], [bool(row[decision_key]) for row in rows])


def group_metrics(rows: list[dict[str, Any]], group_key: str, decision_key: str = "decision") -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    return {key: metrics_from_rows(group, decision_key) for key, group in sorted(grouped.items())}


def participant_macro(rows: list[dict[str, Any]], decision_key: str = "decision") -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        alias = str(row.get("participant_alias") or row.get("speaker_id") or "unknown")
        grouped[alias].append(row)
    per = {alias: metrics_from_rows(group, decision_key) for alias, group in sorted(grouped.items())}
    out: dict[str, Any] = {"participants": len(per), "per_participant": per}
    for key in ("precision", "recall", "false_positive_rate", "f1"):
        vals = [float(m[key]) for m in per.values() if m.get(key) is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def paired_delta(rows: list[dict[str, Any]], baseline_key: str, adapted_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        alias = str(row.get("participant_alias") or row.get("speaker_id") or "unknown")
        grouped[alias].append(row)
    deltas = []
    improved = degraded = unchanged = 0
    for group in grouped.values():
        base = metrics_from_rows(group, baseline_key).get("f1")
        adapted = metrics_from_rows(group, adapted_key).get("f1")
        if base is None or adapted is None:
            continue
        delta = float(adapted) - float(base)
        deltas.append(delta)
        if delta > 1e-12:
            improved += 1
        elif delta < -1e-12:
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
