from __future__ import annotations

from typing import Any


def fpr_safety_gate(
    baseline: dict[str, Any],
    adapted: dict[str, Any],
    *,
    max_absolute_fpr: float = 0.02,
    max_fpr_increase: float = 0.02,
) -> dict[str, Any]:
    base_fpr = float(baseline.get("false_positive_rate") or 0.0)
    adapted_fpr = float(adapted.get("false_positive_rate") or 0.0)
    passed = adapted_fpr <= max_absolute_fpr and (adapted_fpr - base_fpr) <= max_fpr_increase
    return {
        "passed": bool(passed),
        "baseline_fpr": base_fpr,
        "adapted_fpr": adapted_fpr,
        "absolute_limit": max_absolute_fpr,
        "increase_limit": max_fpr_increase,
        "fpr_increase": adapted_fpr - base_fpr,
    }


def select_safe_recipe(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [row for row in candidates if row.get("safety", {}).get("passed")]
    if not safe:
        return {"selected_recipe": "no_adaptation_zero_shot_fallback", "reason": "no_safe_candidate"}
    return sorted(
        safe,
        key=lambda row: (
            float(row.get("participant_macro_recall") or 0.0),
            float(row.get("participant_macro_f1") or 0.0),
            -float(row.get("adaptation_latency_ms") or 0.0),
        ),
        reverse=True,
    )[0]
