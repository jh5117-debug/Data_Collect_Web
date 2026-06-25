#!/usr/bin/env python3
from __future__ import annotations

from vigil_final.safety import fpr_safety_gate, select_safe_recipe
from vigil_final.utils import write_csv, write_json


def main() -> int:
    candidates = []
    for method in ("prototype_only", "base_plus_prototype"):
        for alpha in (0.25, 0.5, 1.0, 2.0):
            baseline = {"false_positive_rate": 0.0046512}
            adapted = {"false_positive_rate": 0.0046512}
            safety = fpr_safety_gate(baseline, adapted)
            candidates.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "beta": 0.0,
                    "participant_macro_recall": 0.0,
                    "participant_macro_f1": 0.0,
                    "adaptation_latency_ms": 0.0,
                    "safety": safety,
                    "status": "requires_embedding_index",
                }
            )
    selected = select_safe_recipe(candidates)
    if selected.get("participant_macro_f1") == 0.0:
        selected = {"selected_recipe": "no_adaptation_zero_shot_fallback", "reason": "prototype_search_not_run_without_embedding_index"}
    write_csv("finetune/experiments/vigil_final/reports/development_prototype_search.csv", candidates)
    write_json("finetune/experiments/vigil_final/reports/development_selected_prototype_recipe.json", selected)
    report = [
        "# Prototype Recipe Selection",
        "",
        "Status: pending real embedding-index evaluation.",
        "",
        "The code path enforces development-only pseudo-target selection and FPR safety gates. No learned prototype improvement is claimed until the embedding-index search is run.",
    ]
    from pathlib import Path

    Path("finetune/experiments/vigil_final/reports/PROTOTYPE_RECIPE_SELECTION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
