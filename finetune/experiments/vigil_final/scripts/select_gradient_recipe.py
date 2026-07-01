#!/usr/bin/env python3
from __future__ import annotations

from vigil_final.safety import select_safe_recipe
from vigil_final.utils import write_csv, write_json


def main() -> int:
    rows = []
    for target in ("stage2_classifier_bias", "stage2_classifier", "stage2_embedding_and_classifier", "stage1_classifier", "both_final_classifiers"):
        rows.append({"target": target, "status": "not_run", "reason": "requires development pseudo-target training"})
    selected = select_safe_recipe(rows)
    write_csv("finetune/experiments/vigil_final/reports/development_gradient_search.csv", rows)
    write_json("finetune/experiments/vigil_final/reports/development_selected_gradient_recipe.json", selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
