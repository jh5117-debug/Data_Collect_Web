#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from vigil_final.utils import read_json, write_json


def main() -> int:
    proto = Path("finetune/experiments/vigil_final/reports/development_selected_prototype_recipe.json")
    grad = Path("finetune/experiments/vigil_final/reports/development_selected_gradient_recipe.json")
    fewshot = Path("finetune/experiments/vigil_final/reports/real_few_shot_summary.json")
    fewshot_summary = read_json(fewshot) if fewshot.exists() else {"learned_personalization_claimed": False}
    selected_method = "prototype_personalization" if fewshot_summary.get("learned_personalization_claimed") else "no_adaptation_zero_shot_fallback"
    result = {
        "status": "ok" if fewshot.exists() else "pending_real_search",
        "prototype": read_json(proto) if proto.exists() else {"status": "missing"},
        "gradient": read_json(grad) if grad.exists() else {"status": "missing"},
        "selected_method": selected_method,
        "outer_test_result": fewshot_summary,
        "reason": "prototype is selected when development-safe; gradient remains unselected unless a safe completed search beats it",
    }
    write_json("finetune/experiments/vigil_final/reports/selected_few_shot_method.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
