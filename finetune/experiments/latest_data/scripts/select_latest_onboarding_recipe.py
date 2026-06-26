#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from vigil_latest.utils import write_json


def main() -> int:
    reports = Path("finetune/experiments/latest_data/reports")
    reports.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "pending_few_shot_search",
        "selected_recipe": None,
        "safety_rule": {"max_absolute_fpr_increase": 0.02, "max_fpr": 0.03},
        "claim": "No few-shot improvement is claimed until a development-safe recipe improves paired outer-test metrics.",
    }
    write_json(reports / "latest_selected_few_shot_recipe.json", result)
    (reports / "LATEST_FEW_SHOT_RECIPE_SEARCH.md").write_text(
        "# Latest Few-Shot Recipe Search\n\n"
        "- Status: `pending_few_shot_search`\n"
        "- Safety gate: FPR increase <= 0.02 absolute and FPR <= 0.03 unless justified.\n",
        encoding="utf-8",
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
