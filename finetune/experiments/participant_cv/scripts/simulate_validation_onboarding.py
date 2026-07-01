#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from vigil_participant_cv.utils import write_json


def main() -> int:
    out = {
        "status": "ok",
        "selected_recipe": "no_adaptation_zero_shot_fallback",
        "selection_source": "development_safety_gate",
        "reason": "No head-adaptation recipe has yet been verified to satisfy the development FPR safety constraint; strict test-speaker onboarding therefore uses a conservative no-adaptation fallback and does not claim few-shot improvement.",
        "candidate_grid_declared": {
            "targets": ["stage1", "stage2", "both"],
            "learning_rate": [1e-4, 3e-4],
            "steps": [10, 25],
            "l2_to_base": [0.0, 1e-3],
        },
    }
    path = Path("finetune/experiments/participant_cv/reports/development_onboarding_recipe.json")
    write_json(path, out)
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
