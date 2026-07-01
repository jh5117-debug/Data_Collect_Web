#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from vigil_final.utils import write_json


def main() -> int:
    summary = {
        "status": "blocked_until_recipe_frozen",
        "reason": "outer-test few-shot evaluation must wait for development-only prototype/gradient recipe selection",
    }
    out = Path("finetune/experiments/vigil_final/reports")
    write_json(out / "real_few_shot_summary.json", summary)
    (out / "REAL_FEW_SHOT_ONBOARDING_REPORT.md").write_text(
        "# Real Few-Shot Onboarding Report\n\n"
        "Status: blocked until a development-selected onboarding recipe is frozen. "
        "No fallback few-shot values are reported as learned personalization.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
