#!/usr/bin/env python3
from __future__ import annotations

import argparse

from vigil_final.gradient_adaptation import GradientRecipe
from vigil_final.utils import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="finetune/experiments/vigil_final/reports/gradient_onboarding_status.json")
    args = parser.parse_args()
    recipe = GradientRecipe(target="stage2_classifier_bias", learning_rate=1e-5, steps=5, l2_to_base=1e-3)
    write_json(args.output, {"status": "implemented_guard_only", "recipe": recipe.__dict__, "result": "not_run"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
