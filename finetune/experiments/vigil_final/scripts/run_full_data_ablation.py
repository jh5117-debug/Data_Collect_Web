#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from vigil_final.utils import read_json, write_json


def main() -> int:
    nested = Path("finetune/experiments/vigil_final/reports/nested_zero_shot_summary.json")
    balanced = read_json(nested) if nested.exists() else {"status": "missing_nested_v2"}
    summary = {"status": "not_run", "balanced_reference": balanced, "full_unbalanced": None, "reason": "formal full-data nested ablation not yet launched"}
    out = Path("finetune/experiments/vigil_final/reports")
    write_json(out / "balanced_vs_full_summary.json", summary)
    (out / "BALANCED_VS_FULL_ABLATION_REPORT.md").write_text(
        "# Balanced Versus Full-Data Ablation\n\n"
        "Status: not run. No full-unbalanced performance is claimed.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
