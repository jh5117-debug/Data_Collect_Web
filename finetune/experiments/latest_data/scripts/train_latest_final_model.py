#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from vigil_latest.utils import write_json


def main() -> int:
    reports = Path("finetune/experiments/latest_data/reports")
    reports.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    result = {
        "status": "not_trained_choices_not_frozen",
        "code_commit": commit,
        "bundle_manifest_template": {
            "stage1_checkpoint": None,
            "stage2_checkpoint": None,
            "thresholds": None,
            "dataset_fingerprint": None,
            "manifest_checksum": None,
            "fold_checksum": None,
        },
    }
    write_json(reports / "latest_final_model_status.json", result)
    (reports / "LATEST_FINAL_MODEL_BUNDLE_REPORT.md").write_text(
        "# Latest Final Model Bundle Report\n\n"
        "- Status: `not_trained_choices_not_frozen`\n"
        "- This script writes the sanitized bundle manifest template only. It does not commit checkpoints.\n",
        encoding="utf-8",
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
