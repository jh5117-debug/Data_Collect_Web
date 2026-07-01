#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

from vigil_final.blind_test import BlindTestLock, validate_lock
from vigil_final.utils import write_json


def main() -> int:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    lock = BlindTestLock(
        code_commit=commit,
        selected_method="pending_final_selection",
        stage1_threshold=0.0,
        stage2_threshold=0.0,
        balanced_dataset_checksum="44815508a013b9022a8efc99a3972b6847884ebbb3578e356f18a50b822f5a03",
        fold_checksum="e6759ee22e4358c2d7f4a3578b8568d6eb829ab7cfad69c4441cdc11b57d01cb",
        onboarding_recipe={"status": "pending_final_selection"},
        inference_stride=0.25,
        top_k=3,
        locked_date=date.today().isoformat(),
    ).to_json()
    validate_lock(lock)
    out = Path("finetune/experiments/vigil_final/reports")
    write_json(out / "blind_test_lock.json", lock)
    (out / "FINAL_BLIND_TEST_PROTOCOL.md").write_text(
        "# Future New-Participant Blind-Test Protocol\n\n"
        "- Accept a future export ZIP.\n"
        "- Reject known development participants.\n"
        "- Do not train base models on blind-test participants.\n"
        "- Optional onboarding uses exactly 3 or 5 positive support clips and removes support from query.\n"
        "- Do not tune thresholds or use query negatives for adaptation.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
