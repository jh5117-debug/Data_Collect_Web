#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from vigil_final.long_speech import summarize_stress
from vigil_final.utils import write_json


def main() -> int:
    summary = {"status": "blocked_until_final_model_locked", "stress_summary": summarize_stress([], 1.0)}
    out = Path("finetune/experiments/vigil_final/reports")
    write_json(out / "long_speech_summary.json", summary)
    (out / "LONG_SPEECH_FALSE_ACTIVATION_REPORT.md").write_text(
        "# Long General-Speech False-Activation Stress Test\n\n"
        "Status: blocked until final model and thresholds are locked. No FAPH result is claimed.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
