from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from utils import write_jsonl


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_asr_runs.py"


def _prediction_row(utt_id: str, split: str, ref: str, hyp: str) -> dict:
    return {
        "id": utt_id,
        "split": split,
        "status": "success",
        "reference": ref.upper(),
        "hypothesis": hyp,
        "normalized_reference": ref,
        "normalized_hypothesis": hyp,
    }


def test_compare_run_directories_and_split_metrics(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    write_jsonl(
        base_dir / "predictions.jsonl",
        [
            _prediction_row("a", "test-clean", "hello world", "hello"),
            _prediction_row("b", "test-other", "twenty one", "twenty two"),
        ],
    )
    write_jsonl(
        cand_dir / "predictions.jsonl",
        [
            _prediction_row("a", "test-clean", "hello world", "hello world"),
            _prediction_row("b", "test-other", "twenty one", "twenty two now extra"),
            _prediction_row("c", "test-other", "extra row", "extra row"),
        ],
    )
    output = tmp_path / "compare" / "result"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline", str(base_dir), "--candidate", str(cand_dir), "--output", str(output)],
        check=True,
    )
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["formal_comparison"] is False
    assert payload["comparison_mode"] == "intersection_only"
    assert payload["common_successes"] == 2
    assert payload["candidate_only_successes"] == 1
    assert payload["baseline_wer"] == 0.5
    assert payload["candidate_wer"] == 0.75
    assert payload["absolute_wer_change"] == 0.25
    assert "test-clean" in payload["per_split"]
    assert "test-other" in payload["per_split"]
