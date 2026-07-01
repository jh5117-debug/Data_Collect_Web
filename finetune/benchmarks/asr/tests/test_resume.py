from __future__ import annotations

from pathlib import Path

from resume import append_prediction, deduplicate_predictions, load_prediction_state
from utils import read_jsonl


def test_resume_state_and_retry_failed(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    append_prediction(path, {"id": "a", "status": "success", "hypothesis": "ok"})
    append_prediction(path, {"id": "b", "status": "failed", "error": "boom"})
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json}\n")

    completed, corrupted = load_prediction_state(path, retry_failed=False)
    assert corrupted == 1
    assert sorted(completed) == ["a", "b"]

    completed_retry, corrupted_retry = load_prediction_state(path, retry_failed=True)
    assert corrupted_retry == 1
    assert sorted(completed_retry) == ["a"]


def test_deduplicate_predictions_last_row_wins(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    append_prediction(path, {"id": "b", "status": "success", "hypothesis": "old"})
    append_prediction(path, {"id": "a", "status": "success", "hypothesis": "ok"})
    append_prediction(path, {"id": "b", "status": "success", "hypothesis": "new"})
    stats = deduplicate_predictions(path)
    rows = read_jsonl(path)
    assert stats["input_rows"] == 3
    assert [row["id"] for row in rows] == ["a", "b"]
    assert rows[1]["hypothesis"] == "new"
