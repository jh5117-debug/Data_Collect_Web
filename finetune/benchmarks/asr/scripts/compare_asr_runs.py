#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from scoring import score_pairs
from utils import ensure_dir, read_jsonl, utc_timestamp, write_json


def _prediction_path(path: Path) -> Path:
    if path.is_dir():
        return path / "predictions.jsonl"
    return path


def _success_by_id(path: Path) -> dict[str, dict[str, Any]]:
    path = _prediction_path(path)
    rows = read_jsonl(path)
    return {str(row["id"]): row for row in rows if row.get("status") == "success" and row.get("id")}


def _per_row_error(row: dict) -> int:
    metrics = score_pairs([str(row.get("normalized_reference", ""))], [str(row.get("normalized_hypothesis", ""))])
    return int(metrics["substitutions"] or 0) + int(metrics["deletions"] or 0) + int(metrics["insertions"] or 0)


def _score_ids(rows_by_id: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, float | int | None]:
    refs = [str(rows_by_id[utt]["normalized_reference"]) for utt in ids]
    hyps = [str(rows_by_id[utt]["normalized_hypothesis"]) for utt in ids]
    return score_pairs(refs, hyps)


def _split_metrics(
    base: dict[str, dict[str, Any]],
    cand: dict[str, dict[str, Any]],
    common_ids: list[str],
) -> dict[str, dict[str, Any]]:
    splits = sorted({str(base[utt].get("split", "unknown")) for utt in common_ids})
    out: dict[str, dict[str, Any]] = {}
    for split in splits:
        split_ids = [utt for utt in common_ids if str(base[utt].get("split", "unknown")) == split]
        base_metrics = _score_ids(base, split_ids)
        cand_metrics = _score_ids(cand, split_ids)
        out[split] = {
            "common_ids": len(split_ids),
            "baseline_wer": base_metrics["wer"],
            "candidate_wer": cand_metrics["wer"],
            "absolute_wer_change": _delta(cand_metrics["wer"], base_metrics["wer"]),
            "relative_wer_change": _relative_delta(cand_metrics["wer"], base_metrics["wer"]),
            "baseline_substitutions": base_metrics["substitutions"],
            "candidate_substitutions": cand_metrics["substitutions"],
            "baseline_deletions": base_metrics["deletions"],
            "candidate_deletions": cand_metrics["deletions"],
            "baseline_insertions": base_metrics["insertions"],
            "candidate_insertions": cand_metrics["insertions"],
        }
    return out


def _delta(candidate: float | int | None, baseline: float | int | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def _relative_delta(candidate: float | int | None, baseline: float | int | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return (float(candidate) - float(baseline)) / float(baseline)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two ASR prediction runs on shared utterance IDs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_predictions = _prediction_path(args.baseline)
    candidate_predictions = _prediction_path(args.candidate)
    base = _success_by_id(baseline_predictions)
    cand = _success_by_id(candidate_predictions)
    common_ids = sorted(set(base) & set(cand))
    if not common_ids:
        raise SystemExit("no common successful utterance IDs")
    base_metrics = _score_ids(base, common_ids)
    cand_metrics = _score_ids(cand, common_ids)

    changes = []
    improved = degraded = unchanged = 0
    for utt_id in common_ids:
        base_err = _per_row_error(base[utt_id])
        cand_err = _per_row_error(cand[utt_id])
        delta = cand_err - base_err
        if delta < 0:
            improved += 1
        elif delta > 0:
            degraded += 1
        else:
            unchanged += 1
        changes.append(
            {
                "id": utt_id,
                "split": base[utt_id].get("split"),
                "reference": base[utt_id].get("reference"),
                "normalized_reference": base[utt_id].get("normalized_reference"),
                "baseline_hypothesis": base[utt_id].get("hypothesis"),
                "candidate_hypothesis": cand[utt_id].get("hypothesis"),
                "baseline_normalized_hypothesis": base[utt_id].get("normalized_hypothesis"),
                "candidate_normalized_hypothesis": cand[utt_id].get("normalized_hypothesis"),
                "baseline_errors": base_err,
                "candidate_errors": cand_err,
                "error_delta": delta,
            }
        )
    largest_degradations = sorted(
        [row for row in changes if int(row["error_delta"]) > 0],
        key=lambda row: (int(row["error_delta"]), str(row["id"])),
        reverse=True,
    )[:20]
    changes.sort(key=lambda row: (int(row["error_delta"]), str(row["id"])), reverse=True)

    formal_comparison = set(base) == set(cand)

    payload = {
        "created_at_utc": utc_timestamp(),
        "baseline": str(baseline_predictions.resolve()),
        "candidate": str(candidate_predictions.resolve()),
        "formal_comparison": formal_comparison,
        "comparison_mode": "same_ids" if formal_comparison else "intersection_only",
        "baseline_only_successes": len(set(base) - set(cand)),
        "candidate_only_successes": len(set(cand) - set(base)),
        "common_successes": len(common_ids),
        "baseline_metrics_on_common": base_metrics,
        "candidate_metrics_on_common": cand_metrics,
        "baseline_wer": base_metrics["wer"],
        "candidate_wer": cand_metrics["wer"],
        "absolute_wer_change": _delta(cand_metrics["wer"], base_metrics["wer"]),
        "relative_wer_change": _relative_delta(cand_metrics["wer"], base_metrics["wer"]),
        "substitution_change": _delta(cand_metrics["substitutions"], base_metrics["substitutions"]),
        "deletion_change": _delta(cand_metrics["deletions"], base_metrics["deletions"]),
        "insertion_change": _delta(cand_metrics["insertions"], base_metrics["insertions"]),
        "per_split": _split_metrics(base, cand, common_ids),
        "utterances_improved": improved,
        "utterances_degraded": degraded,
        "utterances_unchanged": unchanged,
        "largest_degradations": largest_degradations,
    }
    ensure_dir(args.output.parent)
    write_json(args.output.with_suffix(".json"), payload)
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes[:200])
    args.output.with_suffix(".md").write_text(
        "# ASR Run Comparison\n\n"
        f"Common successful utterances: {len(common_ids)}\n\n"
        f"Comparison mode: {payload['comparison_mode']}\n\n"
        f"Baseline WER: {base_metrics['wer']}\n\n"
        f"Candidate WER: {cand_metrics['wer']}\n\n"
        f"Absolute WER change: {payload['absolute_wer_change']}\n\n"
        f"Relative WER change: {payload['relative_wer_change']}\n\n"
        f"S/D/I change: {payload['substitution_change']} / {payload['deletion_change']} / {payload['insertion_change']}\n\n"
        f"Formal same-ID comparison: {payload['formal_comparison']}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
