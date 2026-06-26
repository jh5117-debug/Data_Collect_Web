#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vigil_participant_cv.protocol import validate_no_duplicate_hash_crosses_folds, validate_unique_fold_membership
from vigil_participant_cv.utils import read_json, read_jsonl, write_json


def clip_rows(rows: list[dict]) -> list[dict]:
    seen = {}
    for row in rows:
        seen.setdefault(str(row["clip_id"]), row)
    return [seen[key] for key in sorted(seen)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="finetune/experiments/latest_data/shared/balanced_max100_latest_manifest.jsonl")
    parser.add_argument("--folds", default="finetune/experiments/latest_data/shared/latest_participant_folds_5fold.json")
    parser.add_argument("--out", default="finetune/experiments/latest_data/reports/latest_protocol_validation.json")
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    clips = clip_rows(rows)
    folds = read_json(args.folds)
    validate_unique_fold_membership(folds)
    validate_no_duplicate_hash_crosses_folds(clips, folds)
    aliases = {row["participant_alias"] for row in clips}
    counts = {alias: sum(1 for row in clips if row["participant_alias"] == alias) for alias in aliases}
    result = {
        "status": "ok",
        "clip_rows": len(clips),
        "window_rows": len(rows),
        "participant_alias_count": len(aliases),
        "fold_count": len(folds["folds"]),
        "max_clips_per_participant": max(counts.values()) if counts else 0,
        "no_participant_crosses_outer_folds": True,
        "no_duplicate_audio_hash_crosses_folds": True,
    }
    write_json(args.out, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
