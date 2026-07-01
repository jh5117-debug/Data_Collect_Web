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
    return [seen[k] for k in sorted(seen)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="finetune/experiments/participant_cv/shared/balanced_max100_manifest.jsonl")
    parser.add_argument("--folds", default="finetune/experiments/participant_cv/shared/participant_folds_5fold.json")
    parser.add_argument("--out", default="finetune/experiments/participant_cv/reports/protocol_validation.json")
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    clips = clip_rows(rows)
    folds = read_json(args.folds)
    validate_unique_fold_membership(folds)
    validate_no_duplicate_hash_crosses_folds(clips, folds)
    alias_count = len({row["participant_alias"] for row in clips})
    result = {
        "status": "ok",
        "clip_rows": len(clips),
        "window_rows": len(rows),
        "participant_alias_count": alias_count,
        "fold_count": len(folds["folds"]),
        "max_clips_per_participant": max(
            sum(1 for row in clips if row["participant_alias"] == alias)
            for alias in {row["participant_alias"] for row in clips}
        ),
        "no_participant_crosses_outer_folds": True,
        "no_duplicate_audio_hash_crosses_folds": True,
    }
    write_json(args.out, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
