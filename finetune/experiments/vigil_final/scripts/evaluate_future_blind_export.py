#!/usr/bin/env python3
from __future__ import annotations

import argparse

from vigil_final.blind_test import reject_known_participants, validate_lock
from vigil_final.utils import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="finetune/experiments/vigil_final/reports/blind_test_lock.json")
    parser.add_argument("--export-aliases-json", required=True)
    parser.add_argument("--known-aliases-json", default="finetune/experiments/participant_cv/shared/participant_folds_5fold.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lock = read_json(args.lock)
    validate_lock(lock)
    export_aliases = set(read_json(args.export_aliases_json)["participant_aliases"])
    folds = read_json(args.known_aliases_json)
    known = {alias for fold in folds["folds"] for alias in fold["participant_aliases"]}
    reject_known_participants(export_aliases, known)
    write_json(args.output, {"status": "accepted_new_participants_only", "n_participants": len(export_aliases)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
