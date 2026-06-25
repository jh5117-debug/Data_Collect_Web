#!/usr/bin/env python3
from __future__ import annotations

from vigil_final.utils import read_json, write_json


def main() -> int:
    diagnostic = read_json("finetune/experiments/vigil_final/reports/shared_qwen_diagnostic.json")
    write_json("finetune/experiments/vigil_final/reports/shared_qwen_prototype_validation.json", {"status": diagnostic["status"], "validated": False, "reason": diagnostic["blocker"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
