#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from vigil_final.final_bundle import validate_public_bundle_manifest
from vigil_final.utils import write_json


def main() -> int:
    manifest = {
        "status": "blocked_until_research_choices_frozen",
        "include_qwen_weights": False,
        "model_kind": "final_deployment_candidate_not_scientific_test",
    }
    validate_public_bundle_manifest(manifest)
    out = Path("finetune/experiments/vigil_final/reports")
    write_json(out / "locked_final_model_status.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
