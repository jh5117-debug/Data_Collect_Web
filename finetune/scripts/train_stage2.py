#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vigil_two_stage.utils import ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variant", choices=["bce", "bce_supcon"], default="bce")
    parser.add_argument("--allow-skip", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = ensure_dir(Path(args.run_dir) / ("stage2_" + args.variant))
    feature_manifest = Path(args.run_dir) / "stage2_qwen_features" / "qwen_features_manifest.jsonl"
    if not feature_manifest.exists():
        status = {
            "status": "skipped",
            "variant": args.variant,
            "reason": "Qwen encoder feature manifest is unavailable; verifier training was not run.",
            "model_name": config["stage2"]["model_name"],
            "qwen_parameters_modified": False,
        }
        write_json(out_dir / "metrics.json", status)
        write_json(out_dir / "frozen_qwen_integrity.json", {"status": "not_run", "qwen_parameters_modified": False, "reason": status["reason"]})
        (out_dir / "report.md").write_text(
            f"# Stage 2 {args.variant} Verifier\n\n"
            "Status: skipped.\n\n"
            f"Reason: {status['reason']}\n",
            encoding="utf-8",
        )
        return 0 if args.allow_skip else 2
    write_json(out_dir / "metrics.json", {"status": "not_implemented_for_current_host"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
