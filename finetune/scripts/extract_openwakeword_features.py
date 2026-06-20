#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from vigil_two_stage.audio import extract_fft_features
from vigil_two_stage.utils import ensure_dir, read_jsonl, sha256_file, stable_json, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    run_dir = ensure_dir(Path(args.run_dir) / "stage1")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = read_jsonl(dataset_dir / "manifest_all.jsonl")
    feature_dir = ensure_dir(run_dir / "features")
    openwakeword_available = importlib.util.find_spec("openwakeword") is not None
    allow_fallback = bool(config["stage1"].get("allow_acoustic_fallback_when_openwakeword_missing", False))
    if not openwakeword_available and not allow_fallback:
        write_json(run_dir / "feature_status.json", {"status": "blocked", "reason": "openwakeword_not_installed"})
        return 2
    backend = "official_openwakeword" if openwakeword_available else "fallback_acoustic_fft_not_official_openwakeword"
    feature_rows = []
    for row in rows:
        wav_path = Path(row["window_wav_path"])
        key = sha256_file(wav_path) + "_" + stable_json({"backend": backend, "feature_dim": 96})
        out_path = feature_dir / f"{row['clip_id']}_w{row['window_index']:02d}_{key[:12]}.npz"
        if not out_path.exists():
            if openwakeword_available:
                # The official package interface varies across versions. Until it is
                # installed on this machine, keep extraction version-checked instead
                # of silently using the wrong internal API.
                raise RuntimeError("openwakeword is installed but no version-specific extractor has been configured")
            features = extract_fft_features(wav_path, feature_dim=96)
            np.savez_compressed(out_path, features=features.astype(np.float32))
        feature_row = dict(row)
        feature_row.update({"feature_path": str(out_path.resolve()), "feature_backend": backend, "feature_dim": 96})
        feature_rows.append(feature_row)
    write_jsonl(run_dir / "features_manifest.jsonl", feature_rows)
    report = {
        "status": "ok",
        "feature_backend": backend,
        "official_openwakeword_available": openwakeword_available,
        "official_openwakeword_used": openwakeword_available,
        "fallback_used": not openwakeword_available,
        "feature_rows": len(feature_rows),
        "note": "Fallback features are for engineering smoke only and are not a substitute for official openWakeWord features.",
    }
    write_json(run_dir / "feature_status.json", report)
    (run_dir / "report.md").write_text(
        "# Stage 1 Feature Extraction\n\n"
        f"- Backend: `{backend}`\n"
        f"- Rows: {len(feature_rows)}\n"
        f"- Official openWakeWord available: {openwakeword_available}\n\n"
        "If fallback is true, Stage 1 metrics are only an engineering pipeline smoke result.\n",
        encoding="utf-8",
    )
    print(run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
