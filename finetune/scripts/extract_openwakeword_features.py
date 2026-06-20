#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from vigil_two_stage.audio import extract_fft_features, read_wav
from vigil_two_stage.strict_runtime import official_openwakeword_namespace, require_strict_official_config
from vigil_two_stage.utils import ensure_dir, read_jsonl, sha256_file, stable_json, write_json, write_jsonl


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


class OfficialOpenWakeWordExtractor:
    def __init__(self) -> None:
        try:
            from openwakeword.utils import AudioFeatures
        except Exception as exc:
            raise RuntimeError(f"official openWakeWord AudioFeatures import failed: {exc}") from exc
        self.version = _package_version("openwakeword")
        self.extractor = AudioFeatures()
        self.model_checksum = "package_assets_unresolved"

    @staticmethod
    def _normalize_output(output: object) -> np.ndarray:
        if isinstance(output, (list, tuple)):
            if not output:
                raise RuntimeError("official openWakeWord returned an empty feature list")
            output = output[0]
        arr = np.asarray(output)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise RuntimeError(f"official openWakeWord returned unsupported feature shape {arr.shape}")
        arr = arr.astype(np.float32, copy=False)
        if not np.isfinite(arr).all():
            raise RuntimeError("official openWakeWord features contain NaN or Inf")
        return arr

    def extract(self, wav_path: Path) -> np.ndarray:
        sample_rate, audio = read_wav(wav_path)
        if sample_rate != 16000:
            raise RuntimeError(f"official openWakeWord expects 16 kHz WAV, got {sample_rate}")
        audio_i16 = np.clip(audio, -32768, 32767).astype(np.int16)
        attempts = []
        if hasattr(self.extractor, "embed_clips"):
            attempts.extend(
                [
                    lambda: self.extractor.embed_clips([audio_i16], batch_size=1),
                    lambda: self.extractor.embed_clips(audio_i16, batch_size=1),
                    lambda: self.extractor.embed_clips([audio_i16]),
                    lambda: self.extractor.embed_clips(audio_i16),
                ]
            )
        if callable(self.extractor):
            attempts.append(lambda: self.extractor(audio_i16))
        errors = []
        for attempt in attempts:
            try:
                return self._normalize_output(attempt())
            except Exception as exc:  # pragma: no cover - depends on installed openWakeWord version
                errors.append(f"{type(exc).__name__}: {exc}")
        raise RuntimeError("official openWakeWord feature extraction failed; attempts: " + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    run_dir = ensure_dir(Path(args.run_dir) / "stage1")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    strict_official = bool(config.get("stage1", {}).get("require_official_openwakeword", False))
    if strict_official:
        require_strict_official_config(config)
    rows = read_jsonl(dataset_dir / "manifest_all.jsonl")
    feature_dir = ensure_dir(run_dir / "features")
    openwakeword_available = importlib.util.find_spec("openwakeword") is not None
    allow_fallback = bool(config["stage1"].get("allow_acoustic_fallback_when_openwakeword_missing", False))
    if not openwakeword_available and not allow_fallback:
        write_json(run_dir / "feature_status.json", {"status": "blocked", "reason": "openwakeword_not_installed"})
        return 2
    official_extractor = OfficialOpenWakeWordExtractor() if openwakeword_available else None
    backend = "official_openwakeword" if official_extractor else "fallback_acoustic_fft_not_official_openwakeword"
    official_version = official_extractor.version if official_extractor else None
    official_checksum = official_extractor.model_checksum if official_extractor else None
    preprocessing_fingerprint = stable_json(
        {
            "sample_rate": config["audio"]["sample_rate"],
            "window_seconds": config["audio"]["window_seconds"],
            "backend": backend,
        }
    )
    feature_rows = []
    diagnostic: dict[str, object] = {}
    for row in rows:
        wav_path = Path(row["window_wav_path"])
        if official_extractor:
            namespace = official_openwakeword_namespace(
                official_extractor.version,
                official_extractor.model_checksum,
                preprocessing_fingerprint,
            )
        else:
            namespace = stable_json({"backend": backend, "feature_dim": 96, "preprocessing": preprocessing_fingerprint})
        key = sha256_file(wav_path) + "_" + namespace
        out_path = feature_dir / f"{row['clip_id']}_w{row['window_index']:02d}_{key[:12]}.npz"
        if not out_path.exists():
            if official_extractor:
                features = official_extractor.extract(wav_path)
            else:
                features = extract_fft_features(wav_path, feature_dim=96)
            np.savez_compressed(out_path, features=features.astype(np.float32))
        else:
            features = np.load(out_path)["features"]
        if official_extractor and not diagnostic and row["label"] in (0, 1):
            diagnostic = {
                "status": "ok",
                "backend": backend,
                "openwakeword_version": official_extractor.version,
                "feature_shape": list(np.asarray(features).shape),
                "feature_dim": int(np.asarray(features).shape[-1]),
                "finite": bool(np.isfinite(features).all()),
                "sample_clip_id": row["clip_id"],
            }
        feature_row = dict(row)
        feature_row.update(
            {
                "feature_path": str(out_path.resolve()),
                "feature_backend": backend,
                "feature_dim": int(np.asarray(features).shape[-1]),
                "openwakeword_version": official_version,
                "feature_namespace": namespace,
            }
        )
        feature_rows.append(feature_row)
    write_jsonl(run_dir / "features_manifest.jsonl", feature_rows)
    if official_extractor:
        write_json(Path(args.run_dir) / "openwakeword_feature_diagnostic.json", diagnostic)
        (Path(args.run_dir) / "openwakeword_feature_diagnostic.md").write_text(
            "# Official openWakeWord Feature Diagnostic\n\n"
            f"- Status: {diagnostic.get('status')}\n"
            f"- Version: `{official_version}`\n"
            f"- Feature shape: `{diagnostic.get('feature_shape')}`\n"
            f"- Feature dimension: `{diagnostic.get('feature_dim')}`\n"
            f"- Finite: {diagnostic.get('finite')}\n"
            f"- Model asset checksum: `{official_checksum}`\n",
            encoding="utf-8",
        )
    report = {
        "status": "ok",
        "feature_backend": backend,
        "official_openwakeword_available": openwakeword_available,
        "official_openwakeword_used": official_extractor is not None,
        "fallback_used": official_extractor is None,
        "feature_rows": len(feature_rows),
        "openwakeword_version": official_version,
        "model_checksum": official_checksum,
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
