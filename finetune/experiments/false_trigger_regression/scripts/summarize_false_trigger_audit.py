#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from report import hard_negative_plan_markdown, spectrogram_markdown, write_json, write_text


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _generate_spectrograms(audio_status: dict) -> dict:
    manifest_path = Path(str(audio_status.get("manifest_path") or ""))
    if audio_status.get("status") != "ok" or not manifest_path.exists():
        return {
            "status": "blocked",
            "reason": "No decoded WAV manifest is available.",
            "spectrograms": [],
        }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"matplotlib/numpy unavailable: {type(exc).__name__}: {exc}",
            "spectrograms": [],
        }
    rows = _read_manifest(manifest_path)
    out_dir = manifest_path.parent / "spectrograms"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for row in rows:
        wav_path = Path(row["wav_path"])
        with wave.open(str(wav_path), "rb") as handle:
            sample_rate = handle.getframerate()
            pcm = handle.readframes(handle.getnframes())
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.specgram(audio, NFFT=512, Fs=sample_rate, noverlap=384, cmap="magma")
        ax.set_title(f"{row['case_id']} ({row.get('transcript_hint', '')})")
        ax.set_xlabel("seconds")
        ax.set_ylabel("Hz")
        fig.tight_layout()
        out_path = out_dir / f"{row['case_id']}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        outputs.append({"case_id": row["case_id"], "path": str(out_path)})
    return {
        "status": "ok",
        "reason": "Generated local spectrogram PNGs under ignored runs directory.",
        "manifest_path": str(manifest_path),
        "spectrograms": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=Path("finetune/experiments/false_trigger_regression/reports"))
    args = parser.parse_args()
    audio_status = read_json(args.reports_dir / "audio_extraction_status.json")
    spectrogram_status = _generate_spectrograms(audio_status)
    write_json(args.reports_dir / "spectrogram_status.json", spectrogram_status)
    write_text(args.reports_dir / "SPECTROGRAM_DIAGNOSTIC.md", spectrogram_markdown(audio_status, spectrogram_status))
    write_text(args.reports_dir / "HARD_NEGATIVE_RETRAINING_PLAN.md", hard_negative_plan_markdown())
    print(f"wrote summary reports under {args.reports_dir}")


if __name__ == "__main__":
    main()
