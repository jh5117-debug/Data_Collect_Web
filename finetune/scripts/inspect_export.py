#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from vigil_two_stage.export_parser import canonical_samples, load_export
from vigil_two_stage.utils import ensure_dir, sha256_file, write_json


DEFAULT_HARD_NEGATIVES = ["visual", "visuals", "visible", "digital", "individual", "vigilant", "video", "vital", "residual"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("--report-dir", default="finetune/reports")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    bundle = load_export(zip_path)
    samples, rejected = canonical_samples(bundle, DEFAULT_HARD_NEGATIVES)
    canonical_members = {s["canonical_audio_member"] for s in samples}
    audio_raw = [n for n in bundle.names if n.startswith(bundle.root + "audio_raw/") and not n.endswith("/")]
    raw_audio = [n for n in bundle.names if n.startswith(bundle.root + "raw_audio/") and not n.endswith("/")]
    by_group = [n for n in bundle.names if "/by_prompt_group/" in n and not n.endswith("/") and not n.endswith(".gitkeep")]
    report = {
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "root": bundle.root.rstrip("/"),
        "metadata_clip_rows": len(bundle.clips),
        "canonical_samples": len(samples),
        "rejected_or_inconsistent": len(rejected),
        "metadata_sessions": len(bundle.sessions),
        "metadata_accounts": len(bundle.accounts),
        "unique_clip_ids": len({c.get("clip_id") for c in bundle.clips}),
        "canonical_audio_raw_files": len(audio_raw),
        "raw_audio_view_files": len(raw_audio),
        "by_prompt_group_audio_files": len(by_group),
        "canonical_members_resolved": len(canonical_members),
        "prompt_group_counts": dict(Counter(s["prompt_group"] for s in samples)),
        "label_counts": dict(Counter(s["label"] for s in samples)),
    }
    report_dir = ensure_dir(args.report_dir)
    write_json(report_dir / "export_inspection.json", report)
    md = [
        "# Export Inspection",
        "",
        f"- ZIP SHA-256: `{report['zip_sha256']}`",
        f"- Metadata clip rows: {report['metadata_clip_rows']}",
        f"- Canonical samples: {report['canonical_samples']}",
        f"- Rejected/inconsistent rows: {report['rejected_or_inconsistent']}",
        f"- Canonical `audio_raw/` files: {report['canonical_audio_raw_files']}",
        f"- Duplicate `raw_audio/` view files: {report['raw_audio_view_files']}",
        f"- Duplicate `by_prompt_group/` view files: {report['by_prompt_group_audio_files']}",
        f"- Prompt groups: `{report['prompt_group_counts']}`",
        f"- Labels: `{report['label_counts']}`",
        "",
        "The parser uses `metadata/clips.jsonl` as the canonical table and resolves one `audio_raw/<clip_id>.*` source per clip.",
    ]
    (report_dir / "export_inspection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(report["canonical_samples"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
