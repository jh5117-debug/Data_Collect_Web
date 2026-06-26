from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from vigil_two_stage.export_parser import canonical_samples, load_export

from .utils import ensure_dir, sha256_file, write_json, write_jsonl


DEFAULT_HARD_NEGATIVES = ["visual", "visuals", "visible", "digital", "individual", "vigilant", "video", "vital", "residual"]


def inspect_export_zip(zip_path: Path | str, report_dir: Path | str) -> dict[str, Any]:
    zip_path = Path(zip_path)
    report_dir = ensure_dir(report_dir)
    with zipfile.ZipFile(zip_path) as zf:
        corrupt_member = zf.testzip()
    bundle = load_export(zip_path)
    samples, rejected = canonical_samples(bundle, DEFAULT_HARD_NEGATIVES)
    audio_raw = [n for n in bundle.names if n.startswith(bundle.root + "audio_raw/") and not n.endswith("/")]
    by_group = [n for n in bundle.names if "/by_prompt_group/" in n and not n.endswith("/") and not n.endswith(".gitkeep")]
    prompt_counts = Counter(s["prompt_group"] for s in samples)
    label_counts = Counter(str(s["label"]) for s in samples)
    result = {
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_file_size_bytes": zip_path.stat().st_size,
        "zip_integrity": "ok" if corrupt_member is None else "corrupt",
        "corrupt_member": corrupt_member,
        "root": bundle.root.rstrip("/"),
        "metadata_clip_rows": len(bundle.clips),
        "unique_clip_ids": len({c.get("clip_id") for c in bundle.clips}),
        "metadata_sessions": len(bundle.sessions),
        "metadata_accounts": len(bundle.accounts),
        "canonical_samples": len(samples),
        "rejected_or_inconsistent": len(rejected),
        "canonical_audio_raw_files": len(audio_raw),
        "by_prompt_group_audio_files": len(by_group),
        "prompt_group_counts": dict(sorted(prompt_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
    }
    write_json(report_dir / "latest_export_inspection.json", result)
    write_jsonl(report_dir / "rejected_or_inconsistent.jsonl", rejected)
    md = [
        "# Latest Export Inspection",
        "",
        f"- ZIP: `{zip_path}`",
        f"- SHA-256: `{result['zip_sha256']}`",
        f"- ZIP integrity: `{result['zip_integrity']}`",
        f"- Metadata clip rows: `{result['metadata_clip_rows']}`",
        f"- Canonical samples: `{result['canonical_samples']}`",
        f"- Rejected/inconsistent: `{result['rejected_or_inconsistent']}`",
        f"- Accounts: `{result['metadata_accounts']}`",
        f"- Sessions: `{result['metadata_sessions']}`",
        f"- Prompt counts: `{json.dumps(result['prompt_group_counts'], sort_keys=True)}`",
        f"- Label counts: `{json.dumps(result['label_counts'], sort_keys=True)}`",
    ]
    (report_dir / "LATEST_EXPORT_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return result


def assert_counts_close_to_admin(summary: dict[str, Any], inspection: dict[str, Any]) -> None:
    expected = int(summary["total_clips"])
    observed = int(inspection["canonical_samples"])
    if observed != expected:
        raise ValueError(f"latest export count mismatch: admin summary {expected}, export canonical samples {observed}")
