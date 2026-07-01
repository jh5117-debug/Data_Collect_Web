#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from librispeech import EXPECTED_COUNTS, build_manifest, manifest_summary, smoke_subset, write_manifest
from utils import sha256_file, write_json


URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}


def _archive_report(download_dir: Path) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for split in ("test-clean", "test-other"):
        archive = download_dir / f"{split}.tar.gz"
        report[split] = {
            "url": URLS[split],
            "archive_path": str(archive.resolve()),
            "exists": archive.exists(),
            "size_bytes": archive.stat().st_size if archive.exists() else None,
            "sha256": sha256_file(archive) if archive.exists() else None,
        }
    return report


def _write_markdown_report(report: dict, path: Path) -> None:
    lines = [
        "# LibriSpeech Download And Manifest Report",
        "",
        f"Created: {report['created_at_utc']}",
        "",
        "| Split | Expected | Parsed | FLAC files | Speakers | Duration hours | Archive SHA-256 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split in ("test-clean", "test-other"):
        item = report["splits"][split]
        archive = report["archives"][split]
        lines.append(
            "| {split} | {expected} | {parsed} | {flacs} | {speakers} | {hours:.3f} | {sha} |".format(
                split=split,
                expected=item["expected_utterances"],
                parsed=item["manifest"]["utterances"],
                flacs=item["flac_files"],
                speakers=item["manifest"]["speakers"],
                hours=float(item["manifest"]["duration_hours"]),
                sha=archive.get("sha256") or "missing",
            )
        )
    lines.extend(
        [
            "",
            f"Total utterances: {report['total_utterances']}",
            f"Smoke utterances: {report['smoke_utterances']}",
            "",
            "The smoke subset is deterministic and is only for pipeline validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare deterministic LibriSpeech JSONL manifests.")
    parser.add_argument("--root", type=Path, default=SCRIPT_ROOT / "data" / "LibriSpeech")
    parser.add_argument("--manifest-dir", type=Path, default=SCRIPT_ROOT / "manifests")
    parser.add_argument("--reports-dir", type=Path, default=SCRIPT_ROOT / "reports")
    parser.add_argument("--download-dir", type=Path, default=SCRIPT_ROOT / "downloads")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--smoke-per-split", type=int, default=32)
    parser.add_argument("--validate-audio", action="store_true", default=False)
    parser.add_argument("--expected-counts", action="store_true", default=False)
    args = parser.parse_args()

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict]] = {}
    report: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()),
        "archives": _archive_report(args.download_dir),
        "splits": {},
        "seed": args.seed,
        "smoke_per_split": args.smoke_per_split,
    }

    for split in ("test-clean", "test-other"):
        rows = build_manifest(
            args.root,
            split,
            validate_audio=args.validate_audio,
            expected_counts=args.expected_counts,
        )
        split_rows[split] = rows
        out_name = split.replace("-", "_") + ".jsonl"
        write_manifest(rows, args.manifest_dir / out_name)
        smoke_rows = smoke_subset(rows, count=args.smoke_per_split, seed=args.seed)
        write_manifest(smoke_rows, args.manifest_dir / f"smoke_{out_name}")
        split_dir = args.root / split
        report["splits"][split] = {
            "expected_utterances": EXPECTED_COUNTS[split],
            "manifest_path": str((args.manifest_dir / out_name).resolve()),
            "smoke_manifest_path": str((args.manifest_dir / f"smoke_{out_name}").resolve()),
            "flac_files": len(list(split_dir.glob("*/*/*.flac"))),
            "transcript_files": len(list(split_dir.glob("*/*/*.trans.txt"))),
            "manifest": manifest_summary(rows),
            "smoke": manifest_summary(smoke_rows),
        }

    all_rows = sorted(split_rows["test-clean"] + split_rows["test-other"], key=lambda row: (row["split"], row["id"]))
    smoke_rows = sorted(
        smoke_subset(split_rows["test-clean"], count=args.smoke_per_split, seed=args.seed)
        + smoke_subset(split_rows["test-other"], count=args.smoke_per_split, seed=args.seed),
        key=lambda row: (row["split"], row["id"]),
    )
    write_manifest(all_rows, args.manifest_dir / "all_test.jsonl")
    write_manifest(smoke_rows, args.manifest_dir / "smoke_all.jsonl")

    report["total_utterances"] = len(all_rows)
    report["smoke_utterances"] = len(smoke_rows)
    report["all_manifest_path"] = str((args.manifest_dir / "all_test.jsonl").resolve())
    report["smoke_manifest_path"] = str((args.manifest_dir / "smoke_all.jsonl").resolve())

    write_json(args.reports_dir / "dataset_download_report.json", report)
    _write_markdown_report(report, args.reports_dir / "dataset_download_report.md")
    print(json.dumps({"total_utterances": len(all_rows), "smoke_utterances": len(smoke_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
