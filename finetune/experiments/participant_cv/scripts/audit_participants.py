#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from vigil_participant_cv.participant_stats import (
    attach_aliases,
    dedupe_clips,
    duplicate_audio_groups,
    load_manifest_rows,
    participant_statistics,
    sanitized_clip_rows,
)
from vigil_participant_cv.privacy import assert_public_text_is_sanitized
from vigil_participant_cv.utils import ensure_dir, write_csv, write_json


def bar_plot(path: Path, labels: list[str], series: dict[str, list[int]], title: str, ylabel: str) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = [0] * len(labels)
    for name, values in series.items():
        ax.bar(labels, values, bottom=bottom, label=name)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=70)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", default="finetune/experiments/participant_cv")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    reports = ensure_dir(out_dir / "reports")
    plots = ensure_dir(reports / "plots")
    shared = ensure_dir(out_dir / "shared")
    private = ensure_dir(shared / "private")

    manifest_rows = load_manifest_rows(args.dataset_dir)
    clips, alias_map = attach_aliases(dedupe_clips(manifest_rows))
    stats = participant_statistics(clips)
    duplicates = duplicate_audio_groups(clips)
    public_rows = sanitized_clip_rows(clips)

    write_json(private / "participant_alias_map.private.json", alias_map)
    write_csv(private / "participant_statistics.private.csv", stats)
    write_csv(shared / "participant_distribution_before_cap.csv", public_rows)

    duplicate_json = {
        "duplicate_group_count": len(duplicates),
        "cross_participant_duplicate_group_count": sum(1 for group in duplicates if group["cross_participant"]),
        "groups": duplicates,
        "policy": "cross-participant exact duplicates are excluded from formal evaluation; within-participant duplicates keep one canonical clip",
    }
    write_json(reports / "duplicate_audio_audit.json", duplicate_json)

    labels = [row["participant_alias"] for row in stats]
    bar_plot(plots / "clips_per_participant_before_cap.png", labels, {"clips": [row["total_unique_clips"] for row in stats]}, "Clips per participant before cap", "clips")
    bar_plot(
        plots / "positive_negative_counts_per_participant.png",
        labels,
        {"positive": [row["positive_clips"] for row in stats], "negative": [row["negative_clips"] for row in stats]},
        "Positive/negative clips per participant",
        "clips",
    )
    bar_plot(
        plots / "prompt_group_counts_per_participant.png",
        labels,
        {
            "P1": [row["P1_vigil_only"] for row in stats],
            "P2": [row["P2_phrase_plus_vigil"] for row in stats],
            "P3": [row["P3_vigil_plus_phrase"] for row in stats],
            "P4": [row["P4_negative"] for row in stats],
        },
        "Prompt groups per participant",
        "clips",
    )
    bar_plot(
        plots / "few_shot_eligibility.png",
        labels,
        {
            "3-shot eligible": [1 if row["eligible_3_shot"] else 0 for row in stats],
            "5-shot eligible": [1 if row["eligible_5_shot"] else 0 for row in stats],
        },
        "Few-shot eligibility",
        "eligible",
    )

    prompt_totals = Counter(row["prompt_group"] for row in public_rows)
    report = [
        "# Participant Data Audit",
        "",
        "Privacy: all public rows use aliases `P001` ... and no raw speaker hashes.",
        "",
        f"- Manifest windows: `{len(manifest_rows)}`",
        f"- Unique clips: `{len(clips)}`",
        f"- Participants: `{len(stats)}`",
        f"- Positive clips: `{sum(row['positive_clips'] for row in stats)}`",
        f"- Negative clips: `{sum(row['negative_clips'] for row in stats)}`",
        f"- P1/P2/P3/P4: `{dict(sorted(prompt_totals.items()))}`",
        f"- 3-shot eligible participants: `{sum(1 for row in stats if row['eligible_3_shot'])}`",
        f"- 5-shot eligible participants: `{sum(1 for row in stats if row['eligible_5_shot'])}`",
        f"- Duplicate audio groups: `{len(duplicates)}`",
        f"- Cross-participant duplicate groups: `{duplicate_json['cross_participant_duplicate_group_count']}`",
        "",
        "## Participant Summary",
        "",
        "| Alias | Clips | Pos | Neg | P1 | P2 | P3 | P4 | 3-shot | 5-shot |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in stats:
        report.append(
            f"| {row['participant_alias']} | {row['total_unique_clips']} | {row['positive_clips']} | {row['negative_clips']} | "
            f"{row['P1_vigil_only']} | {row['P2_phrase_plus_vigil']} | {row['P3_vigil_plus_phrase']} | {row['P4_negative']} | "
            f"{row['eligible_3_shot']} | {row['eligible_5_shot']} |"
        )
    report_text = "\n".join(report) + "\n"
    assert_public_text_is_sanitized(report_text)
    (reports / "PARTICIPANT_DATA_AUDIT.md").write_text(report_text, encoding="utf-8")

    duplicate_md = [
        "# Duplicate Audio Audit",
        "",
        f"- Duplicate groups: `{len(duplicates)}`",
        f"- Cross-participant duplicate groups: `{duplicate_json['cross_participant_duplicate_group_count']}`",
        "- Policy: exclude cross-participant exact duplicates from formal evaluation; keep one canonical within-participant duplicate.",
    ]
    if duplicates:
        duplicate_md.extend(["", "| Audio hash prefix | Clips | Participants | Cross participant |", "|---|---:|---:|---|"])
        for group in duplicates:
            duplicate_md.append(
                f"| {group['audio_hash'][:12]} | {group['clip_count']} | {group['participant_count']} | {group['cross_participant']} |"
            )
    duplicate_text = "\n".join(duplicate_md) + "\n"
    assert_public_text_is_sanitized(duplicate_text)
    (reports / "DUPLICATE_AUDIO_AUDIT.md").write_text(duplicate_text, encoding="utf-8")
    print(json.dumps({"clips": len(clips), "participants": len(stats), "duplicates": len(duplicates)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
