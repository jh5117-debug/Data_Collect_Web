from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HARD_NEGATIVE_PHRASES = [
    "go",
    "go go",
    "joe",
    "joke",
    "yo",
    "yo yo",
    "hey yo",
    "hello",
    "no",
    "visual",
    "visible",
    "digital",
    "individual",
    "vigilant",
]


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspection_markdown(inspection: dict[str, Any]) -> str:
    lines = [
        "# ROS Bag Inspection Report",
        "",
        f"- Zip path: `{inspection.get('zip_path')}`",
        f"- Zip size bytes: `{inspection.get('zip_size_bytes')}`",
        f"- Bag count: `{inspection.get('bag_count')}`",
        "",
        "## Bags",
        "",
    ]
    for bag in inspection.get("bags", []):
        meta = bag["metadata"]
        lines.extend(
            [
                f"### {bag['source_bag']}",
                "",
                f"- Expected label: `{bag['expected_label']}`",
                f"- Duration seconds: `{meta['duration_sec']:.6f}`",
                f"- Metadata message count: `{meta['message_count']}`",
                f"- Relative DB files: `{', '.join(meta.get('relative_file_paths') or [])}`",
                "",
                "| Topic | Type | Metadata messages | SQLite messages |",
                "|---|---|---:|---:|",
            ]
        )
        sqlite_counts: dict[str, int] = {}
        if bag.get("db3_files"):
            first = bag["db3_files"][0]
            topic_by_id = {str(row["id"]): row for row in first.get("topics_table", [])}
            for topic_id, count in (first.get("messages_by_topic_id") or {}).items():
                topic = topic_by_id.get(str(topic_id), {})
                sqlite_counts[str(topic.get("name"))] = int(count)
        for topic in meta.get("topics", []):
            lines.append(
                f"| `{topic['name']}` | `{topic['type']}` | {topic['message_count']} | {sqlite_counts.get(str(topic['name']), 0)} |"
            )
        lines.extend(["", "SQLite DB3 files:", ""])
        for db in bag.get("db3_files", []):
            lines.append(f"- `{Path(db['path']).name}`: {db['size_bytes']} bytes, messages rows `{db['messages_table_rows']}`")
        lines.append("")
    lines.extend(
        [
            "## Binary Safety",
            "",
            "This report inspects metadata and SQLite table counts only. It does not dump raw CDR message blobs.",
        ]
    )
    return "\n".join(lines)


def audio_status_markdown(result: dict[str, Any]) -> str:
    missing = result.get("ros2_decode_status", {}).get("missing", [])
    lines = [
        "# Audio Extraction Status",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Reason: {result.get('reason')}",
        f"- Manifest created: `{result.get('manifest_created')}`",
        f"- Manifest path: `{result.get('manifest_path')}`",
        "",
        "## ROS 2 Decode Availability",
        "",
    ]
    if missing:
        for item in missing:
            lines.append(f"- `{item['module']}`: {item['error']}")
    else:
        lines.append("- ROS 2 modules importable.")
    lines.extend(["", "## Exact Resume Command", "", "```bash", str(result.get("exact_resume_command")), "```"])
    return "\n".join(lines)


def score_audit_markdown(result: dict[str, Any]) -> str:
    diagnosis = result.get("diagnosis", {})
    score_rows = result.get("score_rows") or []
    lines = [
        "# False-Trigger Score Audit",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Reason: {result.get('reason')}",
        f"- Diagnosis: `{diagnosis.get('diagnosis')}`",
        f"- Scored rows: `{diagnosis.get('rows')}`",
        "",
        "## Required Decision Logic",
        "",
        "`final_trigger` must equal `stage1_accept AND stage2_accept`.",
        "",
        "## Current Result",
        "",
    ]
    if result.get("status") == "blocked":
        lines.append("No score audit was run because no decoded WAV manifest is available yet.")
    else:
        lines.append("Score rows were available and audited.")
        lines.extend(
            [
                "",
                "| Case | Expected | Transcript hint | Stage 1 | Stage 2 | Final trigger | Feature hash | Embedding hash |",
                "|---|---:|---|---:|---:|---|---|---|",
            ]
        )
        for row in score_rows:
            lines.append(
                "| `{case}` | {label} | `{hint}` | {s1:.6f} | {s2:.6f} | `{final}` | `{feature}` | `{embedding}` |".format(
                    case=row.get("case_id"),
                    label=row.get("expected_label"),
                    hint=row.get("transcript_hint"),
                    s1=float(row.get("stage1_score") or 0.0),
                    s2=float(row.get("stage2_score") or 0.0),
                    final=row.get("final_trigger"),
                    feature=row.get("feature_hash"),
                    embedding=row.get("embedding_hash"),
                )
            )
    lines.extend(
        [
            "",
            "## Constant Score Checks",
            "",
            f"- False accepts: `{diagnosis.get('false_accepts')}`",
            f"- False rejects: `{diagnosis.get('false_rejects')}`",
            f"- Stage 2 negative accepts: `{diagnosis.get('stage2_negative_accepts')}`",
            f"- Stage 2 constant check: `{diagnosis.get('stage2_score_constant_check')}`",
            f"- Feature hash check: `{diagnosis.get('feature_hash_check')}`",
            f"- Embedding hash check: `{diagnosis.get('embedding_hash_check')}`",
        ]
    )
    return "\n".join(lines)


def spectrogram_markdown(audio_status: dict[str, Any], spectrogram_status: dict[str, Any] | None = None) -> str:
    spectrogram_status = spectrogram_status or {}
    if spectrogram_status:
        lines = [
            "# Spectrogram Diagnostic",
            "",
            f"- Status: `{spectrogram_status.get('status')}`",
            f"- Reason: {spectrogram_status.get('reason')}",
            "",
        ]
        for item in spectrogram_status.get("spectrograms", []):
            lines.append(f"- `{item.get('case_id')}`: `{item.get('path')}`")
        lines.extend(
            [
                "",
                "Generated PNGs are written under the ignored `runs/` directory and are not committed.",
            ]
        )
        return "\n".join(lines)
    return "\n".join(
        [
            "# Spectrogram Diagnostic",
            "",
            f"- Status: `{'blocked' if audio_status.get('status') == 'blocked' else 'pending'}`",
            "",
            "Spectrogram generation depends on successful ROS 2 audio extraction from the bags.",
            "No generated PNGs are committed by default. Any local spectrograms should be written under the ignored `runs/` directory.",
        ]
    )


def hard_negative_plan_markdown() -> str:
    phrases = "\n".join(f"- {phrase}" for phrase in HARD_NEGATIVE_PHRASES)
    return f"""# Hard-Negative Retraining Plan

## Rules

- Do not retrain Qwen.
- Qwen ASR transcript target remains exact transcript text.
- KWS label for go / joe / joke / yo is `0`.
- Keep Shaw/Andy false-positive rosbags as held-out regression cases, not training data.
- Collect more speakers for these hard negatives before retraining.
- If model bias is confirmed, retrain or recalibrate the Stage 2 verifier head first.
- Stage 1 should remain high-recall; tune Stage 1 only if candidate rate is too high.

## Current Audit Evidence

- `Joe.` is a final false accept in Shaw's held-out false-positive bag.
- `Go.` and `Joke.` are rejected by the current cascade because Stage 1 rejects them.
- Stage 2 accepts all three negative windows in this tiny held-out set.
- Feature and Stage 2 embedding hashes differ across `Go.`, `Joe.`, `Joke.`, and `VIGIL.`, so this audit does not support stale cached features or identical audio windows.
- This does not mean retraining is complete; it identifies the next hard-negative regression target.

## Prompt 4 Hard Negatives To Add

{phrases}

## Regression Policy

The current rosbag examples should be replayed after any integration change, threshold change, Stage 2 retraining, or hard-negative data expansion. They should not be used to tune thresholds in this task.
"""
