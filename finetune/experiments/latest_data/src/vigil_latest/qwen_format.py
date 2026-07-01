from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vigil_two_stage.utils import normalize_product_casing

from .utils import ensure_dir, read_jsonl, write_json, write_jsonl


ASR_PREFIX = "language English<asr_text>"
LABEL_LEAK_TOKENS = ("positive", "negative", "trigger", "wake_intent", "prompt_group", "P1_", "P2_", "P3_", "P4_")


def qwen_asr_text(transcript: str, *, language: str = "English") -> str:
    prefix = f"language {language}<asr_text>"
    return prefix + normalize_product_casing(transcript).strip()


def has_label_leakage(text: str) -> bool:
    return any(token in text for token in LABEL_LEAK_TOKENS)


def asr_row(row: dict[str, Any]) -> dict[str, Any]:
    text = qwen_asr_text(str(row.get("transcript") or ""))
    if has_label_leakage(text):
        raise ValueError(f"Qwen ASR text contains label/control leakage for clip {row.get('clip_id')}: {text}")
    return {"audio": str(row["full_wav_path"]), "text": text}


def kws_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "audio": str(row["full_wav_path"]),
        "transcript": normalize_product_casing(str(row.get("transcript") or "")).strip(),
        "label": int(row["label"]),
        "prompt_group": row.get("prompt_group"),
        "phrase_id": row.get("phrase_id"),
        "participant_alias": row.get("participant_alias") or row.get("speaker_id"),
        "split": row.get("split"),
        "clip_id": row.get("clip_id"),
    }


def attach_privacy_aliases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    speakers = sorted({str(row.get("speaker_id") or "unknown") for row in rows})
    alias_map = {speaker: f"P{i + 1:03d}" for i, speaker in enumerate(speakers)}
    out = []
    for row in rows:
        item = dict(row)
        item["participant_alias"] = item.get("participant_alias") or alias_map[str(item.get("speaker_id") or "unknown")]
        out.append(item)
    return out


def validate_qwen_asr_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if sorted(row) != ["audio", "text"]:
            raise ValueError(f"Qwen ASR row has non-ASR fields: {sorted(row)}")
        text = str(row.get("text") or "")
        if not text.startswith("language English<asr_text>"):
            raise ValueError(f"Qwen ASR text missing official prefix: {text}")
        if has_label_leakage(text):
            raise ValueError(f"Qwen ASR text contains label/control leakage: {text}")


def build_qwen_and_kws_manifests(manifest_all: Path | str, output_root: Path | str) -> dict[str, Any]:
    rows = attach_privacy_aliases(read_jsonl(manifest_all))
    root = Path(output_root)
    qwen_dir = ensure_dir(root / "qwen_asr")
    kws_dir = ensure_dir(root / "keyword_spotting")
    summary: dict[str, Any] = {
        "source_manifest": str(manifest_all),
        "rows": len(rows),
        "qwen_asr_text_policy": "text is only language prefix plus transcript; labels/prompt metadata are excluded",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        qwen_rows = [asr_row(row) for row in split_rows]
        validate_qwen_asr_rows(qwen_rows)
        kws_rows = [kws_row(row) for row in split_rows]
        write_jsonl(qwen_dir / f"{split}.jsonl", qwen_rows)
        write_jsonl(kws_dir / f"kws_{split}.jsonl", kws_rows)
        summary["splits"][split] = {
            "rows": len(split_rows),
            "qwen_fields": ["audio", "text"],
            "kws_fields": ["audio", "transcript", "label", "prompt_group", "phrase_id", "participant_alias", "split", "clip_id"],
            "label_counts": dict(sorted(Counter(str(row["label"]) for row in split_rows).items())),
            "prompt_counts": dict(sorted(Counter(str(row.get("prompt_group")) for row in split_rows).items())),
        }
    write_json(root / "qwen_kws_manifest_summary.json", summary)
    lines = [
        "# Qwen ASR Format Report",
        "",
        "- Qwen ASR rows contain only `audio` and `text`.",
        "- `text` is `language English<asr_text>` plus the transcript.",
        "- Labels, prompt groups, phrase IDs, participant aliases, and splits are written only to KWS manifests.",
        "",
        "| Split | Rows |",
        "|---|---:|",
    ]
    for split, data in summary["splits"].items():
        lines.append(f"| {split} | {data['rows']} |")
    (root / "QWEN_ASR_FORMAT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
