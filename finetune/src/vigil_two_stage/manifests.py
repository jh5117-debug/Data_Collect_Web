from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import normalize_product_casing, write_jsonl


def write_split_manifests(dataset_dir: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda r: (r.get("split", ""), r.get("clip_id", ""), r.get("window_index", 0)))
    write_jsonl(dataset_dir / "manifest_all.jsonl", ordered)
    for split in ("train", "val", "test"):
        split_rows = [r for r in ordered if r.get("split") == split]
        write_jsonl(dataset_dir / f"{split}.jsonl", split_rows)
        qwen_rows = [{"audio": str(r["full_wav_path"]), "text": qwen_asr_text(r["transcript"])} for r in split_rows]
        write_jsonl(dataset_dir / f"qwen_asr_{split}.jsonl", qwen_rows)


def qwen_asr_text(transcript: str) -> str:
    return "language English<asr_text>" + normalize_product_casing(transcript)
