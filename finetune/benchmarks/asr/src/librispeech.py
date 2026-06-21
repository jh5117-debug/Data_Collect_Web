from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf

from utils import sha256_file, write_jsonl

EXPECTED_COUNTS = {"test-clean": 2620, "test-other": 2939}


def manifest_name(split: str) -> str:
    return split.replace("-", "_") + ".jsonl"


def parse_utterance_id(utterance_id: str) -> tuple[str, str]:
    parts = utterance_id.split("-")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"invalid LibriSpeech utterance id: {utterance_id}")
    return parts[0], parts[1]


def parse_transcript_line(line: str) -> tuple[str, str]:
    line = line.strip()
    if not line:
        raise ValueError("empty transcript line")
    if " " not in line:
        raise ValueError(f"transcript line has no text: {line}")
    utt_id, text = line.split(" ", 1)
    if not text.strip():
        raise ValueError(f"empty reference for {utt_id}")
    parse_utterance_id(utt_id)
    return utt_id, text.strip()


def iter_transcript_entries(split_dir: Path) -> list[tuple[str, str, Path]]:
    entries: list[tuple[str, str, Path]] = []
    for trans_path in sorted(split_dir.glob("*/*/*.trans.txt")):
        for line in trans_path.read_text(encoding="utf-8").splitlines():
            utt_id, text = parse_transcript_line(line)
            entries.append((utt_id, text, trans_path))
    return entries


def _audio_info(path: Path) -> tuple[int, int, float]:
    info = sf.info(str(path))
    return int(info.samplerate), int(info.channels), float(info.frames / info.samplerate)


def build_manifest(
    librispeech_root: Path,
    split: str,
    *,
    validate_audio: bool = True,
    expected_counts: bool = False,
) -> list[dict[str, Any]]:
    split_dir = librispeech_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"missing LibriSpeech split directory: {split_dir}")

    entries = iter_transcript_entries(split_dir)
    ids = [entry[0] for entry in entries]
    duplicated = sorted([utt for utt, count in Counter(ids).items() if count > 1])
    if duplicated:
        raise ValueError(f"duplicate LibriSpeech utterance ids: {duplicated[:10]}")
    by_id = {utt_id: (text, trans_path) for utt_id, text, trans_path in entries}

    flacs = sorted(split_dir.glob("*/*/*.flac"))
    audio_ids = {path.stem for path in flacs}
    missing_audio = sorted(set(by_id) - audio_ids)
    missing_transcript = sorted(audio_ids - set(by_id))
    if missing_audio:
        raise FileNotFoundError(f"{split}: transcript entries missing audio: {missing_audio[:10]}")
    if missing_transcript:
        raise ValueError(f"{split}: audio files missing transcripts: {missing_transcript[:10]}")
    if expected_counts and split in EXPECTED_COUNTS and len(by_id) != EXPECTED_COUNTS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_COUNTS[split]} utterances, found {len(by_id)}")

    rows: list[dict[str, Any]] = []
    for audio_path in flacs:
        utt_id = audio_path.stem
        speaker_id, chapter_id = parse_utterance_id(utt_id)
        reference, _trans_path = by_id[utt_id]
        sample_rate: int | None = None
        channels: int | None = None
        duration_sec: float | None = None
        if validate_audio:
            sample_rate, channels, duration_sec = _audio_info(audio_path)
            if sample_rate != 16000:
                raise ValueError(f"{audio_path} sample rate must be 16000, got {sample_rate}")
            if channels != 1:
                raise ValueError(f"{audio_path} must be mono, got {channels} channels")
            if duration_sec <= 0:
                raise ValueError(f"{audio_path} has non-positive duration")
        rows.append(
            {
                "id": utt_id,
                "split": split,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "audio_path": str(audio_path.resolve()),
                "reference": reference,
                "sample_rate": sample_rate or 16000,
                "channels": channels or 1,
                "duration_sec": duration_sec,
                "audio_sha256": sha256_file(audio_path) if validate_audio else None,
            }
        )
    rows.sort(key=lambda row: row["id"])
    return rows


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    write_jsonl(path, rows)


def smoke_subset(rows: list[dict[str, Any]], count: int = 32, seed: int = 20260620) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) <= count:
        return sorted(rows, key=lambda row: row["id"])
    rng = random.Random(seed)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    selected: list[dict[str, Any]] = []
    # First round: one per speaker, ordered by deterministic duration buckets.
    for speaker in speakers:
        choices = sorted(by_speaker[speaker], key=lambda row: (float(row.get("duration_sec") or 0.0), row["id"]))
        selected.append(choices[len(choices) // 2])
        if len(selected) == count:
            return sorted(selected, key=lambda row: row["id"])
    remaining = [row for row in rows if row["id"] not in {r["id"] for r in selected}]
    remaining.sort(key=lambda row: row["id"])
    # Spread over short/medium/long bins.
    remaining.sort(key=lambda row: float(row.get("duration_sec") or 0.0))
    bins = [remaining[i::3] for i in range(3)]
    for bin_rows in bins:
        rng.shuffle(bin_rows)
    cursor = 0
    while len(selected) < count and any(bins):
        bin_rows = bins[cursor % len(bins)]
        if bin_rows:
            selected.append(bin_rows.pop())
        cursor += 1
    return sorted(selected[:count], key=lambda row: row["id"])


def manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    duration = [float(row.get("duration_sec") or 0.0) for row in rows]
    return {
        "utterances": len(rows),
        "speakers": len({row["speaker_id"] for row in rows}),
        "chapters": len({(row["speaker_id"], row["chapter_id"]) for row in rows}),
        "duration_hours": sum(duration) / 3600.0,
        "min_duration_sec": min(duration) if duration else None,
        "max_duration_sec": max(duration) if duration else None,
    }
