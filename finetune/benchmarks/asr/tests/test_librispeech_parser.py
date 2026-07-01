from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from librispeech import build_manifest, parse_transcript_line, parse_utterance_id, write_manifest


def _write_fixture(root: Path, split: str = "test-clean", utt_id: str = "1234-5678-0001", text: str = "HELLO WORLD") -> Path:
    speaker, chapter = parse_utterance_id(utt_id)
    chapter_dir = root / split / speaker / chapter
    chapter_dir.mkdir(parents=True, exist_ok=True)
    audio = np.zeros(1600, dtype=np.float32)
    sf.write(chapter_dir / f"{utt_id}.flac", audio, 16000, format="FLAC")
    (chapter_dir / f"{speaker}-{chapter}.trans.txt").write_text(f"{utt_id} {text}\n", encoding="utf-8")
    return chapter_dir


def test_parse_utterance_id() -> None:
    assert parse_utterance_id("1234-5678-0001") == ("1234", "5678")
    with pytest.raises(ValueError):
        parse_utterance_id("bad-id")


def test_parse_transcript_line() -> None:
    assert parse_transcript_line("1234-5678-0001 HELLO WORLD") == ("1234-5678-0001", "HELLO WORLD")
    with pytest.raises(ValueError):
        parse_transcript_line("1234-5678-0001")


def test_build_manifest_and_deterministic_write(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    rows = build_manifest(tmp_path, "test-clean", validate_audio=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "1234-5678-0001"
    assert row["speaker_id"] == "1234"
    assert row["chapter_id"] == "5678"
    assert row["sample_rate"] == 16000
    assert row["channels"] == 1
    assert row["duration_sec"] > 0
    assert Path(row["audio_path"]).is_absolute()
    assert row["audio_sha256"]

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest(rows, first)
    write_manifest(rows, second)
    assert first.read_bytes() == second.read_bytes()


def test_duplicate_transcript_id_is_rejected(tmp_path: Path) -> None:
    chapter_dir = _write_fixture(tmp_path)
    trans = chapter_dir / "1234-5678.trans.txt"
    trans.write_text(
        "1234-5678-0001 HELLO WORLD\n1234-5678-0001 DUPLICATE\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_manifest(tmp_path, "test-clean", validate_audio=False)


def test_missing_audio_is_rejected(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "test-clean" / "1234" / "5678"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "1234-5678.trans.txt").write_text("1234-5678-0001 HELLO WORLD\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing audio"):
        build_manifest(tmp_path, "test-clean", validate_audio=False)
