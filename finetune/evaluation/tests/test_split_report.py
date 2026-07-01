from __future__ import annotations

import json

from split_report import build_split_report


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def row(split, clip_id, speaker, session, audio_hash, label=1):
    return {
        "split": split,
        "clip_id": clip_id,
        "window_index": 0,
        "speaker_id": speaker,
        "session_id": session,
        "audio_sha256": audio_hash,
        "full_wav_sha256": f"wav-{audio_hash}",
        "label": label,
        "prompt_group": "P1_vigil_only" if label else "P4_negative",
        "phrase_id": "vigil" if label else "visual",
    }


def test_split_report_detects_leakage(tmp_path):
    train = [row("train", "C1", "spk_a", "S1", "H1")]
    val = [row("val", "C2", "spk_a", "S2", "H2")]
    test = [row("test", "C3", "spk_c", "S1", "H1", label=0)]
    write_jsonl(tmp_path / "train.jsonl", train)
    write_jsonl(tmp_path / "val.jsonl", val)
    write_jsonl(tmp_path / "test.jsonl", test)
    write_jsonl(tmp_path / "manifest_all.jsonl", train + val + test)

    report = build_split_report(tmp_path)
    assert report["validations"]["no_speaker_leakage"] is False
    assert report["validations"]["no_session_leakage"] is False
    assert report["validations"]["no_duplicate_audio_leakage"] is False


def test_split_report_counts_unique_clips_and_windows(tmp_path):
    train = [row("train", "C1", "spk_a", "S1", "H1"), {**row("train", "C1", "spk_a", "S1", "H1"), "window_index": 1}]
    val = [row("val", "C2", "spk_b", "S2", "H2")]
    test = [row("test", "C3", "spk_c", "S3", "H3", label=0)]
    write_jsonl(tmp_path / "train.jsonl", train)
    write_jsonl(tmp_path / "val.jsonl", val)
    write_jsonl(tmp_path / "test.jsonl", test)
    write_jsonl(tmp_path / "manifest_all.jsonl", train + val + test)

    report = build_split_report(tmp_path)
    assert report["splits"]["train"]["windows"] == 2
    assert report["splits"]["train"]["unique_original_clips"] == 1
    assert report["splits"]["test"]["P4"] == 1

