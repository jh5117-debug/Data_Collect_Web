from __future__ import annotations

import json
import zipfile
from pathlib import Path

from vigil_two_stage.export_parser import canonical_samples, load_export


def _write_min_zip(path: Path, clips: list[dict]):
    root = "export/"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(root + "metadata/clips.jsonl", "\n".join(json.dumps(c) for c in clips) + "\n")
        z.writestr(root + "metadata/sessions.jsonl", json.dumps({"session_id": "S1"}) + "\n")
        z.writestr(root + "metadata/accounts.csv", "account_id,email\nA1,redacted@example.com\n")
        for clip in clips:
            cid = clip["clip_id"]
            z.writestr(root + f"audio_raw/{cid}.webm", b"fake")
            z.writestr(root + f"raw_audio/P1/S1/{cid}.webm", b"fake")
            z.writestr(root + f"by_prompt_group/{clip['prompt_group']}/raw_audio/{cid}.webm", b"fake")


def test_export_parser_selects_one_canonical_sample_per_clip_id(tmp_path):
    zp = tmp_path / "x.zip"
    _write_min_zip(
        zp,
        [
            {
                "clip_id": "C1",
                "participant_id": "P1",
                "session_id": "S1",
                "prompt_group": "P1_vigil_only",
                "transcript": "Vigil",
                "contains_vigil": True,
                "wake_intent": True,
                "is_negative": False,
            }
        ],
    )
    samples, rejected = canonical_samples(load_export(zp), ["visual"])
    assert len(samples) == 1
    assert not rejected
    assert samples[0]["canonical_audio_member"].endswith("audio_raw/C1.webm")


def test_duplicate_directory_views_are_not_ingested_as_new_samples(tmp_path):
    zp = tmp_path / "x.zip"
    _write_min_zip(
        zp,
        [
            {
                "clip_id": "C1",
                "participant_id": "P1",
                "session_id": "S1",
                "prompt_group": "P4_negative",
                "transcript": "visual",
                "contains_vigil": False,
                "wake_intent": False,
                "is_negative": True,
            }
        ],
    )
    bundle = load_export(zp)
    assert len([n for n in bundle.names if n.endswith("C1.webm")]) == 3
    samples, _ = canonical_samples(bundle, ["visual"])
    assert len(samples) == 1


def test_positive_and_negative_label_mapping(tmp_path):
    zp = tmp_path / "x.zip"
    _write_min_zip(
        zp,
        [
            {
                "clip_id": "C1",
                "participant_id": "P1",
                "session_id": "S1",
                "prompt_group": "P2_phrase_plus_vigil",
                "transcript": "Hi Vigil.",
                "contains_vigil": True,
                "wake_intent": True,
                "is_negative": False,
            },
            {
                "clip_id": "C2",
                "participant_id": "P2",
                "session_id": "S2",
                "prompt_group": "P4_negative",
                "transcript": "vigilant",
                "contains_vigil": False,
                "wake_intent": False,
                "is_negative": True,
            },
        ],
    )
    samples, rejected = canonical_samples(load_export(zp), ["vigilant"])
    assert not rejected
    assert {s["clip_id"]: s["label"] for s in samples} == {"C1": 1, "C2": 0}
    assert {s["clip_id"]: s["phrase_id"] for s in samples}["C2"] == "vigilant"


def test_inconsistent_metadata_is_quarantined(tmp_path):
    zp = tmp_path / "x.zip"
    _write_min_zip(
        zp,
        [
            {
                "clip_id": "C1",
                "participant_id": "P1",
                "session_id": "S1",
                "prompt_group": "P4_negative",
                "transcript": "Vigil next",
                "contains_vigil": False,
                "wake_intent": False,
                "is_negative": True,
            }
        ],
    )
    samples, rejected = canonical_samples(load_export(zp), ["visual"])
    assert not samples
    assert rejected
    assert "negative_prompt_contains_exact_vigil" in rejected[0]["reasons"]
