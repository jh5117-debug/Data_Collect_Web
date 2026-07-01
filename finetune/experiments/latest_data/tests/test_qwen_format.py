from __future__ import annotations

import json
from pathlib import Path

import pytest

from vigil_latest.qwen_format import build_qwen_and_kws_manifests, qwen_asr_text, validate_qwen_asr_rows


def test_qwen_asr_text_has_transcript_only_payload() -> None:
    assert qwen_asr_text("vigil") == "language English<asr_text>VIGIL"


def test_qwen_manifest_excludes_labels_and_kws_keeps_labels(tmp_path: Path) -> None:
    rows = [
        {
            "clip_id": "c1",
            "speaker_id": "spk_a",
            "split": "train",
            "full_wav_path": "/tmp/c1.wav",
            "transcript": "VIGIL",
            "label": 1,
            "prompt_group": "P1_vigil_only",
            "phrase_id": "vigil",
        },
        {
            "clip_id": "c2",
            "speaker_id": "spk_b",
            "split": "test",
            "full_wav_path": "/tmp/c2.wav",
            "transcript": "visual",
            "label": 0,
            "prompt_group": "P4_negative",
            "phrase_id": "visual",
        },
    ]
    manifest = tmp_path / "manifest_all.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    summary = build_qwen_and_kws_manifests(manifest, tmp_path / "out")
    qwen_train = [json.loads(line) for line in (tmp_path / "out/qwen_asr/train.jsonl").read_text().splitlines()]
    kws_train = [json.loads(line) for line in (tmp_path / "out/keyword_spotting/kws_train.jsonl").read_text().splitlines()]
    assert summary["splits"]["train"]["rows"] == 1
    assert sorted(qwen_train[0]) == ["audio", "text"]
    assert "label" not in qwen_train[0]
    assert kws_train[0]["label"] == 1
    assert kws_train[0]["participant_alias"] == "P001"


def test_qwen_manifest_validator_rejects_label_fields() -> None:
    with pytest.raises(ValueError, match="non-ASR fields"):
        validate_qwen_asr_rows([{"audio": "x.wav", "text": "language English<asr_text>VIGIL", "label": 1}])
