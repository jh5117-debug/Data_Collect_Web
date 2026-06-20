from vigil_two_stage.splits import assert_no_duplicate_audio_leakage, assert_no_speaker_leakage, assign_splits


def _rows(n=6):
    return [
        {
            "clip_id": f"C{i}",
            "participant_key": f"P{i}",
            "audio_sha256": f"H{i}",
            "label": i % 2,
        }
        for i in range(n)
    ]


def test_speaker_split_prevents_leakage_when_possible():
    rows, report = assign_splits(_rows(8))
    assert report["split_mode"] == "speaker_disjoint"
    assert assert_no_speaker_leakage(rows)


def test_duplicate_audio_hashes_cannot_cross_splits():
    rows = _rows(8)
    rows[0]["audio_sha256"] = "same"
    rows[7]["audio_sha256"] = "same"
    rows, _ = assign_splits(rows)
    assert assert_no_duplicate_audio_leakage(rows)
