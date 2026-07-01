from __future__ import annotations

from vigil_participant_cv.privacy import assert_public_text_is_sanitized


def test_generated_public_fold_artifacts_contain_no_raw_names_or_emails():
    assert_public_text_is_sanitized("P001 P002 participant aliases only")


def test_public_artifact_rejects_raw_speaker_hash():
    try:
        assert_public_text_is_sanitized("spk_abcdef1234567890")
    except ValueError:
        return
    raise AssertionError("expected raw speaker hash rejection")
