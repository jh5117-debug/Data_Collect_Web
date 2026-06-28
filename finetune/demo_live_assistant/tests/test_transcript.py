from __future__ import annotations

import pytest

from transcript import safe_extract_transcript


class FakeASRResult:
    def __init__(self, text: str) -> None:
        self.text = text


def test_transcript_extraction_rejects_object_repr() -> None:
    assert safe_extract_transcript([FakeASRResult("VIGIL")]) == "VIGIL"
    with pytest.raises(ValueError):
        safe_extract_transcript("ASRTranscription(language='English', text='VIGIL')")
