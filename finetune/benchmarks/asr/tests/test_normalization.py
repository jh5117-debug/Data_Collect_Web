from __future__ import annotations

from normalization import normalize_librispeech_text


def test_librispeech_normalization_examples() -> None:
    assert normalize_librispeech_text("Hello, World!") == "hello world"
    assert normalize_librispeech_text("IT\u2019S WORKING") == "it's working"
    assert normalize_librispeech_text("TWENTY-ONE") == "twenty one"
    assert normalize_librispeech_text("'HELLO'") == "hello"
    assert normalize_librispeech_text("rock'n'roll") == "rock'n'roll"
