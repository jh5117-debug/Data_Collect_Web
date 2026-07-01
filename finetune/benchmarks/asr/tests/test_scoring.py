from __future__ import annotations

from scoring import score_pairs


def test_corpus_wer_uses_global_counts() -> None:
    metrics = score_pairs(["hello world", "twenty one"], ["hello", "twenty two"])
    assert metrics["substitutions"] == 1
    assert metrics["deletions"] == 1
    assert metrics["insertions"] == 0
    assert metrics["reference_words"] == 4
    assert metrics["wer"] == 0.5
    assert metrics["sentence_error_rate"] == 1.0
    assert metrics["exact_match_rate"] == 0.0


def test_insertions_are_counted() -> None:
    metrics = score_pairs(["hello"], ["well hello there"])
    assert metrics["insertions"] == 2
    assert metrics["wer"] == 2.0


def test_cer_accumulates_per_utterance() -> None:
    metrics = score_pairs(["abc", "de"], ["axc", "d"])
    assert metrics["cer"] == 2 / 5


def test_cer_handles_many_long_rows_without_corpus_wide_dp() -> None:
    refs = ["a" * 200 for _ in range(20)]
    hyps = ["a" * 199 + "b" for _ in range(20)]
    metrics = score_pairs(refs, hyps)
    assert metrics["cer"] == 20 / 4000
