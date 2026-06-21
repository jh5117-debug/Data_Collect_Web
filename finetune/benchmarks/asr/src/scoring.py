from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class AlignmentCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    sentence_errors: int
    total_sentences: int
    exact_matches: int


def _levenshtein_counts(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int, int]:
    # DP cell: (edit distance, substitutions, deletions, insertions)
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        prev = dp[i - 1][0]
        dp[i][0] = (prev[0] + 1, prev[1], prev[2] + 1, prev[3])
    for j in range(1, cols):
        prev = dp[0][j - 1]
        dp[0][j] = (prev[0] + 1, prev[1], prev[2], prev[3] + 1)
    for i in range(1, rows):
        for j in range(1, cols):
            if ref_words[i - 1] == hyp_words[j - 1]:
                candidates = [dp[i - 1][j - 1]]
            else:
                prev = dp[i - 1][j - 1]
                candidates = [(prev[0] + 1, prev[1] + 1, prev[2], prev[3])]
            prev = dp[i - 1][j]
            candidates.append((prev[0] + 1, prev[1], prev[2] + 1, prev[3]))
            prev = dp[i][j - 1]
            candidates.append((prev[0] + 1, prev[1], prev[2], prev[3] + 1))
            dp[i][j] = min(candidates, key=lambda x: (x[0], x[3], x[2], x[1]))
    _, s, d, i = dp[-1][-1]
    return s, d, i


def alignment_counts(
    references: Iterable[str],
    hypotheses: Iterable[str],
    normalizer: Callable[[str], str] | None = None,
) -> AlignmentCounts:
    substitutions = deletions = insertions = reference_words = 0
    sentence_errors = total_sentences = exact_matches = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        if normalizer is not None:
            ref = normalizer(ref)
            hyp = normalizer(hyp)
        ref_words = ref.split()
        hyp_words = hyp.split()
        s, d, i = _levenshtein_counts(ref_words, hyp_words)
        substitutions += s
        deletions += d
        insertions += i
        reference_words += len(ref_words)
        total_sentences += 1
        if s or d or i:
            sentence_errors += 1
        else:
            exact_matches += 1
    return AlignmentCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=reference_words,
        sentence_errors=sentence_errors,
        total_sentences=total_sentences,
        exact_matches=exact_matches,
    )


def character_error_rate(
    references: Iterable[str],
    hypotheses: Iterable[str],
    normalizer: Callable[[str], str] | None = None,
) -> float | None:
    ref_chars = []
    hyp_chars = []
    for ref, hyp in zip(references, hypotheses, strict=True):
        if normalizer is not None:
            ref = normalizer(ref)
            hyp = normalizer(hyp)
        ref_chars.append(" ".join(ref.split()))
        hyp_chars.append(" ".join(hyp.split()))
    joined_ref = "\n".join(ref_chars)
    joined_hyp = "\n".join(hyp_chars)
    if not joined_ref:
        return None
    s, d, i = _levenshtein_counts(list(joined_ref), list(joined_hyp))
    return (s + d + i) / len(joined_ref)


def score_pairs(
    references: Iterable[str],
    hypotheses: Iterable[str],
    normalizer: Callable[[str], str] | None = None,
) -> dict[str, float | int | None]:
    references_list = list(references)
    hypotheses_list = list(hypotheses)
    counts = alignment_counts(references_list, hypotheses_list, normalizer)
    errors = counts.substitutions + counts.deletions + counts.insertions
    wer = errors / counts.reference_words if counts.reference_words else None
    sentence_error_rate = counts.sentence_errors / counts.total_sentences if counts.total_sentences else None
    exact_match_rate = counts.exact_matches / counts.total_sentences if counts.total_sentences else None
    return {
        "wer": wer,
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "reference_words": counts.reference_words,
        "sentence_errors": counts.sentence_errors,
        "total_sentences": counts.total_sentences,
        "sentence_error_rate": sentence_error_rate,
        "exact_match_rate": exact_match_rate,
        "cer": character_error_rate(references_list, hypotheses_list, normalizer),
    }


def score_prediction_rows(rows: list[dict], normalized: bool = True) -> dict[str, float | int | None]:
    ref_key = "normalized_reference" if normalized else "reference"
    hyp_key = "normalized_hypothesis" if normalized else "hypothesis"
    ok_rows = [row for row in rows if row.get("status") == "success"]
    return score_pairs([str(row.get(ref_key, "")) for row in ok_rows], [str(row.get(hyp_key, "")) for row in ok_rows])
