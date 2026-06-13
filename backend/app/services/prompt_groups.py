import re
from dataclasses import dataclass

from fastapi import HTTPException

VIGIL_WORD_RE = re.compile(r"\bvigil\b", re.IGNORECASE)


@dataclass(frozen=True)
class PromptGroupInfo:
    prompt_group: str
    prompt_title: str
    contains_vigil: bool
    wake_intent: bool
    is_negative: bool
    transcript: str
    normalized_transcript: str


PROMPT_GROUP_TITLES = {
    "P1_vigil_only": "VIGIL Only",
    "P2_phrase_plus_vigil": "Phrase/Sentence + VIGIL",
    "P3_vigil_plus_phrase": "VIGIL + Phrase/Sentence",
    "P4_negative": "Negative Examples",
}


def normalize_transcript(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", value.strip())
    return VIGIL_WORD_RE.sub("Vigil", collapsed)


def contains_exact_vigil(value: str) -> bool:
    return bool(VIGIL_WORD_RE.search(value))


def derive_prompt_group_info(prompt_group: str | None, transcript: str | None) -> PromptGroupInfo:
    group = (prompt_group or "").strip()
    raw_transcript = (transcript or "").strip()

    if group == "P1_vigil_only":
        normalized = "Vigil"
        return PromptGroupInfo(
            prompt_group=group,
            prompt_title=PROMPT_GROUP_TITLES[group],
            contains_vigil=True,
            wake_intent=True,
            is_negative=False,
            transcript=normalized,
            normalized_transcript=normalized,
        )

    if group in {"P2_phrase_plus_vigil", "P3_vigil_plus_phrase"}:
        if not raw_transcript:
            raise HTTPException(status_code=400, detail="transcript is required")
        normalized = normalize_transcript(raw_transcript)
        if not contains_exact_vigil(normalized):
            raise HTTPException(status_code=400, detail="This prompt should include the word 'Vigil'.")
        return PromptGroupInfo(
            prompt_group=group,
            prompt_title=PROMPT_GROUP_TITLES[group],
            contains_vigil=True,
            wake_intent=True,
            is_negative=False,
            transcript=normalized,
            normalized_transcript=normalized,
        )

    if group == "P4_negative":
        if not raw_transcript:
            raise HTTPException(status_code=400, detail="transcript is required")
        normalized = normalize_transcript(raw_transcript)
        if contains_exact_vigil(normalized):
            raise HTTPException(
                status_code=400,
                detail="Negative examples should not contain the exact word 'Vigil'. Please use Prompt 2 or Prompt 3 instead.",
            )
        return PromptGroupInfo(
            prompt_group=group,
            prompt_title=PROMPT_GROUP_TITLES[group],
            contains_vigil=False,
            wake_intent=False,
            is_negative=True,
            transcript=normalized,
            normalized_transcript=normalized,
        )

    raise HTTPException(status_code=400, detail="invalid prompt_group")


def legacy_prompt_group_info(prompt_id: str, expected_transcript: str | None = None) -> PromptGroupInfo:
    transcript = normalize_transcript(expected_transcript or prompt_id)
    return PromptGroupInfo(
        prompt_group="legacy",
        prompt_title=prompt_id,
        contains_vigil=contains_exact_vigil(transcript),
        wake_intent=contains_exact_vigil(transcript),
        is_negative=False,
        transcript=transcript,
        normalized_transcript=transcript,
    )
