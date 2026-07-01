from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import contains_exact_vigil, normalize_product_casing, normalized_for_matching, redact_identity


PROMPT_GROUPS: dict[str, dict[str, Any]] = {
    "P1_vigil_only": {
        "title": "VIGIL Only",
        "label": 1,
        "contains_vigil": True,
        "wake_intent": True,
        "is_negative": False,
        "fixed_transcript": "VIGIL",
    },
    "P2_phrase_plus_vigil": {
        "title": "Phrase/Sentence + VIGIL",
        "label": 1,
        "contains_vigil": True,
        "wake_intent": True,
        "is_negative": False,
    },
    "P3_vigil_plus_phrase": {
        "title": "VIGIL + Phrase/Sentence",
        "label": 1,
        "contains_vigil": True,
        "wake_intent": True,
        "is_negative": False,
    },
    "P4_negative": {
        "title": "Negative Examples",
        "label": 0,
        "contains_vigil": False,
        "wake_intent": False,
        "is_negative": True,
    },
}


@dataclass(frozen=True)
class ExportBundle:
    zip_path: Path
    root: str
    clips: list[dict[str, Any]]
    sessions: list[dict[str, Any]]
    accounts: list[dict[str, Any]]
    names: list[str]


def _find_root(names: list[str]) -> str:
    if not names:
        raise ValueError("empty export zip")
    first = names[0]
    return first.split("/")[0] + "/" if "/" in first else ""


def _read_text(zf: zipfile.ZipFile, root: str, rel: str) -> str:
    return zf.read(root + rel).decode("utf-8")


def load_export(zip_path: Path | str) -> ExportBundle:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        root = _find_root(names)
        clips = [json.loads(line) for line in _read_text(zf, root, "metadata/clips.jsonl").splitlines() if line.strip()]
        sessions = [json.loads(line) for line in _read_text(zf, root, "metadata/sessions.jsonl").splitlines() if line.strip()]
        accounts = list(csv.DictReader(io.StringIO(_read_text(zf, root, "metadata/accounts.csv"))))
    return ExportBundle(zip_path=zip_path, root=root, clips=clips, sessions=sessions, accounts=accounts, names=names)


def resolve_canonical_audio_member(bundle: ExportBundle, clip: dict[str, Any]) -> str | None:
    clip_id = clip.get("clip_id")
    if not clip_id:
        return None
    prefix = f"{bundle.root}audio_raw/{clip_id}."
    matches = [name for name in bundle.names if name.startswith(prefix) and not name.endswith("/")]
    if len(matches) == 1:
        return matches[0]
    raw_path = clip.get("raw_audio_path")
    if raw_path:
        candidate = bundle.root + str(raw_path).lstrip("/")
        if candidate in bundle.names:
            return candidate
    suffix_matches = [
        name
        for name in bundle.names
        if name.endswith(f"/{clip_id}.webm") or name.endswith(f"/{clip_id}.m4a") or name.endswith(f"/{clip_id}.wav")
    ]
    return suffix_matches[0] if suffix_matches else None


def infer_phrase_id(transcript: str, label: int, hard_negative_phrases: list[str]) -> str:
    if label == 1:
        return "vigil"
    norm = normalized_for_matching(transcript)
    for phrase in hard_negative_phrases:
        if normalized_for_matching(phrase) == norm:
            return normalized_for_matching(phrase)
    first = norm.split()[0] if norm else ""
    if first in {normalized_for_matching(x) for x in hard_negative_phrases}:
        return first
    return "background"


def validate_and_map_clip(clip: dict[str, Any], hard_negative_phrases: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    clip_id = clip.get("clip_id") or "unknown"
    prompt_group = clip.get("prompt_group") or clip.get("prompt_id") or "legacy"
    reasons: list[str] = []
    if prompt_group == "legacy" or prompt_group not in PROMPT_GROUPS:
        transcript = normalize_product_casing(clip.get("transcript") or clip.get("normalized_transcript") or prompt_group)
        mapped = {
            "clip_id": clip_id,
            "prompt_group": "legacy",
            "prompt_title": clip.get("prompt_title") or str(prompt_group),
            "transcript": transcript,
            "normalized_transcript": normalized_for_matching(transcript),
            "label": 1 if contains_exact_vigil(transcript) else 0,
            "contains_vigil": contains_exact_vigil(transcript),
            "wake_intent": contains_exact_vigil(transcript),
            "is_negative": False,
            "phrase_id": infer_phrase_id(transcript, 1 if contains_exact_vigil(transcript) else 0, hard_negative_phrases),
        }
        return mapped, None
    spec = PROMPT_GROUPS[prompt_group]
    transcript = spec.get("fixed_transcript") or clip.get("transcript") or clip.get("normalized_transcript") or ""
    transcript = normalize_product_casing(transcript)
    if not transcript:
        reasons.append("missing_transcript")
    has_vigil = contains_exact_vigil(transcript)
    if prompt_group in {"P2_phrase_plus_vigil", "P3_vigil_plus_phrase"} and not has_vigil:
        reasons.append("positive_prompt_missing_exact_vigil")
    if prompt_group == "P4_negative" and has_vigil:
        reasons.append("negative_prompt_contains_exact_vigil")
    for key in ("contains_vigil", "wake_intent", "is_negative"):
        if key in clip and clip[key] is not None and bool(clip[key]) != bool(spec[key]):
            reasons.append(f"inconsistent_{key}")
    if reasons:
        return None, {
            "clip_id": clip_id,
            "prompt_group": prompt_group,
            "reasons": reasons,
            "participant_id": redact_identity(clip.get("participant_id")),
            "session_id": clip.get("session_id"),
        }
    mapped = {
        "clip_id": clip_id,
        "prompt_group": prompt_group,
        "prompt_title": spec["title"],
        "transcript": transcript,
        "normalized_transcript": normalized_for_matching(transcript),
        "label": int(spec["label"]),
        "contains_vigil": bool(spec["contains_vigil"]),
        "wake_intent": bool(spec["wake_intent"]),
        "is_negative": bool(spec["is_negative"]),
        "phrase_id": infer_phrase_id(transcript, int(spec["label"]), hard_negative_phrases),
    }
    return mapped, None


def canonical_samples(bundle: ExportBundle, hard_negative_phrases: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for clip in bundle.clips:
        clip_id = clip.get("clip_id")
        if not clip_id:
            rejected.append({"clip_id": "missing", "reasons": ["missing_clip_id"]})
            continue
        if clip_id in seen:
            rejected.append({"clip_id": clip_id, "reasons": ["duplicate_clip_id_in_metadata"]})
            continue
        seen.add(clip_id)
        mapped, rejection = validate_and_map_clip(clip, hard_negative_phrases)
        if rejection:
            rejected.append(rejection)
            continue
        member = resolve_canonical_audio_member(bundle, clip)
        if not member:
            rejected.append({"clip_id": clip_id, "reasons": ["missing_canonical_audio_member"]})
            continue
        assert mapped is not None
        row = dict(mapped)
        row.update(
            {
                "session_id": clip.get("session_id"),
                "participant_key": str(clip.get("participant_id") or clip.get("account_id") or "unknown"),
                "canonical_audio_member": member,
                "file_size_bytes": clip.get("file_size_bytes"),
                "created_at_utc": clip.get("created_at_utc"),
                "auto_qc_status": clip.get("auto_qc_status") or clip.get("status") or "",
            }
        )
        samples.append(row)
    return samples, rejected
