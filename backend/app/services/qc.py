import json
from dataclasses import dataclass

import numpy as np

from ..models import Prompt


@dataclass
class QCResult:
    status: str
    flags: list[str]


def expected_duration_bounds(prompt: Prompt, clip_type: str) -> tuple[float, float]:
    if clip_type == "calibration":
        return 1.0, 12.0
    if prompt.recording_mode == "repeat":
        repetitions = max(prompt.target_repetition_count, 1)
        return repetitions * 0.35, repetitions * 2.8
    return 0.35, 8.0


def run_audio_qc(
    *,
    duration_sec: float,
    samples: np.ndarray,
    prompt: Prompt,
    clip_type: str,
    hard_flags: list[str] | None = None,
) -> QCResult:
    flags = list(hard_flags or [])
    if duration_sec < 0.3:
        flags.append("duration_less_than_0_3_sec")

    min_expected, max_expected = expected_duration_bounds(prompt, clip_type)
    if duration_sec < min_expected:
        flags.append("duration_much_shorter_than_expected")
    if duration_sec > max_expected:
        flags.append("duration_much_longer_than_expected")

    if samples.size == 0:
        flags.append("empty_audio")
    else:
        rms = float(np.sqrt(np.mean(np.square(samples))))
        clipped_ratio = float(np.mean(np.abs(samples) > 0.98))
        if rms < 0.01:
            flags.append("very_low_volume")
        if clipped_ratio > 0.01:
            flags.append("severe_clipping")

    hard_failures = {"empty_audio", "duration_less_than_0_3_sec", "ffmpeg_conversion_failed"}
    if any(flag in hard_failures for flag in flags):
        return QCResult("auto_rejected", sorted(set(flags)))
    if flags:
        return QCResult("flagged_for_review", sorted(set(flags)))
    return QCResult("auto_accepted", [])


def flags_to_json(flags: list[str]) -> str:
    return json.dumps(flags, sort_keys=True)


def flags_from_json(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    return parsed if isinstance(parsed, list) else []
