from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_processing import load_wav_float, write_wav_int16


@dataclass
class SegmentResult:
    status: str
    segments: list[dict]
    flags: list[str]


def _short_time_rms(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
    frame_size = max(int(sample_rate * 0.03), 1)
    hop_size = max(int(sample_rate * 0.01), 1)
    if samples.size < frame_size:
        padded = np.pad(samples, (0, frame_size - samples.size))
        return np.array([float(np.sqrt(np.mean(np.square(padded))))]), frame_size, hop_size

    frame_count = 1 + int((samples.size - frame_size) / hop_size)
    rms_values = np.empty(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = index * hop_size
        frame = samples[start : start + frame_size]
        rms_values[index] = float(np.sqrt(np.mean(np.square(frame))))
    return rms_values, frame_size, hop_size


def segment_repeated_prompt(
    wav_path: Path,
    *,
    output_path_for_index,
    expected_count: int,
) -> SegmentResult:
    samples, sample_rate = load_wav_float(wav_path)
    if samples.size == 0:
        return SegmentResult("failed_no_segments", [], ["empty_audio"])

    rms_values, frame_size, hop_size = _short_time_rms(samples, sample_rate)
    max_energy = float(np.max(rms_values)) if rms_values.size else 0.0
    if max_energy <= 0.0:
        return SegmentResult("failed_no_segments", [], ["no_detectable_energy"])

    low_floor = float(np.percentile(rms_values, 20))
    median_energy = float(np.median(rms_values))
    threshold = max(low_floor * 3.0, median_energy * 1.5, max_energy * 0.08, 0.008)
    speech_frames = rms_values > threshold

    regions: list[tuple[int, int]] = []
    start_frame: int | None = None
    for frame_index, is_speech in enumerate(speech_frames):
        if is_speech and start_frame is None:
            start_frame = frame_index
        elif not is_speech and start_frame is not None:
            regions.append((start_frame, frame_index - 1))
            start_frame = None
    if start_frame is not None:
        regions.append((start_frame, len(speech_frames) - 1))

    if not regions:
        return SegmentResult("failed_no_segments", [], ["no_speech_regions"])

    merge_gap_sec = 0.18
    min_duration_sec = 0.12
    padding_sec = 0.06
    merged: list[tuple[float, float]] = []
    for start, end in regions:
        start_time = (start * hop_size) / sample_rate
        end_time = ((end * hop_size) + frame_size) / sample_rate
        if not merged:
            merged.append((start_time, end_time))
            continue
        previous_start, previous_end = merged[-1]
        if start_time - previous_end <= merge_gap_sec:
            merged[-1] = (previous_start, end_time)
        else:
            merged.append((start_time, end_time))

    kept = [
        (max(0.0, start - padding_sec), min(samples.size / sample_rate, end + padding_sec))
        for start, end in merged
        if end - start >= min_duration_sec
    ]

    if not kept:
        return SegmentResult("failed_no_segments", [], ["segments_too_short"])

    segments: list[dict] = []
    for segment_index, (start_time, end_time) in enumerate(kept, start=1):
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_samples = samples[start_sample:end_sample]
        output_path = output_path_for_index(segment_index)
        write_wav_int16(output_path, segment_samples, sample_rate)
        segments.append(
            {
                "segment_index": segment_index,
                "path": output_path,
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "duration_sec": end_time - start_time,
            }
        )

    detected = len(segments)
    if detected == expected_count:
        return SegmentResult("ok", segments, [])
    if abs(detected - expected_count) == 1:
        return SegmentResult("warning_count_mismatch", segments, ["segmentation_count_mismatch"])
    return SegmentResult("warning_count_mismatch", segments, ["segmentation_count_mismatch"])
