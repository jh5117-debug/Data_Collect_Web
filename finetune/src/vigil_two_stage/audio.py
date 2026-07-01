from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_dir, run_command, sha256_file


def ffmpeg_convert_to_wav(source: Path, target: Path, sample_rate: int = 16000) -> dict[str, Any]:
    ensure_dir(target.parent)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(target),
    ]
    result = run_command(cmd)
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-4000:],
        "cmd": cmd,
    }


def read_wav(path: Path | str) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        frames = wf.getnframes()
        data = wf.readframes(frames)
    if sampwidth != 2:
        raise ValueError(f"expected int16 wav, got sample width {sampwidth}")
    audio = np.frombuffer(data, dtype="<i2").astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


def write_wav(path: Path | str, sample_rate: int, audio: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    clipped = np.clip(audio, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())


def validate_wav(path: Path | str, expected_sample_rate: int = 16000) -> dict[str, Any]:
    try:
        sample_rate, audio = read_wav(path)
        finite = bool(np.isfinite(audio).all())
        samples = int(audio.shape[0])
        duration = samples / float(sample_rate) if sample_rate else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio / 32768.0)))) if samples else 0.0
        peak = float(np.max(np.abs(audio)) / 32768.0) if samples else 0.0
        ok = sample_rate == expected_sample_rate and samples > 0 and finite and peak > 1e-4
        return {
            "ok": ok,
            "sample_rate": sample_rate,
            "channels": 1,
            "samples": samples,
            "duration_sec": duration,
            "rms": rms,
            "peak": peak,
            "sha256": sha256_file(path),
            "reason": "" if ok else "invalid_sample_rate_or_empty_or_silent",
        }
    except Exception as exc:
        return {"ok": False, "reason": f"wav_decode_failed:{type(exc).__name__}:{exc}"}


def detect_speech_bounds(audio: np.ndarray, sample_rate: int, frame_ms: float = 30.0, hop_ms: float = 10.0) -> tuple[int, int]:
    if audio.size == 0:
        return 0, 0
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    abs_audio = np.abs(audio)
    peak = float(abs_audio.max()) if abs_audio.size else 0.0
    if peak <= 0:
        return 0, int(audio.size)
    energies = []
    starts = []
    for start in range(0, max(1, audio.size - frame + 1), hop):
        chunk = abs_audio[start : start + frame]
        energies.append(float(np.sqrt(np.mean(np.square(chunk)))))
        starts.append(start)
    if not energies:
        return 0, int(audio.size)
    arr = np.asarray(energies)
    threshold = max(80.0, float(np.percentile(arr, 20) * 2.0), peak * 0.02)
    active = np.nonzero(arr >= threshold)[0]
    if active.size == 0:
        return 0, int(audio.size)
    begin = max(0, starts[int(active[0])] - int(0.05 * sample_rate))
    end = min(int(audio.size), starts[int(active[-1])] + frame + int(0.05 * sample_rate))
    return begin, max(begin, end)


def fixed_window_bounds(
    prompt_group: str,
    speech_begin: int,
    speech_end: int,
    total_samples: int,
    sample_rate: int,
    window_seconds: float,
) -> list[tuple[int, int, str]]:
    window = int(round(sample_rate * window_seconds))
    if total_samples <= 0:
        return [(0, window, "empty_pad")]
    speech_begin = max(0, min(speech_begin, total_samples))
    speech_end = max(speech_begin, min(speech_end, total_samples))
    speech_len = max(1, speech_end - speech_begin)
    if prompt_group == "P2_phrase_plus_vigil":
        end = speech_end
        start = end - window
        return [(start, end, "p2_final_speech_window")]
    if prompt_group == "P3_vigil_plus_phrase":
        start = speech_begin
        return [(start, start + window, "p3_initial_speech_window")]
    if prompt_group == "P4_negative" and speech_len > int(window * 1.35):
        bounds = []
        step = window
        start = speech_begin
        while start < speech_end:
            bounds.append((start, start + window, "p4_negative_nonoverlap"))
            start += step
        return bounds
    center = (speech_begin + speech_end) // 2
    start = center - window // 2
    return [(start, start + window, "center_speech_window" if prompt_group == "P1_vigil_only" else "center_negative_window")]


def materialize_window(audio: np.ndarray, start: int, end: int) -> tuple[np.ndarray, int, int]:
    length = max(0, end - start)
    out = np.zeros(length, dtype=np.float32)
    src_start = max(0, start)
    src_end = min(int(audio.size), end)
    if src_end > src_start:
        dst_start = src_start - start
        out[dst_start : dst_start + (src_end - src_start)] = audio[src_start:src_end]
    left_pad = max(0, -start)
    right_pad = max(0, end - int(audio.size))
    return out, left_pad, right_pad


def extract_fft_features(wav_path: Path | str, feature_dim: int = 96, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    sample_rate, audio = read_wav(wav_path)
    audio = audio.astype(np.float32) / 32768.0
    frame = max(64, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if audio.size < frame:
        audio = np.pad(audio, (0, frame - audio.size))
    win = np.hanning(frame).astype(np.float32)
    rows = []
    for start in range(0, max(1, audio.size - frame + 1), hop):
        chunk = audio[start : start + frame]
        if chunk.size < frame:
            chunk = np.pad(chunk, (0, frame - chunk.size))
        spec = np.abs(np.fft.rfft(chunk * win))
        bins = np.array_split(spec, feature_dim)
        row = [math.log1p(float(b.mean())) for b in bins]
        rows.append(row)
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        arr = np.zeros((1, feature_dim), dtype=np.float32)
    return arr
