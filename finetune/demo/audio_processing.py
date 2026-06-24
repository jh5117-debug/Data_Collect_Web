from __future__ import annotations

import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from vigil_two_stage.audio import write_wav


@dataclass(frozen=True)
class AudioWindow:
    index: int
    start_sec: float
    end_sec: float
    waveform: np.ndarray
    padded: bool


def ensure_ffmpeg() -> str:
    result = subprocess.run(["which", "ffmpeg"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg command is required for microphone/upload audio conversion")
    return result.stdout.strip()


def convert_to_wav(input_path: Path | str, work_dir: Path | str, sample_rate: int = 16000) -> Path:
    input_path = Path(input_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ensure_ffmpeg()
    output_path = work_dir / "input_16k_mono.wav"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-2000:]}")
    return output_path


def load_wav_float32(path: Path | str, expected_sample_rate: int = 16000) -> tuple[int, np.ndarray]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != expected_sample_rate:
        raise RuntimeError(f"expected {expected_sample_rate} Hz audio after conversion, got {sample_rate}")
    if audio.size == 0:
        raise RuntimeError("audio is empty")
    if not np.isfinite(audio).all():
        raise RuntimeError("audio contains NaN or Inf")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return int(sample_rate), audio.astype(np.float32, copy=False)


def sliding_windows(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float = 2.0,
    stride_seconds: float = 0.25,
) -> list[AudioWindow]:
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    window_samples = int(round(sample_rate * window_seconds))
    stride_samples = int(round(sample_rate * stride_seconds))
    if window_samples <= 0:
        raise ValueError("window_seconds is too small")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError("waveform is empty")
    starts = list(range(0, max(1, waveform.size - window_samples + 1), stride_samples))
    final_start = max(0, waveform.size - window_samples)
    if final_start not in starts:
        starts.append(final_start)
    starts = sorted(set(starts))
    windows = []
    for index, start in enumerate(starts):
        end = start + window_samples
        chunk = waveform[start:min(end, waveform.size)]
        padded = chunk.size < window_samples
        if padded:
            chunk = np.pad(chunk, (0, window_samples - chunk.size))
        windows.append(
            AudioWindow(
                index=index,
                start_sec=start / float(sample_rate),
                end_sec=end / float(sample_rate),
                waveform=chunk.astype(np.float32, copy=False),
                padded=bool(padded),
            )
        )
    return windows


def write_window_wav(path: Path | str, sample_rate: int, waveform: np.ndarray) -> None:
    scaled = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0) * 32767.0
    write_wav(path, sample_rate, scaled.astype(np.float32))


@contextmanager
def temporary_audio_dir(debug_dir: Path | str | None = None) -> Iterator[Path]:
    if debug_dir is not None:
        path = Path(debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="vigil_demo_") as tmp:
        yield Path(tmp)
