from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audio_processing import convert_to_wav, load_wav_float32, sliding_windows, temporary_audio_dir, write_window_wav
from model_loader import VigilRuntime


VARIANT_LABELS = {
    "Validation-selected": None,
    "BCE": "stage2_bce",
    "BCE + SupCon": "stage2_bce_supcon",
}


def _extract_text(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "transcript", "prediction", "output", "hypothesis"):
            if key in result:
                return _extract_text(result[key])
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        return _extract_text(result[0])
    return str(result)


@dataclass
class WindowResult:
    index: int
    start_sec: float
    end_sec: float
    stage1_score: float
    candidate: bool
    stage2_score: float | None = None
    final: bool = False


class VigilInference:
    def __init__(self, runtime: VigilRuntime, *, window_seconds: float = 2.0, stride_seconds: float = 0.25, top_k: int = 3):
        self.runtime = runtime
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.top_k = int(top_k)

    def _stage1_score(self, wav_path: Path) -> float:
        features = self.runtime.openwakeword.extract(wav_path)
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.runtime.device)
        lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=self.runtime.device)
        with torch.inference_mode():
            score = torch.sigmoid(self.runtime.stage1.model(x, lengths)).detach().float().cpu().item()
        return float(score)

    def _stage2_score(self, wav_path: Path, variant: str) -> float:
        stage2 = self.runtime.stage2[variant]
        with torch.inference_mode():
            features = self.runtime.qwen.extract_audio_features(str(wav_path)).detach().float()
            hidden = features.unsqueeze(0).to(self.runtime.device)
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=self.runtime.device)
            score = torch.sigmoid(stage2.model(hidden, mask)["logit"]).detach().float().cpu().item()
        return float(score)

    def _transcribe_full_audio(self, wav_path: Path) -> str:
        model = self.runtime.qwen.wrapper or self.runtime.qwen.model
        if model is None:
            raise RuntimeError("Qwen model is not loaded")
        attempts = []
        if hasattr(model, "transcribe"):
            attempts.extend([lambda: model.transcribe(str(wav_path), language="English"), lambda: model.transcribe(str(wav_path))])
        if hasattr(model, "generate"):
            attempts.extend([lambda: model.generate(str(wav_path), do_sample=False), lambda: model.generate(str(wav_path))])
        if callable(model):
            attempts.append(lambda: model(str(wav_path)))
        errors = []
        with torch.inference_mode():
            for attempt in attempts:
                try:
                    return _extract_text(attempt()).strip()
                except TypeError as exc:
                    errors.append(f"TypeError:{exc}")
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}:{exc}")
        raise RuntimeError("Qwen transcript failed: " + " | ".join(errors))

    def analyze(
        self,
        input_path: Path | str,
        *,
        variant_label: str = "Validation-selected",
        run_transcript_after_trigger: bool = True,
        debug_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        if input_path is None:
            raise ValueError("audio input is required")
        variant = VARIANT_LABELS.get(variant_label)
        if variant is None:
            variant = self.runtime.selected_variant
        if variant not in self.runtime.stage2:
            raise ValueError(f"unknown variant: {variant_label}")
        timings: dict[str, float | None] = {
            "stage1_latency_sec": None,
            "qwen_encoder_latency_sec": None,
            "stage2_head_latency_sec": None,
            "asr_transcript_latency_sec": None,
            "total_latency_sec": None,
        }
        total_started = time.perf_counter()
        with temporary_audio_dir(debug_dir) as work_dir:
            full_wav = convert_to_wav(input_path, work_dir)
            sample_rate, waveform = load_wav_float32(full_wav)
            windows = sliding_windows(
                waveform,
                sample_rate,
                window_seconds=self.window_seconds,
                stride_seconds=self.stride_seconds,
            )
            window_rows: list[WindowResult] = []
            stage1_started = time.perf_counter()
            window_paths = []
            for window in windows:
                window_path = work_dir / f"window_{window.index:04d}.wav"
                write_window_wav(window_path, sample_rate, window.waveform)
                window_paths.append((window, window_path))
                score = self._stage1_score(window_path)
                window_rows.append(
                    WindowResult(
                        index=window.index,
                        start_sec=window.start_sec,
                        end_sec=window.end_sec,
                        stage1_score=score,
                        candidate=score >= self.runtime.stage1.theta,
                    )
                )
            timings["stage1_latency_sec"] = time.perf_counter() - stage1_started
            candidates = sorted(
                [row for row in window_rows if row.candidate],
                key=lambda row: row.stage1_score,
                reverse=True,
            )[: self.top_k]
            by_index = {window.index: path for window, path in window_paths}
            winning: WindowResult | None = None
            qwen_latency = 0.0
            stage2_latency = 0.0
            for candidate in candidates:
                qwen_started = time.perf_counter()
                features = self.runtime.qwen.extract_audio_features(str(by_index[candidate.index])).detach().float()
                qwen_latency += time.perf_counter() - qwen_started
                stage2_started = time.perf_counter()
                stage2 = self.runtime.stage2[variant]
                hidden = features.unsqueeze(0).to(self.runtime.device)
                mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=self.runtime.device)
                with torch.inference_mode():
                    score = torch.sigmoid(stage2.model(hidden, mask)["logit"]).detach().float().cpu().item()
                stage2_latency += time.perf_counter() - stage2_started
                candidate.stage2_score = float(score)
                candidate.final = candidate.stage2_score >= stage2.theta
                if candidate.final and winning is None:
                    winning = candidate
            timings["qwen_encoder_latency_sec"] = qwen_latency if candidates else 0.0
            timings["stage2_head_latency_sec"] = stage2_latency if candidates else 0.0
            transcript = ""
            if winning is not None and run_transcript_after_trigger:
                asr_started = time.perf_counter()
                transcript = self._transcribe_full_audio(full_wav)
                timings["asr_transcript_latency_sec"] = time.perf_counter() - asr_started
            else:
                timings["asr_transcript_latency_sec"] = 0.0
            timings["total_latency_sec"] = time.perf_counter() - total_started
            theta2 = self.runtime.stage2[variant].theta
            table_rows = [
                {
                    "Window": row.index,
                    "Start": round(row.start_sec, 3),
                    "End": round(row.end_sec, 3),
                    "Stage 1": round(row.stage1_score, 6),
                    "Candidate": row.candidate,
                    "Stage 2": None if row.stage2_score is None else round(row.stage2_score, 6),
                    "Final": row.final,
                }
                for row in window_rows
            ]
            stage1_scores = [
                {"window": row.index, "start_sec": row.start_sec, "end_sec": row.end_sec, "score": row.stage1_score}
                for row in window_rows
            ]
            return {
                "final_trigger": winning is not None,
                "result_text": "VIGIL DETECTED" if winning is not None else "REJECTED",
                "variant": variant,
                "theta_1": self.runtime.stage1.theta,
                "theta_2": theta2,
                "stage1_score": max((row.stage1_score for row in window_rows), default=0.0),
                "stage2_score": winning.stage2_score if winning else None,
                "winning_window": None
                if winning is None
                else {
                    "index": winning.index,
                    "start_sec": winning.start_sec,
                    "end_sec": winning.end_sec,
                    "stage1_score": winning.stage1_score,
                    "stage2_score": winning.stage2_score,
                },
                "window_table": table_rows,
                "stage1_scores": stage1_scores,
                "transcript": transcript,
                "timings": timings,
                "model_selection": self.runtime.model_selection,
            }

