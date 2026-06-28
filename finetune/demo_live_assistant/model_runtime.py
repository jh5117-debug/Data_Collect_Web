from __future__ import annotations

import sys
import time
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from trigger import apply_positive_bias
from vigil_two_stage.qwen_text_result import extract_qwen_text


DEFAULT_RUN_DIR = Path("finetune/model_bundles/vigil_latest_optimized_20260626_085405")


@dataclass
class RuntimeStatus:
    mode: str
    message: str
    model_loaded: dict[str, bool]
    gpu: dict[str, str | int | bool | None]


class AssistantModelRuntime:
    def __init__(self, run_dir: Path | str = DEFAULT_RUN_DIR, *, force_mock: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.force_mock = force_mock
        self.inference: Any | None = None
        self.mode = "mock" if force_mock else "unloaded"
        self.message = "not loaded"
        self.theta1 = 0.9973222613334656
        self.theta2 = 0.9877771735191345
        self.selected_variant = "stage2_bce_supcon"

    def load(self) -> None:
        if self.force_mock:
            self.mode = "mock"
            self.message = "forced mock mode"
            return
        try:
            demo_dir = Path("finetune/demo").resolve()
            if str(demo_dir) not in sys.path:
                sys.path.insert(0, str(demo_dir))
            from inference import VigilInference
            from model_loader import load_runtime

            try:
                runtime = load_runtime(self.run_dir)
            except FileNotFoundError:
                if (self.run_dir / "PUBLIC_MANIFEST.json").exists():
                    runtime = self._load_bundle_runtime()
                else:
                    raise
            self.inference = VigilInference(runtime)
            self.theta1 = float(runtime.stage1.theta)
            self.theta2 = float(runtime.stage2[runtime.selected_variant].theta)
            self.selected_variant = runtime.selected_variant
            self.mode = "real"
            self.message = "real VIGIL runtime loaded"
        except Exception as exc:
            self.inference = None
            self.mode = "partial"
            self.message = f"real model not loaded: {type(exc).__name__}: {exc}"

    def _load_bundle_runtime(self) -> Any:
        demo_dir = Path("finetune/demo").resolve()
        if str(demo_dir) not in sys.path:
            sys.path.insert(0, str(demo_dir))
        from model_loader import FrozenQwenAudioAdapter, OfficialOpenWakeWordExtractor, load_stage1, load_stage2, require_single_rtx_3090

        manifest = json.loads((self.run_dir / "PUBLIC_MANIFEST.json").read_text(encoding="utf-8"))
        selected = str(manifest["selected_config"]["variant"])
        device = require_single_rtx_3090()
        openwakeword = OfficialOpenWakeWordExtractor()
        stage1 = load_stage1(self.run_dir, device)
        stage2 = {selected: load_stage2(self.run_dir, selected, device)}
        qwen = FrozenQwenAudioAdapter("Qwen/Qwen3-ASR-1.7B")
        qwen.load()
        integrity = qwen.integrity()
        if integrity.trainable_parameters != 0:
            raise RuntimeError(f"Qwen must remain frozen, got {integrity.trainable_parameters} trainable parameters")
        return SimpleNamespace(
            run_dir=self.run_dir,
            device=device,
            openwakeword=openwakeword,
            qwen=qwen,
            stage1=stage1,
            stage2=stage2,
            selected_variant=selected,
            model_selection={
                "selected_variant": selected,
                "source": "PUBLIC_MANIFEST.json",
                "stage1_threshold": manifest["stage1_threshold"]["threshold"],
                "stage2_threshold": manifest["stage2_threshold"]["threshold"],
            },
        )

    def status(self) -> RuntimeStatus:
        cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda and torch.cuda.device_count() else None
        return RuntimeStatus(
            mode=self.mode,
            message=self.message,
            model_loaded={
                "openwakeword": self.mode == "real",
                "stage1_head": self.mode == "real",
                "qwen_asr": self.mode == "real",
                "stage2_head": self.mode == "real",
            },
            gpu={
                "cuda_available": cuda,
                "visible_device_count": torch.cuda.device_count() if cuda else 0,
                "device_name": gpu_name,
            },
        )

    def support_scores(self, clip_count: int) -> list[float]:
        if self.mode != "real":
            return [0.75] * int(clip_count)
        return [0.75] * int(clip_count)

    def _transcribe_audio(self, audio_path: Path) -> dict[str, str | None]:
        if self.inference is None:
            return {"text": None, "path": None, "result_type": None, "error": "runtime not loaded"}
        model = self.inference.runtime.qwen.wrapper or self.inference.runtime.qwen.model
        if model is None:
            return {"text": None, "path": None, "result_type": None, "error": "Qwen model is not loaded"}
        if not hasattr(model, "transcribe"):
            return {"text": None, "path": None, "result_type": None, "error": "Qwen model has no transcribe method"}
        try:
            with torch.inference_mode():
                raw = model.transcribe(str(audio_path), language=None)
            extracted = extract_qwen_text(raw)
            return {
                "text": extracted.text,
                "path": extracted.extraction_path,
                "result_type": extracted.result_type,
                "error": None,
            }
        except Exception as exc:
            return {"text": None, "path": None, "result_type": None, "error": f"{type(exc).__name__}: {exc}"}

    def analyze_audio(self, audio_path: Path, calibration: dict[str, Any] | None = None) -> dict[str, Any]:
        calibration = calibration or {}
        bias = float(calibration.get("bias") or 0.0)
        start = time.perf_counter()
        if self.inference is not None:
            transcript_started = time.perf_counter()
            transcript = self._transcribe_audio(audio_path)
            transcript_latency_ms = (time.perf_counter() - transcript_started) * 1000.0
            result = self.inference.analyze(audio_path, run_transcript_after_trigger=False)
            stage2_score = result.get("stage2_score")
            if stage2_score is None:
                stage2_values = [row.get("Stage 2") for row in result.get("window_table", []) if row.get("Stage 2") is not None]
                stage2_score = max(stage2_values) if stage2_values else None
            calibrated_score = apply_positive_bias(stage2_score, bias)
            candidate = bool(any(row.get("Candidate") for row in result.get("window_table", [])))
            trigger_detected = bool(candidate and calibrated_score is not None and calibrated_score >= float(result["theta_2"]))
            return {
                "mode": self.mode,
                "rolling_transcript": transcript.get("text") or "",
                "stage1_score": result.get("stage1_score", 0.0),
                "stage2_score": stage2_score,
                "calibrated_stage2_score": calibrated_score,
                "theta_1": result["theta_1"],
                "theta_2": result["theta_2"],
                "candidate": candidate,
                "trigger_detected": trigger_detected,
                "winning_window": result.get("winning_window"),
                "latency_ms": (time.perf_counter() - start) * 1000.0,
                "debug": {
                    "variant": result.get("variant"),
                    "calibration_bias": bias,
                    "qwen_weight_instances": 1,
                    "qwen_compute_paths": "transcribe plus stage2_feature_extraction_for_candidates",
                    "stage2_qwen_feature_path_used": candidate,
                    "qwen_transcript_extraction_path": transcript.get("path"),
                    "qwen_transcript_result_type": transcript.get("result_type"),
                    "qwen_transcript_error": transcript.get("error"),
                    "qwen_transcript_latency_ms": transcript_latency_ms,
                },
            }
        data = audio_path.read_bytes() if audio_path.exists() else b""
        text = "VIGIL" if b"VIGIL" in data.upper() or len(data) % 2 == 1 else ""
        stage1_score = 0.998 if text else 0.12
        stage2_score = 0.995 if text else 0.08
        calibrated_score = apply_positive_bias(stage2_score, bias)
        trigger_detected = bool(stage1_score >= self.theta1 and calibrated_score is not None and calibrated_score >= self.theta2)
        return {
            "mode": self.mode,
            "rolling_transcript": text,
            "stage1_score": stage1_score,
            "stage2_score": stage2_score,
            "calibrated_stage2_score": calibrated_score,
            "theta_1": self.theta1,
            "theta_2": self.theta2,
            "candidate": stage1_score >= self.theta1,
            "trigger_detected": trigger_detected,
            "winning_window": {"index": 0, "start_sec": 0.0, "end_sec": 2.0} if trigger_detected else None,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "debug": {"variant": self.selected_variant, "calibration_bias": bias, "mock_reason": self.message},
        }
