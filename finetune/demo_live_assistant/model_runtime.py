from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from prototype import PrototypeCalibrationError, build_prototype, cosine_similarity
from trigger import apply_positive_bias, bounded_positive_bias
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

    def build_support_calibration(self, clip_records: list[dict[str, Any]]) -> dict[str, Any]:
        support_count = len(clip_records)
        if support_count < 3:
            return {
                "calibration_status": "need_more_positive_clips",
                "support_count": support_count,
                "method": None,
                "calibration_active": False,
                "bias": 0.0,
                "warnings": ["At least 3 accepted positive VIGIL clips are required."],
            }
        if self.mode != "real" or self.inference is None:
            if self.force_mock or self.mode == "mock":
                return self._mock_support_calibration(clip_records)
            return {
                "calibration_status": "model_not_loaded",
                "support_count": support_count,
                "method": None,
                "calibration_active": False,
                "bias": 0.0,
                "warnings": [f"Real VIGIL runtime is not loaded: {self.message}"],
            }

        started = time.perf_counter()
        embeddings: list[np.ndarray] = []
        support_scores: list[float] = []
        support_rows: list[dict[str, Any]] = []
        for clip in clip_records:
            try:
                support = self._stage2_embedding_and_score(Path(str(clip["audio_path"])))
            except Exception as exc:
                return {
                    "calibration_status": "support_embedding_failed",
                    "support_count": support_count,
                    "method": "few_shot_qwen_stage2_prototype",
                    "calibration_active": False,
                    "bias": 0.0,
                    "warnings": [f"{clip.get('clip_id', 'unknown')}: {type(exc).__name__}: {exc}"],
                }
            embeddings.append(np.asarray(support["embedding"], dtype=np.float32))
            score = float(support["score"])
            support_scores.append(score)
            support_rows.append(
                {
                    "clip_id": clip.get("clip_id"),
                    "prompt_group": clip.get("prompt_group"),
                    "transcript": clip.get("transcript"),
                    "stage2_score": score,
                }
            )

        try:
            prototype = build_prototype(embeddings)
        except PrototypeCalibrationError as exc:
            return {
                "calibration_status": "prototype_failed",
                "support_count": support_count,
                "method": "few_shot_qwen_stage2_prototype",
                "calibration_active": False,
                "bias": 0.0,
                "warnings": [str(exc)],
            }

        bias = bounded_positive_bias(support_scores, self.theta2)
        prototype_embedding = [float(value) for value in prototype.embedding.tolist()]
        return {
            "calibration_status": "ok",
            "support_count": support_count,
            "method": "few_shot_qwen_stage2_prototype",
            "calibration_active": True,
            "bias": bias,
            "threshold": self.theta2,
            "prototype_threshold": 0.6,
            "prototype_score_name": "cosine_similarity",
            "prototype_embedding": prototype_embedding,
            "prototype_embedding_dim": len(prototype_embedding),
            "support_stage2_scores": support_scores,
            "support_clips": support_rows,
            "support_pairwise_mean_similarity": prototype.pairwise_mean_similarity,
            "support_pairwise_min_similarity": prototype.pairwise_min_similarity,
            "model_variant": self.selected_variant,
            "calibration_latency_ms": (time.perf_counter() - started) * 1000.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "qwen_weights_updated": False,
            "openwakeword_weights_updated": False,
            "stage2_weights_updated": False,
            "warnings": [],
        }

    def _mock_support_calibration(self, clip_records: list[dict[str, Any]]) -> dict[str, Any]:
        support_scores = [0.75] * len(clip_records)
        prototype_embedding = [1.0] + [0.0] * 127
        return {
            "calibration_status": "ok",
            "support_count": len(clip_records),
            "method": "mock_few_shot_qwen_stage2_prototype",
            "calibration_active": True,
            "bias": bounded_positive_bias(support_scores, self.theta2),
            "threshold": self.theta2,
            "prototype_threshold": 0.6,
            "prototype_score_name": "cosine_similarity",
            "prototype_embedding": prototype_embedding,
            "prototype_embedding_dim": len(prototype_embedding),
            "support_stage2_scores": support_scores,
            "support_clips": [
                {
                    "clip_id": clip.get("clip_id"),
                    "prompt_group": clip.get("prompt_group"),
                    "transcript": clip.get("transcript"),
                    "stage2_score": 0.75,
                }
                for clip in clip_records
            ],
            "support_pairwise_mean_similarity": 1.0,
            "support_pairwise_min_similarity": 1.0,
            "model_variant": self.selected_variant,
            "calibration_latency_ms": 0.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "qwen_weights_updated": False,
            "openwakeword_weights_updated": False,
            "stage2_weights_updated": False,
            "warnings": ["Mock mode uses a deterministic fake prototype."],
        }

    def _stage2_embedding_and_score(self, audio_path: Path) -> dict[str, Any]:
        if self.inference is None:
            raise RuntimeError("runtime not loaded")
        demo_dir = Path("finetune/demo").resolve()
        if str(demo_dir) not in sys.path:
            sys.path.insert(0, str(demo_dir))
        from audio_processing import convert_to_wav, temporary_audio_dir

        with temporary_audio_dir() as work_dir:
            wav_path = convert_to_wav(audio_path, work_dir)
            stage2 = self.inference.runtime.stage2[self.selected_variant]
            with torch.inference_mode():
                features = self.inference.runtime.qwen.extract_audio_features(str(wav_path)).detach().float()
                hidden = features.unsqueeze(0).to(self.inference.runtime.device)
                mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=self.inference.runtime.device)
                output = stage2.model(hidden, mask)
                score = torch.sigmoid(output["logit"]).detach().float().cpu().item()
                embedding = output["embedding"].detach().float().cpu().numpy()[0]
        return {"score": float(score), "embedding": embedding}

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
            trigger_error = None
            try:
                result = self.inference.analyze(audio_path, run_transcript_after_trigger=False)
            except Exception as exc:
                trigger_error = f"{type(exc).__name__}: {exc}"
                result = {
                    "variant": self.selected_variant,
                    "theta_1": self.theta1,
                    "theta_2": self.theta2,
                    "stage1_score": 0.0,
                    "stage2_score": None,
                    "window_table": [],
                    "winning_window": None,
                }
            stage2_score = result.get("stage2_score")
            if stage2_score is None:
                stage2_values = [row.get("Stage 2") for row in result.get("window_table", []) if row.get("Stage 2") is not None]
                stage2_score = max(stage2_values) if stage2_values else None
            calibrated_score = apply_positive_bias(stage2_score, bias)
            candidate = bool(any(row.get("Candidate") for row in result.get("window_table", [])))
            prototype_similarity = None
            prototype_threshold = calibration.get("prototype_threshold")
            prototype_detected = False
            prototype_error = None
            prototype_embedding = calibration.get("prototype_embedding") if calibration.get("calibration_active") else None
            if candidate and prototype_embedding:
                try:
                    query_support = self._stage2_embedding_and_score(audio_path)
                    prototype_similarity = cosine_similarity(query_support["embedding"], prototype_embedding)
                    prototype_threshold = float(prototype_threshold or 0.6)
                    prototype_detected = prototype_similarity >= prototype_threshold
                    if stage2_score is None:
                        stage2_score = float(query_support["score"])
                        calibrated_score = apply_positive_bias(stage2_score, bias)
                except Exception as exc:
                    prototype_error = f"{type(exc).__name__}: {exc}"
            base_trigger = bool(candidate and calibrated_score is not None and calibrated_score >= float(result["theta_2"]))
            trigger_detected = bool(base_trigger or (candidate and prototype_detected))
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
                    "trigger_path_error": trigger_error,
                    "calibration_method": calibration.get("method"),
                    "calibration_support_count": calibration.get("support_count"),
                    "prototype_similarity": prototype_similarity,
                    "prototype_threshold": prototype_threshold,
                    "prototype_detected": prototype_detected,
                    "prototype_error": prototype_error,
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
