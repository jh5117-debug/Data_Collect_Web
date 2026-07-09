from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def final_trigger(stage1_accept: bool, stage2_accept: bool) -> bool:
    return bool(stage1_accept and stage2_accept)


def validate_final_trigger(row: dict[str, Any]) -> bool:
    expected = final_trigger(bool(row.get("stage1_accept")), bool(row.get("stage2_accept")))
    return bool(row.get("final_trigger")) == expected


def score_range(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return float(max(scores) - min(scores))


def detect_constant_stage2_scores(scores: list[float], *, tolerance: float = 1e-4) -> dict[str, Any]:
    values = [float(score) for score in scores if score is not None]
    return {
        "n": len(values),
        "constant": len(values) >= 2 and score_range(values) <= tolerance,
        "range": score_range(values),
        "mean": statistics.fmean(values) if values else None,
        "tolerance": tolerance,
    }


def detect_identical_hashes(hashes: list[str]) -> dict[str, Any]:
    values = [value for value in hashes if value]
    unique = sorted(set(values))
    return {
        "n": len(values),
        "unique": len(unique),
        "identical": len(values) >= 2 and len(unique) == 1,
        "unique_hashes": unique,
    }


def short_hash_bytes(data: bytes, length: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:length]


def _tensor_hash(tensor: Any) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy()
    return short_hash_bytes(array.tobytes())


def _load_demo_runtime(model_run_dir: Path) -> Any:
    demo_dir = Path("finetune/demo").resolve()
    if str(demo_dir) not in sys.path:
        sys.path.insert(0, str(demo_dir))
    from model_loader import load_runtime

    return load_runtime(model_run_dir)


def _score_case(runtime: Any, case: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    wav_path = Path(case["wav_path"])
    started = time.perf_counter()
    openwakeword_features = runtime.openwakeword.extract(wav_path)
    x = torch.from_numpy(openwakeword_features.astype(np.float32)).unsqueeze(0).to(runtime.device)
    lengths = torch.tensor([openwakeword_features.shape[0]], dtype=torch.long, device=runtime.device)
    with torch.inference_mode():
        stage1_score = torch.sigmoid(runtime.stage1.model(x, lengths)).detach().float().cpu().item()
        qwen_features = runtime.qwen.extract_audio_features(str(wav_path)).detach().float()
        stage2 = runtime.stage2[runtime.selected_variant]
        hidden = qwen_features.unsqueeze(0).to(runtime.device)
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=runtime.device)
        output = stage2.model(hidden, mask)
        stage2_score = torch.sigmoid(output["logit"]).detach().float().cpu().item()
        embedding = output["embedding"].detach().float()
    latency_ms = (time.perf_counter() - started) * 1000.0
    theta1 = float(runtime.stage1.theta)
    theta2 = float(runtime.stage2[runtime.selected_variant].theta)
    stage1_accept = float(stage1_score) >= theta1
    stage2_accept = float(stage2_score) >= theta2
    return {
        **case,
        "stage1_score": float(stage1_score),
        "stage2_score": float(stage2_score),
        "final_trigger": final_trigger(stage1_accept, stage2_accept),
        "theta_1": theta1,
        "theta_2": theta2,
        "stage1_accept": stage1_accept,
        "stage2_accept": stage2_accept,
        "checkpoint_path": {
            "stage1": str(Path(runtime.run_dir) / "stage1" / "checkpoint_best.pt"),
            "stage2": str(Path(runtime.run_dir) / runtime.stage2[runtime.selected_variant].variant_dir / "checkpoint_best.pt"),
        },
        "model_config_path": str(Path(runtime.run_dir) / "stage1" / "model_config.json"),
        "selected_variant": runtime.selected_variant,
        "audio_window_start_sec": case.get("window_start_sec"),
        "audio_window_end_sec": case.get("window_end_sec"),
        "audio_hash": short_hash_bytes(wav_path.read_bytes()),
        "openwakeword_feature_hash": short_hash_bytes(np.asarray(openwakeword_features, dtype=np.float32).tobytes()),
        "feature_hash": _tensor_hash(qwen_features),
        "embedding_hash": _tensor_hash(embedding),
        "qwen_feature_norm": float(torch.linalg.vector_norm(qwen_features.float()).detach().cpu().item()),
        "stage2_embedding_norm": float(torch.linalg.vector_norm(embedding.float()).detach().cpu().item()),
        "latency_ms": float(latency_ms),
    }


def score_manifest_with_current_detector(manifest_path: Path, model_run_dir: Path) -> dict[str, Any]:
    rows = load_jsonl(manifest_path)
    if not rows:
        return blocked_score_audit("Decoded manifest exists but contains no rows.", manifest_path)
    try:
        runtime = _load_demo_runtime(Path(model_run_dir))
        score_rows = [_score_case(runtime, row) for row in rows]
    except Exception as exc:
        return blocked_score_audit(f"Detector scoring failed: {type(exc).__name__}: {exc}", manifest_path)
    return {
        "status": "ok",
        "reason": "Scored decoded rosbag WAV windows with the current VIGIL two-stage detector.",
        "manifest_path": str(manifest_path),
        "model_run_dir": str(model_run_dir),
        "score_rows": score_rows,
        "diagnosis": diagnose_score_rows(score_rows),
    }


def diagnose_score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_logic_failures = [row.get("case_id") for row in rows if not validate_final_trigger(row)]
    false_accepts = [
        row.get("case_id")
        for row in rows
        if int(row.get("expected_label", 0)) == 0 and bool(row.get("final_trigger"))
    ]
    false_rejects = [
        row.get("case_id")
        for row in rows
        if int(row.get("expected_label", 0)) == 1 and not bool(row.get("final_trigger"))
    ]
    stage2_scores = [float(row["stage2_score"]) for row in rows if row.get("stage2_score") is not None]
    stage2_negative_accepts = [
        row.get("case_id")
        for row in rows
        if int(row.get("expected_label", 0)) == 0 and bool(row.get("stage2_accept"))
    ]
    feature_hashes = [str(row.get("feature_hash") or "") for row in rows]
    embedding_hashes = [str(row.get("embedding_hash") or "") for row in rows]
    constant_scores = detect_constant_stage2_scores(stage2_scores)
    identical_features = detect_identical_hashes(feature_hashes)
    identical_embeddings = detect_identical_hashes(embedding_hashes)
    if final_logic_failures:
        diagnosis = "cascade_decision_bug"
    elif constant_scores["constant"] and identical_features["identical"]:
        diagnosis = "integration_cache_or_window_bug"
    elif constant_scores["constant"] and not identical_features["identical"]:
        diagnosis = "possible_model_calibration_or_bias"
    elif false_accepts:
        diagnosis = "heldout_false_positive_model_or_threshold_issue"
    elif rows:
        diagnosis = "no_constant_score_bug_observed"
    else:
        diagnosis = "not_run_no_scored_audio"
    return {
        "diagnosis": diagnosis,
        "rows": len(rows),
        "final_logic_failures": final_logic_failures,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "stage2_negative_accepts": stage2_negative_accepts,
        "stage2_score_constant_check": constant_scores,
        "feature_hash_check": identical_features,
        "embedding_hash_check": identical_embeddings,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def blocked_score_audit(reason: str, manifest_path: Path | None = None) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "score_rows": [],
        "diagnosis": diagnose_score_rows([]),
    }
