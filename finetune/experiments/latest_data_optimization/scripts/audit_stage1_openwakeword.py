#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from vigil_two_stage.stage1_model import Stage1GRUClassifier, count_parameters
from vigil_latest_opt.utils import read_json, sha256_file, write_json


ROOT = Path("finetune/experiments/latest_data_optimization")
REPORTS = ROOT / "reports"
COMPUTE_REPORT = REPORTS / "latest_opt_compute_cost.json"
FINAL_MANIFEST = REPORTS / "latest_opt_final_model_manifest.json"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def onnx_shape_info(path: Path) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return {"status": "unavailable", "reason": f"onnxruntime import failed: {type(exc).__name__}: {exc}"}
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return {
            "status": "ok",
            "inputs": [{"name": i.name, "shape": list(i.shape), "type": i.type} for i in sess.get_inputs()],
            "outputs": [{"name": o.name, "shape": list(o.shape), "type": o.type} for o in sess.get_outputs()],
            "providers": sess.get_providers(),
        }
    except Exception as exc:
        return {"status": "unavailable", "reason": f"onnxruntime session failed: {type(exc).__name__}: {exc}"}


def component_parameters(model: Stage1GRUClassifier) -> list[dict[str, Any]]:
    return [
        {"component": "Stage 1 LayerNorm", "parameters": count_parameters(model.norm)["total"], "frozen": False, "trainable": True},
        {"component": "Stage 1 2-layer GRU", "parameters": count_parameters(model.gru)["total"], "frozen": False, "trainable": True},
        {"component": "Stage 1 Linear classifier", "parameters": count_parameters(model.classifier)["total"], "frozen": False, "trainable": True},
        {"component": "Stage 1 total head", "parameters": count_parameters(model)["total"], "frozen": False, "trainable": True},
    ]


def load_openwakeword_assets() -> dict[str, Any]:
    try:
        from openwakeword.utils import AudioFeatures
    except Exception as exc:
        return {
            "status": "unavailable",
            "version": package_version("openwakeword"),
            "reason": f"openwakeword AudioFeatures import failed: {type(exc).__name__}: {exc}",
        }
    signature = inspect.signature(AudioFeatures)
    assets = []
    for name in ("melspec_onnx_model_path", "embedding_onnx_model_path"):
        param = signature.parameters.get(name)
        if param is None or param.default is inspect.Signature.empty:
            continue
        path = Path(str(param.default))
        assets.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
                "shape_info": onnx_shape_info(path) if path.exists() else {"status": "missing"},
            }
        )
    return {
        "status": "ok",
        "version": package_version("openwakeword"),
        "source_file": inspect.getsourcefile(AudioFeatures),
        "signature": str(signature),
        "embed_clips_signature": str(inspect.signature(AudioFeatures.embed_clips)),
        "assets": assets,
    }


def component_latency() -> dict[str, Any]:
    if not COMPUTE_REPORT.exists():
        return {"status": "missing", "reason": f"{COMPUTE_REPORT} does not exist"}
    compute = read_json(COMPUTE_REPORT)
    by_component = {row["component"]: row for row in compute.get("components", [])}
    oww = by_component.get("official_openwakeword_feature_extraction", {})
    head = by_component.get("stage1_head", {})
    cached = by_component.get("stage1_cached_feature_load_plus_head", {})
    median_sum = None
    p95_sum = None
    if oww.get("median_ms") is not None and head.get("median_ms") is not None:
        median_sum = float(oww["median_ms"]) + float(head["median_ms"])
    if oww.get("p95_ms") is not None and head.get("p95_ms") is not None:
        p95_sum = float(oww["p95_ms"]) + float(head["p95_ms"])
    return {
        "status": "ok",
        "source": str(COMPUTE_REPORT),
        "official_openwakeword_feature_extraction": oww,
        "stage1_head": head,
        "stage1_cached_feature_load_plus_head": cached,
        "full_stage1_component_sum_estimate": {"median_ms": median_sum, "p95_ms": p95_sum},
    }


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    input_dim = 96
    model = Stage1GRUClassifier(input_dim=input_dim, hidden_size=64, layers=2, dropout=0.10)
    total = count_parameters(model)
    assets = load_openwakeword_assets()
    latency = component_latency()
    final = read_json(FINAL_MANIFEST) if FINAL_MANIFEST.exists() else {}
    theta1 = final.get("stage1_threshold", {}).get("threshold")
    summary = {
        "status": "ok",
        "stage1_not_qwen": True,
        "stage1_not_lora": True,
        "input_sample_rate_hz": 16000,
        "openwakeword": assets,
        "head_architecture": "LayerNorm -> 2-layer unidirectional GRU -> Linear trigger logit",
        "head_input_dim": input_dim,
        "head_parameters": total,
        "component_parameters": component_parameters(model),
        "loss": "weighted BCEWithLogitsLoss",
        "output": "p1 = sigmoid(trigger logit)",
        "candidate_rule": "p1 >= theta_1",
        "latest_deployment_theta_1": theta1,
        "latency": latency,
        "qwen_comparison": {
            "qwen_model": "Qwen/Qwen3-ASR-1.7B",
            "qwen_parameter_count_from_latest_cost": 2038052480,
            "stage1_head_parameter_count": total["total"],
            "stage1_produces_transcript": False,
        },
    }
    write_json(REPORTS / "stage1_openwakeword_structure.json", summary)

    oww_rows = []
    if assets.get("status") == "ok":
        for asset in assets.get("assets", []):
            size_kib = asset["size_bytes"] / 1024.0 if asset.get("size_bytes") is not None else None
            oww_rows.append(
                f"| `{Path(asset['path']).name}` | yes | N/A (ONNX runtime asset) | no | official openWakeWord feature asset, {size_kib:.1f} KiB |"
            )
    else:
        oww_rows.append("| openWakeWord AudioFeatures | yes | N/A | no | package import failed |")

    param_rows = [
        "| Component | Frozen? | Parameters | Trainable? | Notes |",
        "|---|---:|---:|---:|---|",
        *oww_rows,
        f"| Stage 1 LayerNorm | no | {component_parameters(model)[0]['parameters']} | yes | normalizes 96-d openWakeWord embeddings |",
        f"| Stage 1 GRU | no | {component_parameters(model)[1]['parameters']} | yes | 2-layer unidirectional GRU, hidden size 64 |",
        f"| Stage 1 Linear | no | {component_parameters(model)[2]['parameters']} | yes | maps final GRU state to one trigger logit |",
        f"| Stage 1 total head | no | {total['total']} | yes | verified from PyTorch module |",
    ]
    lat = latency.get("full_stage1_component_sum_estimate", {})
    oww = latency.get("official_openwakeword_feature_extraction", {})
    head = latency.get("stage1_head", {})
    cached = latency.get("stage1_cached_feature_load_plus_head", {})
    report = [
        "# Stage 1 openWakeWord Structure Report",
        "",
        "## Simple English Summary",
        "",
        "Stage 1 is a lightweight wake-word candidate detector. It is not Qwen, not LoRA, and it does not produce a transcript. It reads 16 kHz audio, uses the official openWakeWord audio feature extractor as a frozen front-end, and trains only a small VIGIL-specific PyTorch head. The head outputs `p1`, the probability that this audio window should become a VIGIL candidate. The candidate rule is `p1 >= theta_1`.",
        "",
        "## Technical Summary",
        "",
        "- Input: 16 kHz mono audio windows.",
        "- Preprocessing: official openWakeWord `AudioFeatures` computes mel features and 96-dimensional speech embeddings with ONNX Runtime.",
        "- Frozen component: openWakeWord `melspectrogram.onnx` and `embedding_model.onnx` assets are used as feature extractors; they are not trained by this project.",
        "- Trainable component: `Stage1GRUClassifier`, implemented as `LayerNorm -> 2-layer GRU -> Linear`.",
        "- Loss: weighted `BCEWithLogitsLoss` through the local wrapper used by `train_stage1.py`.",
        "- Output: one trigger logit per window; `p1 = sigmoid(logit)`.",
        "- Role: high-recall candidate detector that cheaply reduces how often Stage 2 must run.",
        "- Difference from Qwen: Qwen3-ASR-1.7B produces transcript and high-dimensional audio states; Stage 1 has only a 56k trainable head and no decoder.",
        "",
        "## Architecture Diagram",
        "",
        "```text",
        "Microphone 16 kHz audio",
        "    -> openWakeWord preprocessing",
        "    -> frozen openWakeWord melspectrogram ONNX",
        "    -> frozen openWakeWord embedding ONNX",
        "    -> embeddings [T, 96]",
        "    -> trainable LayerNorm + 2-layer GRU + Linear",
        "    -> p1 candidate score",
        "    -> candidate if p1 >= theta_1",
        "```",
        "",
        "## Parameter Count",
        "",
        *param_rows,
        "",
        "## Latency",
        "",
        "| Component | n | Median ms | p95 ms | Source |",
        "|---|---:|---:|---:|---|",
        f"| official openWakeWord feature extraction | {oww.get('n')} | {oww.get('median_ms')} | {oww.get('p95_ms')} | latest compute report |",
        f"| Stage 1 head | {head.get('n')} | {head.get('median_ms')} | {head.get('p95_ms')} | latest compute report |",
        f"| cached feature load + Stage 1 head | {cached.get('n')} | {cached.get('median_ms')} | {cached.get('p95_ms')} | latest compute report |",
        f"| full Stage 1 component-sum estimate | - | {lat.get('median_ms')} | {lat.get('p95_ms')} | openWakeWord feature + head sum |",
        "",
        "## Comparison To Continuous Qwen ASR",
        "",
        "| Component | Primary role | Produces transcript? | Approx parameters | Runtime role |",
        "|---|---|---:|---:|---|",
        f"| Stage 1 VIGIL head | cheap candidate detector | no | {total['total']} trainable head params | runs before Stage 2 |",
        "| Qwen3-ASR-1.7B | continuous clinical ASR and Stage 2 audio features | yes | 2,038,052,480 frozen params | main transcript branch and verifier features |",
        "",
        "Stage 1 is designed to be high-recall and lightweight. Stage 2 then uses frozen Qwen audio features to reject most false candidates.",
        "",
    ]
    (REPORTS / "STAGE1_OPENWAKEWORD_STRUCTURE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(REPORTS / "STAGE1_OPENWAKEWORD_STRUCTURE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
