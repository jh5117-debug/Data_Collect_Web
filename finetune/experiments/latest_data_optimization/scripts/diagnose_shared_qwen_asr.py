#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any

import torch

from vigil_two_stage.qwen_audio_adapter import FrozenQwenAudioAdapter
from vigil_two_stage.qwen_text_result import extract_qwen_text
from vigil_two_stage.shared_qwen_adapter import MethodCallCounter, blocker_text
from vigil_latest_opt.utils import read_json, read_jsonl, write_json


REPORTS = Path("finetune/experiments/latest_data_optimization/reports")
BALANCED = Path("finetune/experiments/latest_data/shared/balanced_max100_latest_manifest.jsonl")
COMPUTE = REPORTS / "latest_opt_compute_cost.json"


def pick_audio() -> str:
    for row in read_jsonl(BALANCED):
        if int(row["label"]) == 1 and row.get("prompt_group") == "P1_vigil_only":
            return str(row["window_wav_path"])
    return str(read_jsonl(BALANCED)[0]["window_wav_path"])


def source_diagnostic() -> dict[str, Any]:
    try:
        import importlib.metadata as md
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
    except Exception as exc:
        return {"status": "qwen_asr_unavailable", "error": f"{type(exc).__name__}: {exc}"}
    src = Path(inspect.getsourcefile(Qwen3ASRModel) or "")
    public = [name for name in dir(Qwen3ASRModel) if not name.startswith("_")]
    text = src.read_text(encoding="utf-8", errors="ignore") if src.exists() else ""
    return {
        "status": "ok",
        "package_version": md.version("qwen-asr"),
        "source_file": str(src),
        "transcribe_signature": str(inspect.signature(Qwen3ASRModel.transcribe)),
        "from_pretrained_signature": str(inspect.signature(Qwen3ASRModel.from_pretrained)),
        "public_methods": sorted(public),
        "transcribe_uses_model_generate": "self.model.generate" in text,
        "public_methods_with_hidden": [name for name in public if "hidden" in name.lower() or "feature" in name.lower()],
    }


def model_call_diagnostic(audio_path: str, model_name: str) -> dict[str, Any]:
    from qwen_asr import Qwen3ASRModel

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    start_load = time.perf_counter()
    wrapper = Qwen3ASRModel.from_pretrained(model_name, torch_dtype=dtype, max_inference_batch_size=1, max_new_tokens=128)
    load_sec = time.perf_counter() - start_load
    wrapper.model.eval()
    for param in wrapper.model.parameters():
        param.requires_grad = False
    counter = MethodCallCounter()
    counter.patch(wrapper, "transcribe", "transcribe")
    counter.patch(wrapper.model, "generate", "generate")
    thinker = getattr(wrapper.model, "thinker", None)
    counter.patch(thinker, "forward", "thinker_forward")
    counter.patch(thinker, "get_audio_features", "get_audio_features")
    try:
        start = time.perf_counter()
        raw = wrapper.transcribe(audio_path, language=None, return_time_stamps=False)
        transcribe_latency = time.perf_counter() - start
        extracted = extract_qwen_text(raw)
        after_transcribe = dict(counter.counts)
        adapter = FrozenQwenAudioAdapter(model_name)
        adapter._set_loaded_model(wrapper)
        start = time.perf_counter()
        hidden = adapter.extract_audio_features(audio_path)
        feature_latency = time.perf_counter() - start
        after_features = dict(counter.counts)
    finally:
        counter.restore()
    return {
        "status": "ok",
        "model_load_count": 1,
        "model_name": model_name,
        "dtype": str(dtype),
        "load_latency_sec": load_sec,
        "audio_path_sha256_12": hashlib.sha256(audio_path.encode("utf-8")).hexdigest()[:12],
        "transcript": extracted.text,
        "transcript_result_type": extracted.result_type,
        "transcript_extraction_path": extracted.extraction_path,
        "transcribe_latency_sec": transcribe_latency,
        "feature_latency_sec": feature_latency,
        "hidden_shape": list(hidden.shape),
        "feature_extraction_path": adapter.extraction_path,
        "call_counts_after_transcribe": after_transcribe,
        "call_counts_after_feature_extraction": after_features,
    }


def write_report(summary: dict[str, Any]) -> None:
    cost = read_json(COMPUTE) if COMPUTE.exists() else {}
    components = {row["component"]: row for row in cost.get("components", [])}
    qwen_encoder = components.get("qwen_audio_encoder_forward", {})
    model_diag = summary.get("model_call_diagnostic", {})
    source = summary.get("source_diagnostic", {})
    current_latency = qwen_encoder.get("median_ms")
    shared_latency = None
    status = summary["status"]
    report = [
        "# Shared Qwen-ASR Hidden-State Report",
        "",
        "## Status",
        "",
        f"- Status: `{status}`",
        f"- Blocker: {summary.get('blocker')}",
        "",
        "## Evidence",
        "",
        f"- qwen-asr version: `{source.get('package_version')}`",
        f"- Source file inspected: `{source.get('source_file')}`",
        f"- Public transcribe signature: `{source.get('transcribe_signature')}`",
        f"- Public methods exposing hidden/features: `{source.get('public_methods_with_hidden')}`",
        f"- `transcribe` uses model generation internally: `{source.get('transcribe_uses_model_generate')}`",
        "",
        "## Call Counter Diagnostic",
        "",
        f"- Model load count: `{model_diag.get('model_load_count')}`",
        f"- Counts after public transcribe: `{model_diag.get('call_counts_after_transcribe')}`",
        f"- Counts after separate Stage 2 feature extraction: `{model_diag.get('call_counts_after_feature_extraction')}`",
        f"- Transcript extraction path: `{model_diag.get('transcript_extraction_path')}`",
        f"- Stage 2 feature path: `{model_diag.get('feature_extraction_path')}`",
        f"- Hidden shape: `{model_diag.get('hidden_shape')}`",
        "",
        "## Cost Table",
        "",
        "| Variant | Qwen copies | Encoder forwards | Transcript available? | Stage2 score available? | Median latency | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| Current prototype | 1 | extra encoder forward per Stage 1 candidate | yes | yes | {current_latency} ms extra encoder median | working |",
        f"| Shared hidden-state prototype | 1 | 1 only if upstream exposes handoff | no verified handoff | no verified handoff | {shared_latency} | {status} |",
        "",
        "## Professor Wording",
        "",
        "The current public Qwen ASR wrapper does not expose a reusable hidden-state handoff. Therefore the current prototype still needs an extra encoder forward for Stage 2 candidates. The measured median extra cost is around 13.66 ms per Stage 1 candidate from the latest compute report.",
        "",
    ]
    (REPORTS / "SHARED_QWEN_ASR_HIDDEN_STATE_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--audio-path")
    parser.add_argument("--skip-model-load", action="store_true")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    source = source_diagnostic()
    audio_path = args.audio_path or pick_audio()
    model_diag: dict[str, Any]
    if args.skip_model_load:
        model_diag = {"status": "skipped"}
    else:
        model_diag = model_call_diagnostic(audio_path, args.model)
    feature_path = model_diag.get("feature_extraction_path") or "model.thinker.get_audio_features"
    blocker = blocker_text(str(source.get("transcribe_signature")), str(feature_path))
    summary = {
        "status": "blocked_by_runtime_interface",
        "source_diagnostic": source,
        "model_call_diagnostic": model_diag,
        "blocker": blocker,
        "minimum_upstream_interface_needed": "Expose a public one-forward call that returns ASR decoder-compatible audio hidden states and accepts those same states for decoding, or returns both transcript and reusable audio encoder states.",
    }
    write_json(REPORTS / "shared_qwen_asr_diagnostic.json", summary)
    write_report(summary)
    print(json.dumps({"status": summary["status"], "model_diag": model_diag.get("status")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
