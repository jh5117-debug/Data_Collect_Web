#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import model_info

from vigil_two_stage.qwen_text_result import extract_qwen_text
from vigil_two_stage.utils import contains_exact_vigil
from vigil_participant_cv.utils import ensure_dir, read_jsonl, write_json, write_jsonl


def _model_parameter_module(obj: Any) -> torch.nn.Module | None:
    if hasattr(obj, "named_parameters"):
        return obj
    nested = getattr(obj, "model", None)
    if nested is not None and hasattr(nested, "named_parameters"):
        return nested
    return None


class QwenClipTranscriber:
    def __init__(self, model_name: str):
        if importlib.util.find_spec("qwen_asr") is None:
            raise RuntimeError("qwen_asr package is not importable")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; refusing CPU Qwen inference")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")
        from qwen_asr import Qwen3ASRModel  # type: ignore

        self.model_name = model_name
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model_revision = model_info(model_name).sha
        self.model = Qwen3ASRModel.from_pretrained(model_name, max_new_tokens=1024, torch_dtype=self.dtype)
        if hasattr(self.model, "eval"):
            self.model.eval()
        module = _model_parameter_module(self.model)
        if module is None:
            raise RuntimeError("Qwen object does not expose named_parameters")
        module.to("cuda:0")
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        self.total_parameters = sum(p.numel() for p in module.parameters())
        self.trainable_parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if self.trainable_parameters:
            raise RuntimeError(f"Qwen must remain frozen, got {self.trainable_parameters} trainable parameters")

    def transcribe(self, wav_path: str) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            raw = self.model.transcribe(str(wav_path), language=None)
            extracted = extract_qwen_text(raw)
        latency = time.perf_counter() - started
        peak = float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None
        return {
            "predicted_transcript": extracted.text,
            "text_extraction_path": extracted.extraction_path,
            "qwen_result_type": extracted.result_type,
            "latency_sec": latency,
            "peak_gpu_memory_gb": peak,
        }


def unique_clip_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clips: dict[str, dict[str, Any]] = {}
    for row in rows:
        clip_id = str(row["clip_id"])
        clips.setdefault(clip_id, row)
    return [clips[key] for key in sorted(clips)]


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("text_extraction_path") == "$[0].text" and "ASRTranscription(" not in str(row.get("predicted_transcript", "")):
            existing[str(row["clip_id"])] = row
    return existing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-manifest", default="finetune/experiments/participant_cv/shared/balanced_max100_manifest.jsonl")
    parser.add_argument("--output", default="finetune/experiments/participant_cv/shared/qwen_transcript_cache_balanced_max100.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen3-ASR-1.7B")
    args = parser.parse_args()
    rows = unique_clip_rows(read_jsonl(args.balanced_manifest))
    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    existing = load_existing(out_path)
    pending = [row for row in rows if str(row["clip_id"]) not in existing]
    transcriber = None if not pending else QwenClipTranscriber(args.model_name)
    all_rows = dict(existing)
    for idx, row in enumerate(pending, start=1):
        result = transcriber.transcribe(str(row["full_wav_path"]))  # type: ignore[union-attr]
        pred = {
            "clip_id": row["clip_id"],
            "participant_alias": row["participant_alias"],
            "label": int(row["label"]),
            "prompt_group": row.get("prompt_group"),
            "phrase_id": row.get("phrase_id"),
            "reference_transcript": row.get("transcript"),
            "full_wav_sha256": row.get("full_wav_sha256"),
            "model_name": args.model_name,
            "model_revision": transcriber.model_revision,  # type: ignore[union-attr]
            "qwen_dtype": str(transcriber.dtype),  # type: ignore[union-attr]
            "qwen_total_parameters": transcriber.total_parameters,  # type: ignore[union-attr]
            "qwen_trainable_parameters": transcriber.trainable_parameters,  # type: ignore[union-attr]
            "exact_trigger_decision": contains_exact_vigil(result["predicted_transcript"]),
            **result,
        }
        all_rows[str(row["clip_id"])] = pred
        if idx % 25 == 0 or idx == len(pending):
            write_jsonl(out_path, [all_rows[key] for key in sorted(all_rows)])
            print(json.dumps({"completed": len(all_rows), "total": len(rows), "last_clip_id": row["clip_id"]}, sort_keys=True), flush=True)
    write_jsonl(out_path, [all_rows[key] for key in sorted(all_rows)])
    summary = {
        "status": "ok",
        "output": str(out_path),
        "clips": len(rows),
        "cached": len(all_rows),
        "pending": len(rows) - len(all_rows),
        "model_name": args.model_name,
        "model_revision": next(iter(all_rows.values())).get("model_revision") if all_rows else None,
        "text_extraction_path": sorted({row.get("text_extraction_path") for row in all_rows.values()}),
        "result_type": sorted({row.get("qwen_result_type") for row in all_rows.values()}),
    }
    write_json(out_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["pending"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
