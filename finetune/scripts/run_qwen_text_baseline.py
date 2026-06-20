#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import torch

from vigil_two_stage.metrics import binary_metrics
from vigil_two_stage.utils import contains_exact_vigil, ensure_dir, read_jsonl, write_json, write_jsonl


def _extract_text(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "transcript", "prediction", "output"):
            if key in result:
                return _extract_text(result[key])
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        return _extract_text(result[0])
    return str(result)


class QwenAsrTranscriber:
    def __init__(self, model_name: str):
        if importlib.util.find_spec("qwen_asr") is None:
            raise RuntimeError("qwen_asr package is not importable in this environment")
        from qwen_asr import Qwen3ASRModel  # type: ignore

        self.model = Qwen3ASRModel.from_pretrained(model_name)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def transcribe(self, wav_path: Path) -> str:
        candidates = []
        if hasattr(self.model, "transcribe"):
            candidates.extend(
                [
                    lambda: self.model.transcribe(str(wav_path)),
                    lambda: self.model.transcribe([str(wav_path)]),
                ]
            )
        if hasattr(self.model, "generate"):
            candidates.append(lambda: self.model.generate(str(wav_path)))
        if callable(self.model):
            candidates.append(lambda: self.model(str(wav_path)))
        errors = []
        for candidate in candidates:
            try:
                return _extract_text(candidate()).strip()
            except Exception as exc:  # pragma: no cover - depends on installed Qwen runtime
                errors.append(f"{type(exc).__name__}: {exc}")
        raise RuntimeError("Qwen ASR transcription failed; attempts: " + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--allow-skip", action="store_true")
    args = parser.parse_args()
    out_dir = ensure_dir(Path(args.run_dir) / "baseline_qwen_exact")
    test_rows = read_jsonl(Path(args.dataset_dir) / "test.jsonl")
    if not torch.cuda.is_available():
        status = {
            "status": "skipped",
            "reason": "CUDA is not available; unchanged Qwen3-ASR-1.7B text baseline was not run.",
            "model_name": args.model_name,
        }
        write_json(out_dir / "metrics.json", status)
        (out_dir / "report.md").write_text(
            "# Qwen Exact Text Baseline\n\n"
            "Status: skipped.\n\n"
            "Reason: CUDA is not available on this host, and Qwen3-ASR-1.7B inference was not run.\n",
            encoding="utf-8",
        )
        return 0 if args.allow_skip else 2
    if torch.cuda.device_count() != 1:
        status = {
            "status": "blocked",
            "reason": f"strict Qwen baseline expected exactly one visible CUDA device, got {torch.cuda.device_count()}",
            "model_name": args.model_name,
        }
        write_json(out_dir / "metrics.json", status)
        return 2
    try:
        transcriber = QwenAsrTranscriber(args.model_name)
    except Exception as exc:
        status = {
            "status": "blocked",
            "reason": f"Qwen ASR runtime load failed: {type(exc).__name__}: {exc}",
            "model_name": args.model_name,
        }
        write_json(out_dir / "metrics.json", status)
        (out_dir / "report.md").write_text(
            "# Qwen Exact Text Baseline\n\n"
            "Status: blocked.\n\n"
            f"Reason: {status['reason']}\n",
            encoding="utf-8",
        )
        return 0 if args.allow_skip else 2

    preds = []
    latencies = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for row in test_rows:
        pred = dict(row)
        started = time.perf_counter()
        with torch.inference_mode():
            predicted_text = transcriber.transcribe(Path(row["full_wav_path"]))
        latency = time.perf_counter() - started
        latencies.append(latency)
        pred["predicted_transcript"] = predicted_text
        pred["exact_trigger_decision"] = contains_exact_vigil(pred["predicted_transcript"])
        pred["latency_sec"] = latency
        preds.append(pred)
    write_jsonl(out_dir / "predictions.jsonl", preds)
    metrics = binary_metrics([p["label"] for p in preds], [float(p["exact_trigger_decision"]) for p in preds], 0.5)
    metrics.update(
        {
            "status": "ok",
            "model_name": args.model_name,
            "mean_latency_sec": float(sum(latencies) / len(latencies)) if latencies else None,
            "peak_gpu_memory_gb": float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        }
    )
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "report.md").write_text(
        "# Qwen Exact Text Baseline\n\n"
        f"- Status: ok\n"
        f"- Model: `{args.model_name}`\n"
        f"- Clips: {len(preds)}\n"
        f"- Mean latency seconds: {metrics['mean_latency_sec']}\n"
        f"- Recall: {metrics.get('recall')}\n"
        f"- False-positive rate: {metrics.get('false_positive_rate')}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
