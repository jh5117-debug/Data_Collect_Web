#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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


def load_rows(dataset_dir: Path | str, split: str) -> list[dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    if split == "all":
        manifest = dataset_dir / "manifest_all.jsonl"
        if manifest.exists():
            return read_jsonl(manifest)
        rows: list[dict[str, Any]] = []
        for name in ("train", "val", "test"):
            rows.extend(read_jsonl(dataset_dir / f"{name}.jsonl"))
        return rows
    return read_jsonl(dataset_dir / f"{split}.jsonl")


def group_rows_for_evaluation(
    rows: list[dict[str, Any]],
    *,
    evaluation_unit: str,
    deduplicate_by: str = "clip_id",
) -> list[dict[str, Any]]:
    if evaluation_unit == "window":
        return list(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if deduplicate_by not in row:
            raise KeyError(f"deduplicate key {deduplicate_by!r} missing from row")
        grouped[str(row[deduplicate_by])].append(row)
    selected = []
    for key, group in sorted(grouped.items()):
        labels = {int(row["label"]) for row in group}
        if len(labels) != 1:
            raise ValueError(f"deduplicated group {key} has inconsistent labels: {sorted(labels)}")
        wav_paths = {str(row.get("full_wav_path", "")) for row in group}
        if len(wav_paths) != 1:
            raise ValueError(f"deduplicated group {key} has inconsistent full_wav_path values")
        split_values = {str(row.get("split", "")) for row in group}
        if len(split_values) != 1 and any(split_values):
            raise ValueError(f"deduplicated group {key} crosses splits: {sorted(split_values)}")
        first = sorted(group, key=lambda row: int(row.get("window_index", 0)))[0]
        row = dict(first)
        row["deduplicated_rows"] = len(group)
        row["deduplicate_by"] = deduplicate_by
        row["evaluation_unit"] = "clip"
        selected.append(row)
    return selected


def score_exact_predictions(preds: list[dict[str, Any]]) -> dict[str, Any]:
    return binary_metrics([int(p["label"]) for p in preds], [float(p["exact_trigger_decision"]) for p in preds], 0.5)


def per_prompt_metrics(preds: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in preds:
        grouped[str(pred.get("prompt_group", ""))].append(pred)
    return {key: score_exact_predictions(group) for key, group in sorted(grouped.items())}


def confusion_examples(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for pred in preds:
        decision = bool(pred["exact_trigger_decision"])
        label = bool(int(pred["label"]))
        if decision == label:
            continue
        out.append(
            {
                "clip_id": pred.get("clip_id"),
                "split": pred.get("split"),
                "prompt_group": pred.get("prompt_group"),
                "phrase_id": pred.get("phrase_id"),
                "label": int(pred["label"]),
                "exact_trigger_decision": decision,
                "transcript": pred.get("transcript"),
                "predicted_transcript": pred.get("predicted_transcript"),
            }
        )
    return out


def default_output_dir(run_dir: Path, evaluation_unit: str, split: str) -> Path:
    if evaluation_unit == "window":
        suffix = "" if split == "test" else f"_{split}"
        return run_dir / f"baseline_qwen_exact{suffix}"
    suffix = "_clip" if split == "test" else f"_clip_{split}"
    return run_dir / f"baseline_qwen_exact{suffix}"


def write_baseline_outputs(
    out_dir: Path,
    preds: list[dict[str, Any]],
    *,
    model_name: str,
    split: str,
    evaluation_unit: str,
    deduplicate_by: str,
    mean_latency_sec: float | None,
    peak_gpu_memory_gb: float | None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    write_jsonl(out_dir / "predictions.jsonl", preds)
    write_json(out_dir / "per_prompt_metrics.json", per_prompt_metrics(preds))
    write_jsonl(out_dir / "confusion_examples.jsonl", confusion_examples(preds))
    metrics = score_exact_predictions(preds)
    metrics.update(
        {
            "status": status,
            "model_name": model_name,
            "split": split,
            "evaluation_unit": evaluation_unit,
            "deduplicate_by": deduplicate_by if evaluation_unit == "clip" else None,
            "legacy_window_manifest_qwen_baseline": evaluation_unit == "window",
            "metric_scope": "heldout_test" if split == "test" else "full_corpus_diagnostic_not_heldout" if split == "all" else split,
            "rows_evaluated": len(preds),
            "mean_latency_sec": mean_latency_sec,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        }
    )
    if extra:
        metrics.update(extra)
    write_json(out_dir / "metrics.json", metrics)
    report_title = "Qwen Exact Text Baseline"
    if evaluation_unit == "window":
        report_title += " (Legacy Window Manifest)"
    else:
        report_title += " (Clip Level)"
    (out_dir / "report.md").write_text(
        f"# {report_title}\n\n"
        f"- Status: {status}\n"
        f"- Model: `{model_name}`\n"
        f"- Split: `{split}`\n"
        f"- Evaluation unit: `{evaluation_unit}`\n"
        f"- Rows evaluated: {len(preds)}\n"
        f"- Mean latency seconds: {mean_latency_sec}\n"
        f"- Precision: {metrics.get('precision')}\n"
        f"- Recall: {metrics.get('recall')}\n"
        f"- False-positive rate: {metrics.get('false_positive_rate')}\n"
        f"- F1: {metrics.get('f1')}\n",
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--evaluation-unit", choices=["clip", "window"], default="clip")
    parser.add_argument("--deduplicate-by", default="clip_id")
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-skip", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(run_dir, args.evaluation_unit, args.split)
    source_rows = load_rows(args.dataset_dir, args.split)
    try:
        eval_rows = group_rows_for_evaluation(source_rows, evaluation_unit=args.evaluation_unit, deduplicate_by=args.deduplicate_by)
    except Exception as exc:
        status = {
            "status": "blocked",
            "reason": f"baseline row grouping failed: {type(exc).__name__}: {exc}",
            "model_name": args.model_name,
            "split": args.split,
            "evaluation_unit": args.evaluation_unit,
        }
        ensure_dir(out_dir)
        write_json(out_dir / "metrics.json", status)
        return 2
    if not torch.cuda.is_available():
        status = {
            "status": "skipped",
            "reason": "CUDA is not available; unchanged Qwen3-ASR-1.7B text baseline was not run.",
            "model_name": args.model_name,
            "split": args.split,
            "evaluation_unit": args.evaluation_unit,
        }
        ensure_dir(out_dir)
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
            "split": args.split,
            "evaluation_unit": args.evaluation_unit,
        }
        ensure_dir(out_dir)
        write_json(out_dir / "metrics.json", status)
        return 2
    try:
        transcriber = QwenAsrTranscriber(args.model_name)
    except Exception as exc:
        status = {
            "status": "blocked",
            "reason": f"Qwen ASR runtime load failed: {type(exc).__name__}: {exc}",
            "model_name": args.model_name,
            "split": args.split,
            "evaluation_unit": args.evaluation_unit,
        }
        ensure_dir(out_dir)
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
    for row in eval_rows:
        pred = dict(row)
        started = time.perf_counter()
        with torch.inference_mode():
            predicted_text = transcriber.transcribe(Path(row["full_wav_path"]))
        latency = time.perf_counter() - started
        latencies.append(latency)
        pred["predicted_transcript"] = predicted_text
        pred["exact_trigger_decision"] = contains_exact_vigil(pred["predicted_transcript"])
        pred["latency_sec"] = latency
        pred["evaluation_unit"] = args.evaluation_unit
        preds.append(pred)
    write_baseline_outputs(
        out_dir,
        preds,
        model_name=args.model_name,
        split=args.split,
        evaluation_unit=args.evaluation_unit,
        deduplicate_by=args.deduplicate_by,
        mean_latency_sec=float(sum(latencies) / len(latencies)) if latencies else None,
        peak_gpu_memory_gb=float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        extra={"source_manifest_rows": len(source_rows), "unique_audio_transcriptions": len(eval_rows)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

