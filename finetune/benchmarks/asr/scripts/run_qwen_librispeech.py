#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from normalization import normalize_librispeech_text
from qwen_runner import QwenASRRunner, QwenRunnerError
from resume import append_prediction, deduplicate_predictions, load_prediction_state, update_progress
from scoring import score_pairs, score_prediction_rows
from utils import (
    BENCHMARK_ROOT,
    command_output,
    ensure_dir,
    environment_snapshot,
    read_jsonl,
    safe_run_name,
    utc_timestamp,
    write_json,
    write_jsonl,
)


def _run_dir(run_name: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return ensure_dir(output_dir)
    return ensure_dir(BENCHMARK_ROOT / "runs" / f"{utc_timestamp()}_{safe_run_name(run_name)}")


def _latency_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = [float(row["latency_sec"]) for row in rows if row.get("status") == "success" and row.get("latency_sec") is not None]
    rtfs = [float(row["real_time_factor"]) for row in rows if row.get("status") == "success" and row.get("real_time_factor") is not None]
    peaks = [float(row["peak_gpu_memory_gb"]) for row in rows if row.get("status") == "success" and row.get("peak_gpu_memory_gb") is not None]
    return {
        "count": len(values),
        "mean_latency_sec": statistics.fmean(values) if values else None,
        "median_latency_sec": statistics.median(values) if values else None,
        "max_latency_sec": max(values) if values else None,
        "mean_real_time_factor": statistics.fmean(rtfs) if rtfs else None,
        "max_peak_gpu_memory_gb": max(peaks) if peaks else None,
    }


def _group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {name: score_prediction_rows(group_rows, normalized=True) for name, group_rows in sorted(grouped.items())}


def _error_rows(rows: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "success":
            continue
        metrics = score_pairs([str(row.get("normalized_reference", ""))], [str(row.get("normalized_hypothesis", ""))])
        errors = int(metrics["substitutions"] or 0) + int(metrics["deletions"] or 0) + int(metrics["insertions"] or 0)
        if errors:
            out.append(
                {
                    "id": row.get("id"),
                    "split": row.get("split"),
                    "reference": row.get("reference"),
                    "hypothesis": row.get("hypothesis"),
                    "normalized_reference": row.get("normalized_reference"),
                    "normalized_hypothesis": row.get("normalized_hypothesis"),
                    "wer": metrics["wer"],
                    "substitutions": metrics["substitutions"],
                    "deletions": metrics["deletions"],
                    "insertions": metrics["insertions"],
                }
            )
    out.sort(key=lambda item: (float(item["wer"] or 0.0), str(item["id"])), reverse=True)
    return out[:limit]


def _write_error_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = [
        "id",
        "split",
        "reference",
        "hypothesis",
        "normalized_reference",
        "normalized_hypothesis",
        "wer",
        "substitutions",
        "deletions",
        "insertions",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _report_markdown(path: Path, payload: dict[str, Any]) -> None:
    normalized = payload["metrics"]["normalized"]
    raw = payload["metrics"]["raw"]
    lines = [
        "# LibriSpeech Qwen ASR Benchmark Report",
        "",
        f"Run name: `{payload['run_name']}`",
        f"Model: `{payload['model_name']}`",
        f"Manifest: `{payload['manifest_path']}`",
        f"Samples requested: {payload['samples_requested']}",
        f"Successful predictions: {payload['successful_predictions']}",
        f"Failed predictions: {payload['failed_predictions']}",
        "",
    ]
    if payload.get("smoke_only"):
        lines.extend(
            [
                "**SMOKE SUBSET — NOT FULL LIBRISPEECH RESULT.**",
                "",
            ]
        )
    lines.extend(
        [
            "## Main Metrics",
            "",
            f"Normalized WER: {normalized.get('wer')}",
            f"Raw WER: {raw.get('wer')}",
            f"Normalized CER: {normalized.get('cer')}",
            f"Sentence error rate: {normalized.get('sentence_error_rate')}",
            f"Exact-match rate: {normalized.get('exact_match_rate')}",
            "",
            "## Latency",
            "",
            f"Mean latency seconds: {payload['latency'].get('mean_latency_sec')}",
            f"Mean real-time factor: {payload['latency'].get('mean_real_time_factor')}",
            f"Peak GPU memory GB: {payload['latency'].get('max_peak_gpu_memory_gb')}",
            "",
            "## Notes",
            "",
            "- Qwen weights are loaded in eval mode and are not fine-tuned.",
            "- No VIGIL detector, openWakeWord stage, Stage 2 verifier, or wake-word gate is used in this benchmark.",
            "- WER is calculated with transparent local Levenshtein scoring.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _score_and_write(run_dir: Path, manifest_rows: list[dict[str, Any]], predictions_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    deduplicate_predictions(predictions_path)
    rows = read_jsonl(predictions_path)
    successful = [row for row in rows if row.get("status") == "success"]
    failed = [row for row in rows if row.get("status") == "failed"]
    normalized_metrics = score_prediction_rows(successful, normalized=True)
    raw_metrics = score_prediction_rows(successful, normalized=False)
    per_split_metrics = _group_metrics(successful, "split")
    per_speaker_metrics = _group_metrics(successful, "speaker_id")
    metrics = {
        "normalized": normalized_metrics,
        "raw": raw_metrics,
        "per_split": per_split_metrics,
        "per_speaker": per_speaker_metrics,
    }
    payload = {
        "run_name": args.run_name,
        "model_name": args.model,
        "manifest_path": str(args.manifest.resolve()),
        "created_at_utc": utc_timestamp(),
        "samples_requested": len(manifest_rows),
        "prediction_rows": len(rows),
        "successful_predictions": len(successful),
        "failed_predictions": len(failed),
        "smoke_only": "smoke" in args.manifest.name or len(manifest_rows) < 5559,
        "metrics": metrics,
        "latency": _latency_stats(successful),
        "failure_examples": failed[:20],
        "environment": environment_snapshot(),
    }
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "metrics_raw.json", raw_metrics)
    write_json(run_dir / "metrics_normalized.json", normalized_metrics)
    write_json(run_dir / "per_split_metrics.json", per_split_metrics)
    write_json(run_dir / "per_speaker_metrics.json", per_speaker_metrics)
    write_jsonl(run_dir / "failures.jsonl", failed)
    _write_error_csv(run_dir / "error_analysis.csv", _error_rows(successful))
    write_json(
        run_dir / "reproducibility.json",
        {
            "created_at_utc": payload["created_at_utc"],
            "manifest_path": payload["manifest_path"],
            "model_name": args.model,
            "run_name": args.run_name,
            "seed": args.seed,
            "config": {
                "batch_size": args.batch_size,
                "backend": args.backend,
                "dtype": args.dtype,
                "language": args.language,
                "limit": args.limit,
                "max_new_tokens": args.max_new_tokens,
                "resume": args.resume,
                "retry_failed": args.retry_failed,
            },
            "environment": payload["environment"],
        },
    )
    _report_markdown(run_dir / "FINAL_REPORT.md", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR on a LibriSpeech manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="qwen3_asr_1_7b_smoke")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--backend", default="transformers")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--require-baseline-model", action="store_true")
    args = parser.parse_args()

    if args.batch_size != 1:
        raise SystemExit("batch_size must be 1 for the current strict Qwen runner")
    manifest_rows = read_jsonl(args.manifest)
    manifest_rows.sort(key=lambda row: (str(row.get("split", "")), str(row.get("id", ""))))
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]
    run_dir = _run_dir(args.run_name, args.output_dir)
    predictions_path = run_dir / "predictions.jsonl"
    write_json(run_dir / "environment.json", environment_snapshot())
    resolved_config = {
        "manifest": str(args.manifest.resolve()),
        "run_name": args.run_name,
        "model": args.model,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "language": args.language,
        "limit": args.limit,
        "seed": args.seed,
        "require_baseline_model": args.require_baseline_model,
        "nvidia_smi": command_output(["nvidia-smi"]),
    }
    write_json(run_dir / "config_resolved.json", resolved_config)
    _write_yaml(run_dir / "config_resolved.yaml", resolved_config)

    completed, corrupted = load_prediction_state(predictions_path, retry_failed=args.retry_failed)
    if corrupted:
        print(f"warning: ignored {corrupted} corrupted prediction rows", file=sys.stderr)
    runner = QwenASRRunner(
        args.model,
        dtype=args.dtype,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
        require_baseline_model=args.require_baseline_model,
    )
    try:
        runner.load()
    except QwenRunnerError as exc:
        print(f"Qwen runner failed to load: {exc}", file=sys.stderr)
        return 2
    model_info_payload = runner.model_info()
    write_json(run_dir / "model_info.json", model_info_payload)

    total = len(manifest_rows)
    skipped = 0
    completed_now = 0
    last_id = None
    for index, item in enumerate(manifest_rows, start=1):
        utt_id = str(item["id"])
        last_id = utt_id
        if args.resume and utt_id in completed:
            skipped += 1
            continue
        audio_path = Path(str(item["audio_path"]))
        row: dict[str, Any] = {
            "id": utt_id,
            "split": item.get("split"),
            "speaker_id": item.get("speaker_id"),
            "chapter_id": item.get("chapter_id"),
            "audio_path": str(audio_path),
            "audio_sha256": item.get("audio_sha256"),
            "duration_sec": item.get("duration_sec"),
            "audio_duration_sec": item.get("duration_sec"),
            "reference": item.get("reference", ""),
            "normalized_reference": normalize_librispeech_text(str(item.get("reference", ""))),
            "model_name": args.model,
            "model_revision": model_info_payload.get("model_revision"),
            "language": "English" if args.language in ("auto", "", None) else args.language,
            "status": "pending",
            "error": None,
        }
        try:
            result = runner.transcribe(audio_path)
            row.update(
                {
                    "status": "success",
                    "hypothesis": result.hypothesis,
                    "normalized_hypothesis": normalize_librispeech_text(result.hypothesis),
                    "text_extraction_path": result.text_extraction_path,
                    "result_type": result.result_type,
                    "latency_sec": result.latency_sec,
                    "real_time_factor": result.latency_sec / float(item.get("duration_sec") or 1.0),
                    "peak_gpu_memory_gb": result.peak_gpu_memory_gb,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "hypothesis": "",
                    "normalized_hypothesis": "",
                    "latency_sec": None,
                    "real_time_factor": None,
                    "peak_gpu_memory_gb": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        append_prediction(predictions_path, row)
        completed_now += 1
        update_progress(
            run_dir / "progress.json",
            {
                "completed_now": completed_now,
                "skipped_existing": skipped,
                "total_requested": total,
                "last_id": utt_id,
                "updated_at_utc": utc_timestamp(),
            },
        )
        print(json.dumps({"id": utt_id, "status": row["status"], "index": index, "total": total}, sort_keys=True), flush=True)

    update_progress(
        run_dir / "progress.json",
        {
            "completed_now": completed_now,
            "skipped_existing": skipped,
            "total_requested": total,
            "last_id": last_id,
            "updated_at_utc": utc_timestamp(),
        },
    )
    print(
        json.dumps(
            {
                "status": "resume_summary",
                "completed_now": completed_now,
                "skipped_existing": skipped,
                "total_requested": total,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    payload = _score_and_write(run_dir, manifest_rows, predictions_path, args)
    print(json.dumps({"run_dir": str(run_dir), "wer": payload["metrics"]["normalized"]["wer"]}, sort_keys=True))
    return 0 if payload["successful_predictions"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
