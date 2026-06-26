#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from vigil_latest_opt.utils import read_json, write_json


REPORTS = Path("finetune/experiments/latest_data_optimization/reports")
RUN = Path("finetune/benchmarks/asr/runs/20260624_185419_qwen3_asr_1_7b_fixed_text_extraction_baseline_full")
PROF_REPORT = REPORTS / "LATEST_OPT_PROFESSOR_MEETING_REPORT.md"


OBJECT_REPR_RE = re.compile(r"(ASRTranscription\(|ASRResult\(|TranscriptionResult\(|language=|text=|<.* object at 0x[0-9a-fA-F]+>)")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_run() -> dict[str, Any]:
    required = ["FINAL_REPORT.md", "predictions.jsonl", "metrics.json", "per_split_metrics.json", "failures.jsonl"]
    missing = [name for name in required if not (RUN / name).exists()]
    if missing:
        return {"status": "missing", "run": str(RUN), "missing": missing}
    metrics = read_json(RUN / "metrics.json")
    normalized = read_json(RUN / "metrics_normalized.json")
    raw = read_json(RUN / "metrics_raw.json")
    per_split = read_json(RUN / "per_split_metrics.json")
    predictions = load_jsonl(RUN / "predictions.jsonl")
    failures = [line for line in (RUN / "failures.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    malformed = []
    paths: set[str] = set()
    result_types: set[str] = set()
    for row in predictions:
        hyp = str(row.get("hypothesis") or "")
        if OBJECT_REPR_RE.search(hyp):
            malformed.append(row.get("utterance_id"))
        extraction = row.get("text_extraction_path") or row.get("extraction_path")
        if extraction:
            paths.add(str(extraction))
        result_type = row.get("result_type") or row.get("raw_result_type")
        if result_type:
            result_types.add(str(result_type))
    status = "verified" if len(predictions) == 5559 and not failures and not malformed else "invalid_or_incomplete"
    return {
        "status": status,
        "run": str(RUN),
        "successes": len(predictions),
        "failures": len(failures),
        "malformed_hypotheses": len(malformed),
        "malformed_sample_ids": malformed[:10],
        "metrics_json_successful_predictions": metrics.get("successful_predictions"),
        "metrics_json_failed_predictions": metrics.get("failed_predictions"),
        "combined_normalized_wer": normalized.get("wer"),
        "normalized_cer": normalized.get("cer"),
        "raw_wer": raw.get("wer"),
        "per_split": per_split,
        "extraction_paths": sorted(paths),
        "result_types": sorted(result_types),
    }


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.4f}%"


def write_report(summary: dict[str, Any]) -> None:
    split = summary.get("per_split", {})
    clean = split.get("test-clean", {})
    other = split.get("test-other", {})
    report = [
        "# Current Qwen LibriSpeech Benchmark",
        "",
        "This benchmark evaluates the frozen continuous Qwen3-ASR branch in the VIGIL clinical workflow architecture. It does not evaluate the two-stage trigger detector.",
        "",
        "The current method does not fine-tune Qwen. Qwen weights are frozen, so LibriSpeech measures the unchanged general ASR branch. VIGIL recall/FPR is a separate wake-word trigger metric. If future Qwen LoRA or SFT is performed, LibriSpeech must be rerun on that fine-tuned Qwen checkpoint.",
        "",
        "| Qwen module | Qwen updated? | Benchmark | test-clean WER | test-other WER | Combined WER |",
        "|---|---:|---|---:|---:|---:|",
        f"| Continuous frozen Qwen3-ASR-1.7B | No | LibriSpeech | {pct(clean.get('wer'))} | {pct(other.get('wer'))} | {pct(summary.get('combined_normalized_wer'))} |",
        "",
        "## Verification",
        "",
        f"- Run: `{summary.get('run')}`",
        f"- Status: `{summary.get('status')}`",
        f"- Successful predictions: `{summary.get('successes')}`",
        f"- Failures: `{summary.get('failures')}`",
        f"- Malformed/object-repr hypotheses: `{summary.get('malformed_hypotheses')}`",
        f"- Extraction path: `{summary.get('extraction_paths')}`",
        f"- Result type: `{summary.get('result_types')}`",
        "",
        "The old approximately 40% WER run is invalid because it stored `ASRTranscription(...)` object representations instead of the `.text` transcript. It is not used here.",
        "",
    ]
    (REPORTS / "CURRENT_QWEN_LIBRISPEECH_BENCHMARK.md").write_text("\n".join(report), encoding="utf-8")

    insertion = [
        "## Current Frozen-Qwen LibriSpeech Benchmark",
        "",
        "LibriSpeech has been completed for the frozen base Qwen branch. It is not a new fine-tuned Qwen benchmark because Qwen is not fine-tuned in the current method.",
        "",
        f"- test-clean WER: {pct(clean.get('wer'))}",
        f"- test-other WER: {pct(other.get('wer'))}",
        f"- combined normalized WER: {pct(summary.get('combined_normalized_wer'))}",
        f"- successes/failures: `{summary.get('successes')}` / `{summary.get('failures')}`",
        f"- text extraction path: `{summary.get('extraction_paths')}`",
        "",
    ]
    if PROF_REPORT.exists():
        text = PROF_REPORT.read_text(encoding="utf-8")
        marker = "## Current Frozen-Qwen LibriSpeech Benchmark"
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
        else:
            text = text.rstrip()
        text = text + "\n\n" + "\n".join(insertion).rstrip()
        PROF_REPORT.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = validate_run()
    write_json(REPORTS / "current_qwen_librispeech_benchmark.json", summary)
    write_report(summary)
    print(json.dumps({"status": summary["status"], "wer": summary.get("combined_normalized_wer")}, sort_keys=True))
    return 0 if summary["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
