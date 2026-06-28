#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from call_counter import MethodCallCounter, can_claim_verified_one_encoder_forward
from parity import compare_transcripts, cost_table_row, score_parity_status
from shared_qwen_adapter import BLOCKER, SharedQwenASRRuntime
from vigil_two_stage.qwen_audio_adapter import FrozenQwenAudioAdapter
from vigil_two_stage.qwen_text_result import extract_qwen_text
from vigil_two_stage.stage2_model import QwenVerifierHead


ROOT = Path(".")
EXP = Path("finetune/experiments/shared_qwen_runtime")
REPORTS = EXP / "reports"
BALANCED = Path("finetune/experiments/latest_data/shared/balanced_max100_latest_manifest.jsonl")
LIBRI = Path("finetune/benchmarks/asr/manifests/smoke_all.jsonl")
BUNDLE = Path("finetune/model_bundles/vigil_latest_optimized_20260626_085405")
COMPUTE = Path("finetune/experiments/latest_data_optimization/reports/latest_opt_compute_cost.json")
MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha12(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def file_sha12(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def source_snippet(path: Path, start: int, end: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [f"{idx}: {lines[idx - 1]}" for idx in range(start, min(end, len(lines)) + 1)]


def inspect_qwen_runtime(runtime: SharedQwenASRRuntime | None = None) -> dict[str, Any]:
    import importlib.metadata as md
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

    src = Path(inspect.getsourcefile(Qwen3ASRModel) or "")
    text = src.read_text(encoding="utf-8", errors="ignore")
    public_methods = [name for name in dir(Qwen3ASRModel) if not name.startswith("_")]
    result: dict[str, Any] = {
        "package_version": md.version("qwen-asr"),
        "model_name": MODEL_NAME,
        "qwen3_asr_class": f"{Qwen3ASRModel.__module__}.{Qwen3ASRModel.__name__}",
        "source_file": str(src),
        "transcribe_signature": str(inspect.signature(Qwen3ASRModel.transcribe)),
        "from_pretrained_signature": str(inspect.signature(Qwen3ASRModel.from_pretrained)),
        "streaming_transcribe_signature": str(inspect.signature(Qwen3ASRModel.streaming_transcribe)),
        "public_methods_with_hidden_or_feature": [
            name for name in public_methods if "hidden" in name.lower() or "feature" in name.lower()
        ],
        "public_transcribe_code_path": [
            "transcribe normalizes audio with normalize_audios",
            "transcribe splits audio into chunks",
            "transcribe calls _infer_asr(contexts, wavs, languages)",
            "_infer_asr_transformers builds processor(text=..., audio=...) inputs",
            "_infer_asr_transformers calls self.model.generate(**inputs)",
            "decoded text is parsed into ASRTranscription(language, text, time_stamps)",
        ],
        "transcribe_lines": source_snippet(src, 338, 384),
        "transformers_infer_lines": source_snippet(src, 498, 517),
        "streaming_vllm_lines": source_snippet(src, 748, 754),
        "transcribe_calls_get_audio_features_literal": "get_audio_features" in "\n".join(source_snippet(src, 300, 540)),
        "decoder_accepts_external_hidden_states_in_public_wrapper": False,
        "hidden_states_accessible_from_public_transcribe": False,
        "exact_blocker": BLOCKER,
    }
    if runtime is not None and runtime.wrapper is not None:
        wrapper = runtime.wrapper
        thinker = getattr(wrapper.model, "thinker", None)
        get_audio = getattr(thinker, "get_audio_features", None)
        result.update(
            {
                "loaded_backend": getattr(wrapper, "backend", None),
                "loaded_model_class": f"{type(wrapper.model).__module__}.{type(wrapper.model).__name__}",
                "loaded_model_device": str(getattr(wrapper.model, "device", "unknown")),
                "loaded_model_dtype": str(getattr(wrapper.model, "dtype", "unknown")),
                "model_generate_signature": safe_signature(getattr(wrapper.model, "generate", None)),
                "thinker_class": f"{type(thinker).__module__}.{type(thinker).__name__}" if thinker is not None else None,
                "thinker_get_audio_features_signature": safe_signature(get_audio),
                "stage2_feature_code_path": "FrozenQwenAudioAdapter -> qwen_asr.inference.utils.normalize_audio_input -> processor(text=[audio_token], audio=[audio]) -> model.thinker.get_audio_features(input_features, feature_attention_mask)",
                "two_paths_share_processor_audio_preprocessing": True,
                "two_paths_share_reusable_hidden_state_object": False,
            }
        )
    return result


def safe_signature(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        return str(inspect.signature(obj))
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def select_vigil_subset() -> list[dict[str, Any]]:
    rows = read_jsonl(BALANCED)
    groups = [
        ("P1_vigil_only", 5),
        ("P2_phrase_plus_vigil", 5),
        ("P3_vigil_plus_phrase", 5),
        ("P4_negative", 10),
    ]
    selected: list[dict[str, Any]] = []
    for group, limit in groups:
        pool = [row for row in rows if row.get("prompt_group") == group]
        selected.extend(pool[:limit])
    return selected


def select_librispeech_subset(limit: int = 4) -> list[dict[str, Any]]:
    if not LIBRI.exists():
        return []
    return read_jsonl(LIBRI)[:limit]


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    audio_path = str(row.get("window_wav_path") or row.get("audio_path"))
    return {
        "audio_hash": row.get("window_audio_sha256", row.get("audio_sha256", file_sha12(audio_path)))[:12],
        "path_hash": sha12(audio_path),
        "prompt_group": row.get("prompt_group"),
        "label": row.get("label"),
        "reference": row.get("transcript") or row.get("reference"),
        "source": "vigil" if "window_wav_path" in row else "librispeech",
    }


def load_stage2(device: str) -> tuple[QwenVerifierHead, dict[str, Any]]:
    manifest = read_json(BUNDLE / "PUBLIC_MANIFEST.json")
    checkpoint_path = Path(manifest["stage2_checkpoint"]["path"])
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model = QwenVerifierHead(
        input_dim=int(ckpt["input_dim"]),
        projection_dim=int(config.get("projection_dim", 256)),
        embedding_dim=int(config.get("embedding_dim", 128)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, {
        "bundle": str(BUNDLE),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_path_hash": sha12(str(checkpoint_path)),
        "stage2_variant": manifest["selected_config"]["variant"],
        "stage2_threshold": manifest["stage2_threshold"]["threshold"],
        "input_dim": int(ckpt["input_dim"]),
        "qwen_trainable_params": 0,
    }


def score_stage2(model: QwenVerifierHead, hidden: torch.Tensor, device: str) -> dict[str, Any]:
    with torch.inference_mode():
        output = model(hidden.unsqueeze(0).to(device))
        logit = float(output["logit"].detach().float().cpu().item())
        score = float(torch.sigmoid(output["logit"]).detach().float().cpu().item())
    return {"logit": logit, "score": score}


def run_call_counter_diagnostic(runtime: SharedQwenASRRuntime, audio_path: str) -> dict[str, Any]:
    public = runtime.public_transcribe(audio_path)
    separate = runtime.separate_stage2_features(audio_path)

    counter = runtime._patch_counter()
    try:
        start = time.perf_counter()
        raw = runtime.wrapper.transcribe(audio_path, language=None, return_time_stamps=False)
        extracted = extract_qwen_text(raw)
        adapter = FrozenQwenAudioAdapter(runtime.model_name_or_path)
        adapter._set_loaded_model(runtime.wrapper)
        hidden = adapter.extract_audio_features(audio_path)
        latency_ms = (time.perf_counter() - start) * 1000.0
        combined_counts = counter.snapshot(runtime.model_load_count).as_dict()
    finally:
        counter.restore()

    shared = runtime.transcribe_and_get_features(audio_path).as_dict()
    return {
        "sample": {"path_hash": sha12(audio_path), "audio_hash": file_sha12(audio_path)},
        "public_transcribe_only": public,
        "separate_stage2_get_audio_features_only": separate,
        "public_transcribe_plus_separate_stage2": {
            "status": "ok",
            "transcript": extracted.text,
            "result_type": extracted.result_type,
            "extraction_path": extracted.extraction_path,
            "feature_shape": list(hidden.shape),
            "feature_dtype": str(hidden.dtype),
            "feature_path": adapter.extraction_path,
            "latency_ms": latency_ms,
            "counts": combined_counts,
        },
        "attempted_shared_path": shared,
    }


def run_stage2_score_parity(runtime: SharedQwenASRRuntime, rows: list[dict[str, Any]], device: str) -> dict[str, Any]:
    stage2, stage2_info = load_stage2(device)
    adapter = FrozenQwenAudioAdapter(runtime.model_name_or_path)
    adapter._set_loaded_model(runtime.wrapper)
    records = []
    latencies = []
    for row in rows:
        audio_path = str(row["window_wav_path"])
        start = time.perf_counter()
        hidden = adapter.extract_audio_features(audio_path)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)
        score = score_stage2(stage2, hidden, device)
        pub = public_row(row)
        records.append(
            {
                **pub,
                "separate_feature_shape": list(hidden.shape),
                "separate_feature_dtype": str(hidden.dtype),
                "separate_feature_path": adapter.extraction_path,
                "separate_stage2_logit": score["logit"],
                "separate_stage2_score": score["score"],
                "shared_stage2_score": None,
                "abs_score_diff": None,
                "status": "blocked_by_runtime_interface",
                "blocker": BLOCKER,
            }
        )
    return {
        "status": "blocked_by_runtime_interface",
        "stage2_info": stage2_info,
        "subset_size": len(records),
        "records": records,
        "max_abs_score_diff": None,
        "parity_status": score_parity_status(None),
        "separate_feature_latency_ms": {
            "median": statistics.median(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "blocker": BLOCKER,
    }


def run_transcript_parity(runtime: SharedQwenASRRuntime, vigil_rows: list[dict[str, Any]], libri_rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for row in [*vigil_rows, *libri_rows]:
        audio_path = str(row.get("window_wav_path") or row.get("audio_path"))
        public = runtime.public_transcribe(audio_path)
        pub = public_row(row)
        parity = compare_transcripts(public["transcript"], None)
        records.append(
            {
                **pub,
                "public_transcript": public["transcript"],
                "public_result_type": public["result_type"],
                "public_extraction_path": public["extraction_path"],
                "shared_transcript": None,
                "parity": parity.as_dict() if parity else None,
                "status": "blocked_by_runtime_interface",
                "blocker": BLOCKER,
            }
        )
    return {
        "status": "blocked_by_runtime_interface",
        "vigil_examples": len(vigil_rows),
        "librispeech_examples": len(libri_rows),
        "records": records,
        "transcript_parity_status": "blocked",
        "blocker": BLOCKER,
    }


def run_non_regression(stage2_parity: dict[str, Any], transcript_parity: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "blocked_by_runtime_interface",
        "asr_smoke_status": "not_run_because_shared_path_blocked",
        "vigil_metric_status": "not_run_because_shared_path_blocked",
        "current_frozen_qwen_librispeech": {
            "test_clean_wer": 0.018411442483262326,
            "test_other_wer": 0.03666201784383776,
            "combined_wer": 0.02751646508258752,
            "successes": 5559,
            "failures": 0,
        },
        "current_vigil_trigger_metrics": {
            "recall": 0.9408602151,
            "fpr": 0.0049833887,
            "precision": 0.9957325747,
            "f1": 0.9675190048,
        },
        "shared_stage2_score_parity_status": stage2_parity["parity_status"],
        "shared_transcript_parity_status": transcript_parity["transcript_parity_status"],
        "blocker": BLOCKER,
    }


def cost_tradeoff() -> dict[str, Any]:
    compute = read_json(COMPUTE) if COMPUTE.exists() else {}
    components = {row.get("component"): row for row in compute.get("components", [])}
    qwen = components.get("qwen_audio_encoder_forward", {})
    extra = qwen.get("median_ms", 13.663365971297026)
    rows = [
        cost_table_row(
            "Current prototype",
            1,
            "public transcribe path plus one extra get_audio_features call for Stage 2 candidates",
            "yes",
            "yes",
            float(extra) if extra is not None else None,
            "working",
        ),
        cost_table_row(
            "Shared hidden-state prototype",
            1,
            "1 only if upstream exposes decoder-compatible hidden-state handoff",
            "blocked",
            "blocked",
            None,
            "blocked_by_runtime_interface",
        ),
    ]
    return {
        "status": "blocked_by_runtime_interface",
        "extra_encoder_median_ms_per_candidate": extra,
        "rows": rows,
        "blocker": BLOCKER,
    }


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return out


def write_reports(
    inspection: dict[str, Any],
    calls: dict[str, Any],
    stage2: dict[str, Any],
    transcript: dict[str, Any],
    nonreg: dict[str, Any],
    cost: dict[str, Any],
) -> None:
    write_json(REPORTS / "qwen_runtime_deep_inspection.json", inspection)
    write_json(REPORTS / "call_counter_diagnostic.json", calls)
    write_json(REPORTS / "stage2_score_parity.json", stage2)
    write_json(REPORTS / "transcript_parity.json", transcript)
    write_json(REPORTS / "shared_qwen_non_regression.json", nonreg)

    (REPORTS / "QWEN_RUNTIME_DEEP_INSPECTION.md").write_text(
        "\n".join(
            [
                "# Qwen Runtime Deep Inspection",
                "",
                f"- qwen-asr version: `{inspection['package_version']}`",
                f"- Qwen class: `{inspection['qwen3_asr_class']}`",
                f"- Source file: `{inspection['source_file']}`",
                f"- Transcribe signature: `{inspection['transcribe_signature']}`",
                f"- Loaded backend: `{inspection.get('loaded_backend')}`",
                f"- Model generate signature: `{inspection.get('model_generate_signature')}`",
                f"- thinker.get_audio_features signature: `{inspection.get('thinker_get_audio_features_signature')}`",
                f"- Public methods exposing hidden/features: `{inspection['public_methods_with_hidden_or_feature']}`",
                "",
                "## Code Path",
                "",
                *[f"- {item}" for item in inspection["public_transcribe_code_path"]],
                "",
                "## Finding",
                "",
                f"- Hidden states accessible from public transcribe: `{inspection['hidden_states_accessible_from_public_transcribe']}`",
                f"- Decoder accepts external hidden states in public wrapper: `{inspection['decoder_accepts_external_hidden_states_in_public_wrapper']}`",
                f"- Stage 2 feature path: `{inspection.get('stage2_feature_code_path')}`",
                f"- Exact blocker: {inspection['exact_blocker']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    combined = calls["public_transcribe_plus_separate_stage2"]["counts"]
    shared = calls["attempted_shared_path"]
    (REPORTS / "CALL_COUNTER_DIAGNOSTIC.md").write_text(
        "\n".join(
            [
                "# Call Counter Diagnostic",
                "",
                f"- Model load count: `{calls['public_transcribe_only']['counts']['model_load_count']}`",
                f"- Public transcribe counts: `{calls['public_transcribe_only']['counts']}`",
                f"- Separate Stage 2 feature counts: `{calls['separate_stage2_get_audio_features_only']['counts']}`",
                f"- Public transcribe + separate Stage 2 counts: `{combined}`",
                f"- Attempted shared status: `{shared['status']}`",
                f"- Attempted shared encoder calls: `{shared['encoder_call_count']}`",
                f"- Attempted shared decoder calls: `{shared['decoder_call_count']}`",
                f"- Can claim one encoder forward: `{can_claim_verified_one_encoder_forward(shared['status'], {'encoder_call_count': shared['encoder_call_count']})}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (REPORTS / "STAGE2_SCORE_PARITY_REPORT.md").write_text(
        "\n".join(
            [
                "# Stage 2 Score Parity Report",
                "",
                f"- Status: `{stage2['status']}`",
                f"- Fixed VIGIL subset size: `{stage2['subset_size']}`",
                f"- Separate feature median latency: `{stage2['separate_feature_latency_ms']['median']}` ms",
                f"- Shared Stage 2 score parity: `{stage2['parity_status']}`",
                f"- Max absolute score difference: `{stage2['max_abs_score_diff']}`",
                f"- Blocker: {stage2['blocker']}",
                "",
                "The separate production Stage 2 feature path was measured. A shared score was not computed because the runtime did not expose reusable decoder-compatible hidden states.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (REPORTS / "TRANSCRIPT_PARITY_REPORT.md").write_text(
        "\n".join(
            [
                "# Transcript Parity Report",
                "",
                f"- Status: `{transcript['status']}`",
                f"- VIGIL examples transcribed with public path: `{transcript['vigil_examples']}`",
                f"- LibriSpeech examples transcribed with public path: `{transcript['librispeech_examples']}`",
                f"- Shared transcript parity: `{transcript['transcript_parity_status']}`",
                f"- Blocker: {transcript['blocker']}",
                "",
                "Public transcripts were collected as a sanity check. Shared-path transcript parity is blocked because no project-owned one-forward path can pass hidden states into the Qwen decoder.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (REPORTS / "SHARED_QWEN_NON_REGRESSION_REPORT.md").write_text(
        "\n".join(
            [
                "# Shared Qwen Non-Regression Report",
                "",
                f"- Status: `{nonreg['status']}`",
                f"- ASR smoke: `{nonreg['asr_smoke_status']}`",
                f"- VIGIL metric check: `{nonreg['vigil_metric_status']}`",
                f"- Current frozen Qwen combined WER: `{nonreg['current_frozen_qwen_librispeech']['combined_wer']}`",
                f"- Current VIGIL F1: `{nonreg['current_vigil_trigger_metrics']['f1']}`",
                f"- Blocker: {nonreg['blocker']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cost_lines = [
        "# Shared Qwen Cost Tradeoff Report",
        "",
        f"- Status: `{cost['status']}`",
        f"- Extra encoder median cost per Stage 1 candidate: `{cost['extra_encoder_median_ms_per_candidate']}` ms",
        "",
        *md_table(
            cost["rows"],
            ["variant", "qwen_weight_copies", "encoder_forwards", "transcript", "stage2_score", "median_latency_ms", "status"],
        ),
        "",
    ]
    (REPORTS / "SHARED_QWEN_COST_TRADEOFF_REPORT.md").write_text("\n".join(cost_lines), encoding="utf-8")

    final_lines = [
        "# Final Shared Qwen Runtime Report",
        "",
        "## Professor Question",
        "",
        "Can one frozen Qwen3-ASR runtime provide both the continuous transcript and the Stage 2 VIGIL verifier features without a second audio encoder forward?",
        "",
        "## Current System",
        "",
        "The current clinical workflow uses a continuous frozen Qwen ASR branch for transcript and a parallel VIGIL trigger branch. Stage 2 uses frozen Qwen audio features with a small verifier head. Qwen weights remain frozen.",
        "",
        "## Inspection And Attempt",
        "",
        f"- qwen-asr version: `{inspection['package_version']}`",
        f"- Public transcribe result extraction path: `$[0].text`",
        f"- Stage 2 feature path: `{inspection.get('stage2_feature_code_path')}`",
        f"- Call-counter combined path: `{combined}`",
        f"- Attempted shared status: `{shared['status']}`",
        "",
        "## Final Status",
        "",
        "`blocked_by_runtime_interface`",
        "",
        "The current public qwen_asr wrapper does not expose decoder-compatible audio hidden states and does not accept externally supplied audio hidden states for decoding. Therefore, we cannot yet prove one-forward shared Qwen-ASR. Current prototype still uses same frozen Qwen weights but one extra Qwen encoder forward for Stage 2 candidates.",
        "",
        "## Transcript Parity",
        "",
        f"- Status: `{transcript['transcript_parity_status']}`",
        "",
        "## Stage 2 Score Parity",
        "",
        f"- Status: `{stage2['parity_status']}`",
        "",
        "## Metric Non-Regression",
        "",
        f"- Status: `{nonreg['status']}`",
        "- No shared-path LibriSpeech/VIGIL metric claim is made because the shared path is blocked.",
        "",
        "## Cost",
        "",
        f"- Current extra encoder median cost: `{cost['extra_encoder_median_ms_per_candidate']}` ms per Stage 1 candidate.",
        "",
        "## Chinese Notes",
        "",
        "当前 qwen_asr 公共接口只返回转写文本，不返回可以复用给解码器和 Stage 2 的同一份 audio hidden states。因此现在不能声称一个 encoder forward 同时服务 ASR 和 VIGIL Stage 2。下一步需要上游接口暴露 decoder-compatible audio hidden states，或允许 decoder 接收外部传入的 hidden states。",
        "",
        "## Exact Next Technical Step",
        "",
        "Request or implement a project-owned Qwen wrapper API that returns decoder-compatible audio hidden states and accepts those same states for generation, then rerun call-counter, transcript parity, Stage 2 score parity, and non-regression checks.",
        "",
    ]
    (REPORTS / "FINAL_SHARED_QWEN_RUNTIME_REPORT.md").write_text("\n".join(final_lines), encoding="utf-8")


def write_handoff(
    inspection: dict[str, Any],
    calls: dict[str, Any],
    stage2: dict[str, Any],
    transcript: dict[str, Any],
    nonreg: dict[str, Any],
    cost: dict[str, Any],
) -> None:
    handoff = Path("docs/CODEX_HANDOFF_VIGIL_SHARED_QWEN_RUNTIME.md")
    handoff.write_text(
        "\n".join(
            [
                "# Codex Handoff: VIGIL Shared Qwen Runtime",
                "",
                "## Branch And Commit",
                "",
                "- Branch: `research/vigil-shared-qwen-runtime-20260627`",
                "- Commit: pending until git commit.",
                "",
                "## Qwen Runtime",
                "",
                f"- qwen-asr version: `{inspection['package_version']}`",
                f"- Model: `{MODEL_NAME}`",
                f"- Source file: `{inspection['source_file']}`",
                f"- Backend: `{inspection.get('loaded_backend')}`",
                "",
                "## Stage 2 Checkpoint And Config",
                "",
                f"- Bundle: `{BUNDLE}`",
                f"- Variant: `{stage2['stage2_info']['stage2_variant']}`",
                f"- Stage 2 threshold: `{stage2['stage2_info']['stage2_threshold']}`",
                f"- Checkpoint exists: `{stage2['stage2_info']['checkpoint_exists']}`",
                "",
                "## Shared-Qwen Status",
                "",
                "- Status: `blocked_by_runtime_interface`",
                f"- Blocker: {BLOCKER}",
                f"- Success proof available: `{can_claim_verified_one_encoder_forward(calls['attempted_shared_path']['status'], {'encoder_call_count': calls['attempted_shared_path']['encoder_call_count']})}`",
                "",
                "## Parity And Non-Regression",
                "",
                f"- Transcript parity: `{transcript['transcript_parity_status']}`",
                f"- Stage 2 score parity: `{stage2['parity_status']}`",
                f"- LibriSpeech check: `{nonreg['asr_smoke_status']}`",
                f"- VIGIL trigger metric parity: `{nonreg['vigil_metric_status']}`",
                "",
                "## Cost",
                "",
                f"- Extra Qwen encoder median cost remains `{cost['extra_encoder_median_ms_per_candidate']}` ms per Stage 1 candidate.",
                "",
                "## GPU Assignment",
                "",
                "- Real diagnostic used one idle local RTX 3090 through `CUDA_VISIBLE_DEVICES`.",
                "",
                "## Tmux Sessions",
                "",
                "- None required for this branch.",
                "",
                "## Artifact Paths",
                "",
                f"- Reports: `{REPORTS}`",
                f"- Final report: `{REPORTS / 'FINAL_SHARED_QWEN_RUNTIME_REPORT.md'}`",
                f"- Deep inspection JSON: `{REPORTS / 'qwen_runtime_deep_inspection.json'}`",
                f"- Call counter JSON: `{REPORTS / 'call_counter_diagnostic.json'}`",
                "",
                "## Exact Next Command",
                "",
                "```bash",
                "cd /home/hj/Data_Collect_Web && PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH PYTHONPATH=finetune/src:finetune/experiments/shared_qwen_runtime/src:. pytest -q finetune/experiments/shared_qwen_runtime/tests",
                "```",
                "",
                "## Push Status",
                "",
                "- Pending commit and push.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != "cuda:0":
        raise RuntimeError("CUDA is required for real shared-Qwen runtime diagnostic.")
    vigil_subset = select_vigil_subset()
    libri_subset = select_librispeech_subset()
    runtime = SharedQwenASRRuntime(MODEL_NAME)
    runtime.load()
    inspection = inspect_qwen_runtime(runtime)
    sample_audio = str(vigil_subset[0]["window_wav_path"])
    calls = run_call_counter_diagnostic(runtime, sample_audio)
    stage2 = run_stage2_score_parity(runtime, vigil_subset, device)
    transcript = run_transcript_parity(runtime, vigil_subset, libri_subset)
    nonreg = run_non_regression(stage2, transcript)
    cost = cost_tradeoff()
    write_reports(inspection, calls, stage2, transcript, nonreg, cost)
    write_handoff(inspection, calls, stage2, transcript, nonreg, cost)
    print(json.dumps({"status": "blocked_by_runtime_interface", "vigil_subset": len(vigil_subset), "librispeech_subset": len(libri_subset)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
