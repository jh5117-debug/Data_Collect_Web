#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vigil_latest_opt.metrics import metrics_from_rows, paired_delta, participant_macro
from vigil_latest_opt.prototype import build_prototype, cosine_similarity
from vigil_latest_opt.support import choose_positive_support
from vigil_latest_opt.utils import logit, read_json, read_jsonl, stable_sigmoid, write_csv, write_json
from vigil_two_stage.stage2_model import QwenVerifierHead


SUPPORT_SEEDS = [20260620, 20260621, 20260622, 20260623, 20260624]


def prediction_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["clip_id"]), int(row.get("window_index", 0))


def load_npz(path: str) -> np.ndarray:
    data = np.load(path)
    return (data["features"] if "features" in data else data[data.files[0]]).astype(np.float32)


def load_stage2_model(checkpoint: Path, device: str) -> QwenVerifierHead:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = QwenVerifierHead(int(ckpt["input_dim"]), int(ckpt["config"]["projection_dim"]), int(ckpt["config"]["embedding_dim"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def qwen_exact_by_clip(path: Path) -> dict[str, bool]:
    return {str(row["clip_id"]): bool(row.get("exact_trigger_decision")) for row in read_jsonl(path)}


def embed_rows(run_dir: Path, variant: str, split: str, device: str) -> dict[tuple[str, int], np.ndarray]:
    model = load_stage2_model(run_dir / variant / "checkpoint_best.pt", device)
    rows = [row for row in read_jsonl(run_dir / "stage2_qwen_features/qwen_features_manifest.jsonl") if row["split"] == split]
    out: dict[tuple[str, int], np.ndarray] = {}
    with torch.no_grad():
        for row in rows:
            arr = load_npz(row["feature_path"])
            hidden = torch.from_numpy(arr).unsqueeze(0).to(device)
            mask = torch.ones(1, arr.shape[0], dtype=torch.bool, device=device)
            result = model(hidden, mask)
            out[prediction_key(row)] = result["embedding"].detach().cpu().numpy()[0].astype(np.float32)
    return out


def build_clip_records(run_root: Path, fold: int, split: str, selected: dict[str, Any], qwen_exact: dict[str, bool], device: str) -> list[dict[str, Any]]:
    run_dir = run_root / f"fold_{fold}"
    variant = selected["variant"]
    threshold = float(selected["thresholds"][fold])
    theta1 = float(read_json(run_dir / "stage1/threshold.json")["threshold"])
    stage1 = read_jsonl(run_dir / f"stage1/{split}_predictions.jsonl")
    stage2 = read_jsonl(run_dir / f"{variant}/{split}_predictions.jsonl")
    features = [row for row in read_jsonl(run_dir / "stage2_qwen_features/qwen_features_manifest.jsonl") if row["split"] == split]
    feature_meta = {prediction_key(row): row for row in features}
    stage2_by_key = {prediction_key(row): row for row in stage2}
    embeddings = embed_rows(run_dir, variant, split, device)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage1:
        key = prediction_key(row)
        s2 = stage2_by_key.get(key)
        meta = feature_meta.get(key)
        embedding = embeddings.get(key)
        if s2 is None or meta is None or embedding is None:
            continue
        stage1_score = float(row["score"])
        stage2_score = float(s2["stage2_score"])
        grouped[str(row["clip_id"])].append(
            {
                "clip_id": str(row["clip_id"]),
                "window_index": int(row.get("window_index", 0)),
                "participant_alias": str(meta["participant_alias"]),
                "label": int(row["label"]),
                "prompt_group": row.get("prompt_group"),
                "transcript": row.get("transcript"),
                "stage1_score": stage1_score,
                "stage2_score": stage2_score,
                "stage2_logit": logit(stage2_score),
                "embedding": embedding,
                "candidate": stage1_score >= theta1,
            }
        )
    clips = []
    for clip_id, windows in sorted(grouped.items()):
        candidates = sorted([row for row in windows if row["candidate"]], key=lambda row: float(row["stage1_score"]), reverse=True)[: int(selected["top_k"])]
        pool = candidates if candidates else sorted(windows, key=lambda row: float(row["stage1_score"]), reverse=True)[:1]
        best = max(pool, key=lambda row: float(row["stage2_score"]))
        base_decision = bool(candidates and any(float(row["stage2_score"]) >= threshold for row in candidates))
        first = windows[0]
        clips.append(
            {
                "clip_id": clip_id,
                "participant_alias": first["participant_alias"],
                "label": int(first["label"]),
                "prompt_group": first.get("prompt_group"),
                "transcript": first.get("transcript"),
                "base_decision": base_decision,
                "qwen_exact_decision": bool(qwen_exact.get(clip_id, False)),
                "stage1_candidate": bool(candidates),
                "base_score": max((float(row["stage2_score"]) for row in candidates), default=0.0),
                "base_logit": max((float(row["stage2_logit"]) for row in candidates), default=-20.0),
                "embedding": best["embedding"],
                "threshold": threshold,
            }
        )
    return clips


def by_participant(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["participant_alias"])].append(row)
    return grouped


def support_quantile(values: list[float], mode: str) -> float:
    if mode == "min":
        return min(values)
    if mode == "median":
        return float(np.median(values))
    if mode == "q20":
        return float(np.quantile(values, 0.2))
    raise ValueError(f"unsupported support quantile: {mode}")


def apply_recipe(query: list[dict[str, Any]], support: list[dict[str, Any]], recipe: dict[str, Any]) -> list[dict[str, Any]]:
    method = recipe["method"]
    out = []
    support_scores = [float(row["base_score"]) for row in support]
    support_logits = [float(row["base_logit"]) for row in support]
    prototype = build_prototype([row["embedding"] for row in support])
    if method == "support_threshold_calibration":
        base_threshold = float(support[0]["threshold"])
        q = support_quantile(support_scores, recipe["support_quantile"])
        target_threshold = min(base_threshold, max(base_threshold - float(recipe["max_threshold_drop"]), q - float(recipe["margin"])))
    elif method == "positive_bias_adaptation":
        base_logit_threshold = logit(float(support[0]["threshold"]))
        q = support_quantile(support_logits, recipe["support_quantile"])
        bias = min(float(recipe["max_bias"]), max(0.0, base_logit_threshold - q + float(recipe["margin"])))
    else:
        target_threshold = float(recipe.get("threshold", 0.5))
        bias = 0.0
    for row in query:
        decision = bool(row["base_decision"])
        score = float(row["base_score"])
        if method == "support_threshold_calibration":
            decision = bool(row["stage1_candidate"] and score >= target_threshold)
            adapted_score = score
        elif method == "positive_bias_adaptation":
            adapted_logit = float(row["base_logit"]) + bias
            adapted_score = stable_sigmoid(adapted_logit)
            decision = bool(row["stage1_candidate"] and adapted_logit >= logit(float(row["threshold"])))
        elif method == "prototype_fusion":
            sim = cosine_similarity(row["embedding"], prototype)
            adapted_score = score + float(recipe["alpha"]) * (sim - float(recipe["beta"]))
            decision = bool(row["stage1_candidate"] and adapted_score >= target_threshold)
        elif method == "qwen_exact_or_two_stage":
            adapted_score = max(score, 1.0 if row["qwen_exact_decision"] else 0.0)
            decision = bool(row["base_decision"] or row["qwen_exact_decision"])
        else:
            raise ValueError(f"unsupported recipe method: {method}")
        out.append({**row, "adapted_decision": decision, "adapted_score": adapted_score})
    return out


def recipe_grid(base_thresholds: list[float]) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for quantile in ("min", "q20", "median"):
        for margin in (0.0, 0.05, 0.10, 0.15):
            for drop in (0.05, 0.10, 0.20, 0.30):
                grid.append({"method": "support_threshold_calibration", "support_quantile": quantile, "margin": margin, "max_threshold_drop": drop, "support_based": True})
        for margin in (0.0, 0.25, 0.5, 1.0):
            for max_bias in (0.25, 0.5, 1.0, 2.0):
                grid.append({"method": "positive_bias_adaptation", "support_quantile": quantile, "margin": margin, "max_bias": max_bias, "support_based": True})
    for alpha in (0.25, 0.5, 1.0, 2.0, 4.0):
        for beta in (0.0, 0.25, 0.5):
            for threshold in sorted(set([0.50, 0.75, 0.90, 1.0, *base_thresholds])):
                grid.append({"method": "prototype_fusion", "alpha": alpha, "beta": beta, "threshold": float(threshold), "support_based": True})
    grid.append({"method": "qwen_exact_or_two_stage", "support_based": False})
    return grid


def evaluate_recipe(rows: list[dict[str, Any]], recipe: dict[str, Any], *, shots: int, seed: int) -> list[dict[str, Any]]:
    out = []
    for alias, group in by_participant(rows).items():
        try:
            support, query = choose_positive_support(group, shots=shots, seed=seed)
        except ValueError:
            continue
        adapted = apply_recipe(query, support, recipe)
        support_ids = ",".join(sorted(row["clip_id"] for row in support))
        for row in adapted:
            out.append(
                {
                    "participant_alias": alias,
                    "clip_id": row["clip_id"],
                    "label": int(row["label"]),
                    "prompt_group": row.get("prompt_group"),
                    "shots": shots,
                    "support_seed": seed,
                    "support_ids_hash": str(abs(hash(support_ids)) % 10**12),
                    "zero_decision": bool(row["base_decision"]),
                    "adapted_decision": bool(row["adapted_decision"]),
                    "adapted_score": float(row["adapted_score"]),
                }
            )
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    zero_rows = [{**row, "decision": row["zero_decision"]} for row in rows]
    adapted_rows = [{**row, "decision": row["adapted_decision"]} for row in rows]
    zero = metrics_from_rows(zero_rows) if zero_rows else {"n": 0}
    adapted = metrics_from_rows(adapted_rows) if adapted_rows else {"n": 0}
    return {
        "query_rows": len(rows),
        "zero": zero,
        "adapted": adapted,
        "delta_f1": (adapted.get("f1") or 0.0) - (zero.get("f1") or 0.0) if zero.get("f1") is not None and adapted.get("f1") is not None else None,
        "paired_delta": paired_delta(rows, "zero_decision", "adapted_decision"),
        "participant_macro_zero": participant_macro(zero_rows, "decision") if zero_rows else {"participants": 0},
        "participant_macro_adapted": participant_macro(adapted_rows, "decision") if adapted_rows else {"participants": 0},
    }


def select_recipe(dev_rows: list[dict[str, Any]], base_thresholds: list[float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    search_rows = []
    best: dict[str, Any] | None = None
    for recipe in recipe_grid(base_thresholds):
        rows = []
        for seed in SUPPORT_SEEDS:
            for shots in (3, 5):
                rows.extend(evaluate_recipe(dev_rows, recipe, shots=shots, seed=seed))
        if not rows:
            continue
        summary = summarize_rows(rows)
        zero = summary["zero"]
        adapted = summary["adapted"]
        fpr_increase = (adapted.get("false_positive_rate") or 0.0) - (zero.get("false_positive_rate") or 0.0)
        safe = (adapted.get("false_positive_rate") or 0.0) <= 0.03 and fpr_increase <= 0.02
        changed = any(bool(row["zero_decision"]) != bool(row["adapted_decision"]) for row in rows)
        item = {
            **recipe,
            "safe": safe,
            "changed_outputs": changed,
            "query_rows": len(rows),
            "zero_recall": zero.get("recall"),
            "zero_fpr": zero.get("false_positive_rate"),
            "zero_precision": zero.get("precision"),
            "zero_f1": zero.get("f1"),
            "adapted_recall": adapted.get("recall"),
            "adapted_fpr": adapted.get("false_positive_rate"),
            "adapted_precision": adapted.get("precision"),
            "adapted_f1": adapted.get("f1"),
            "delta_f1": summary["delta_f1"],
            "mean_paired_delta_f1": summary["paired_delta"].get("mean_delta_f1"),
        }
        search_rows.append(item)
        selectable = safe and changed and bool(recipe.get("support_based")) and (item["delta_f1"] is not None and item["delta_f1"] > 0)
        if selectable and (
            best is None
            or (
                item["delta_f1"],
                item["adapted_recall"] or 0.0,
                -(item["adapted_fpr"] or 0.0),
            )
            > (
                best["delta_f1"],
                best["adapted_recall"] or 0.0,
                -(best["adapted_fpr"] or 0.0),
            )
        ):
            best = item
    if best is None:
        return {"method": "no_adaptation_zero_shot_fallback", "support_based": False, "reason": "no_safe_support_based_f1_improvement"}, search_rows
    return best, search_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/latest_data/runs/nested_zero_shot")
    parser.add_argument("--reports", default="finetune/experiments/latest_data_optimization/reports")
    parser.add_argument("--qwen-cache", default="finetune/experiments/latest_data/shared/qwen_transcript_cache_balanced_max100_latest.jsonl")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_root = Path(args.run_root)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    selected = read_json(reports / "latest_opt_stage2_selected_config.json")
    qwen_exact = qwen_exact_by_clip(Path(args.qwen_cache))
    all_search_rows = []
    all_outer_rows = []
    selected_by_fold = {}
    for fold in range(5):
        dev_rows = build_clip_records(run_root, fold, "val", selected, qwen_exact, device)
        outer_rows = build_clip_records(run_root, fold, "test", selected, qwen_exact, device)
        recipe, search_rows = select_recipe(dev_rows, [float(selected["thresholds"][fold])])
        selected_by_fold[str(fold)] = recipe
        all_search_rows.extend({"fold": fold, **row} for row in search_rows)
        eval_recipe = recipe
        if recipe["method"] == "no_adaptation_zero_shot_fallback":
            # Still evaluate the strict support/query split; adapted equals paired zero-shot.
            eval_recipe = {"method": "support_threshold_calibration", "support_quantile": "median", "margin": -999.0, "max_threshold_drop": 0.0, "support_based": False}
        for seed in SUPPORT_SEEDS:
            for shots in (3, 5):
                rows = evaluate_recipe(outer_rows, eval_recipe, shots=shots, seed=seed)
                if recipe["method"] == "no_adaptation_zero_shot_fallback":
                    for row in rows:
                        row["adapted_decision"] = row["zero_decision"]
                all_outer_rows.extend({"fold": fold, **row} for row in rows)
    summary = {"status": "ok", "device": device, "selected_by_fold": selected_by_fold, "conditions": {}}
    for shots in (3, 5):
        subset = [row for row in all_outer_rows if int(row["shots"]) == shots]
        summary["conditions"][f"0-shot paired for {shots}-shot"] = summarize_rows([{**row, "adapted_decision": row["zero_decision"]} for row in subset])
        summary["conditions"][f"{shots}-shot"] = summarize_rows(subset)
    support_based_selected = any(row.get("support_based") and row.get("method") != "no_adaptation_zero_shot_fallback" for row in selected_by_fold.values())
    summary["support_based_selected"] = support_based_selected
    summary["claim"] = (
        "Real support-based onboarding selected by development pseudo-targets."
        if support_based_selected
        else "Real support-based onboarding was implemented and evaluated, but no safe improvement was found on the latest dataset."
    )
    participant_deltas = []
    for shots in (3, 5):
        subset = [row for row in all_outer_rows if int(row["shots"]) == shots]
        grouped = by_participant(subset)
        for alias, rows in grouped.items():
            item = paired_delta(rows, "zero_decision", "adapted_decision")
            participant_deltas.append({"participant_alias": alias, "shots": shots, **item})

    write_csv(reports / "latest_opt_fewshot_recipe_search.csv", all_search_rows)
    write_json(reports / "latest_opt_selected_fewshot_recipe.json", {"selected_by_fold": selected_by_fold, "support_based_selected": support_based_selected})
    write_json(reports / "latest_opt_real_few_shot_summary.json", summary)
    public_rows = [
        {
            key: row[key]
            for key in ("fold", "participant_alias", "shots", "support_seed", "label", "prompt_group", "zero_decision", "adapted_decision")
        }
        for row in all_outer_rows
    ]
    write_csv(reports / "latest_opt_real_few_shot_results.csv", public_rows)
    write_csv(reports / "latest_opt_real_few_shot_participant_deltas.csv", participant_deltas)

    lines = [
        "# Latest Optimized Few-Shot Recipe Search",
        "",
        f"- Device: `{device}`",
        "- Recipe selection uses each fold's validation participants only.",
        "- Target support uses positive clips only; target negatives and query positives are not used for adaptation.",
        f"- Support-based recipe selected: `{support_based_selected}`",
        f"- Claim: {summary['claim']}",
        "",
        "| Fold | Selected method | Reason or delta |",
        "|---:|---|---:|",
    ]
    for fold, recipe in selected_by_fold.items():
        lines.append(f"| {fold} | {recipe.get('method')} | {recipe.get('delta_f1', recipe.get('reason'))} |")
    (reports / "LATEST_OPT_FEWSHOT_RECIPE_SEARCH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    table = [
        "# Latest Optimized Real Few-Shot Onboarding",
        "",
        summary["claim"],
        "",
        "| Condition | Recall | FPR | Precision | F1 | Delta F1 vs paired 0-shot |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for shots in (3, 5):
        zero = summary["conditions"][f"0-shot paired for {shots}-shot"]["zero"]
        adapted = summary["conditions"][f"{shots}-shot"]["adapted"]
        delta = summary["conditions"][f"{shots}-shot"]["delta_f1"]
        table.append(f"| 0-shot paired for {shots}-shot | {zero.get('recall')} | {zero.get('false_positive_rate')} | {zero.get('precision')} | {zero.get('f1')} | - |")
        table.append(f"| {shots}-shot | {adapted.get('recall')} | {adapted.get('false_positive_rate')} | {adapted.get('precision')} | {adapted.get('f1')} | {delta} |")
    (reports / "LATEST_OPT_REAL_FEW_SHOT_ONBOARDING_REPORT.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print({"status": "ok", "support_based_selected": support_based_selected, "rows": len(all_outer_rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
