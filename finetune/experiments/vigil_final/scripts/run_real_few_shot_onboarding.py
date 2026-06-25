#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from finetune.evaluation.aggregate_clip_predictions import aggregate_clip_cascade_predictions
from vigil_final.metrics import metric_from_decisions, paired_delta, participant_macro
from vigil_final.prototype import build_prototype, cosine_similarity
from vigil_final.safety import fpr_safety_gate
from vigil_final.support import choose_nested_supports
from vigil_final.utils import read_json, read_jsonl, write_csv, write_json
from vigil_two_stage.stage2_model import QwenVerifierHead


SUPPORT_SEEDS = [20260620, 20260621, 20260622, 20260623, 20260624]


def pred_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["clip_id"]), int(row.get("window_index", 0))


def load_npz(path: str) -> np.ndarray:
    data = np.load(path)
    return (data["features"] if "features" in data else data[data.files[0]]).astype(np.float32)


def load_stage2_model(checkpoint: Path, device: str) -> QwenVerifierHead:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = QwenVerifierHead(int(ckpt["input_dim"]), int(ckpt["config"]["projection_dim"]), int(ckpt["config"]["embedding_dim"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def embed_windows(run_dir: Path, variant: str, split: str, device: str) -> dict[tuple[str, int], dict[str, Any]]:
    model = load_stage2_model(run_dir / variant / "checkpoint_best.pt", device)
    rows = [row for row in read_jsonl(run_dir / "stage2_qwen_features" / "qwen_features_manifest.jsonl") if row["split"] == split]
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with torch.no_grad():
        for row in rows:
            arr = load_npz(row["feature_path"])
            hidden = torch.from_numpy(arr).unsqueeze(0).to(device)
            mask = torch.ones(1, arr.shape[0], dtype=torch.bool, device=device)
            result = model(hidden, mask)
            embedding = result["embedding"].detach().cpu().numpy()[0]
            out[pred_key(row)] = {
                "clip_id": row["clip_id"],
                "window_index": int(row.get("window_index", 0)),
                "participant_alias": row["participant_alias"],
                "label": int(row["label"]),
                "prompt_group": row.get("prompt_group"),
                "embedding": embedding,
                "stage2_score": float(torch.sigmoid(result["logit"]).detach().cpu().item()),
            }
    return out


def build_window_records(run_dir: Path, variant: str, split: str, theta1: float, theta2: float, device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage1 = [row for row in read_jsonl(run_dir / "stage1" / f"{split}_predictions.jsonl")]
    stage2 = [row for row in read_jsonl(run_dir / variant / f"{split}_predictions.jsonl")]
    embeddings = embed_windows(run_dir, variant, split, device)
    stage2_by_key = {pred_key(row): row for row in stage2}
    records = []
    for row in stage1:
        key = pred_key(row)
        emb = embeddings.get(key)
        s2 = stage2_by_key.get(key)
        if emb is None or s2 is None:
            continue
        records.append(
            {
                "clip_id": row["clip_id"],
                "window_index": int(row.get("window_index", 0)),
                "participant_alias": emb["participant_alias"],
                "label": int(row["label"]),
                "prompt_group": row.get("prompt_group"),
                "stage1_score": float(row["score"]),
                "stage2_score": float(s2["stage2_score"]),
                "embedding": emb["embedding"],
                "candidate": float(row["score"]) >= theta1,
            }
        )
    cascade = aggregate_clip_cascade_predictions(stage1, stage2, theta1, theta2, top_k=3)
    return records, cascade


def group_by_clip(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["clip_id"])].append(row)
    return grouped


def clip_rows_for_support(records: list[dict[str, Any]], cascade: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows_by_clip = group_by_clip(records)
    zero_by_clip = {str(row["clip_id"]): bool(row["final_trigger"]) for row in cascade}
    return [
        {
            "clip_id": clip_id,
            "participant_alias": group[0]["participant_alias"],
            "label": int(group[0]["label"]),
            "prompt_group": group[0].get("prompt_group"),
            "zero_decision": zero_by_clip.get(clip_id, False),
        }
        for clip_id, group in sorted(windows_by_clip.items())
    ]


def selected_support_embedding(clip_windows: list[dict[str, Any]]) -> np.ndarray | None:
    if not clip_windows:
        return None
    candidates = [row for row in clip_windows if row["candidate"]]
    pool = candidates if candidates else clip_windows
    best = sorted(pool, key=lambda row: float(row["stage1_score"]), reverse=True)[0]
    return np.asarray(best["embedding"], dtype=np.float32)


def adapted_clip_score(clip_windows: list[dict[str, Any]], prototype: np.ndarray, recipe: dict[str, Any]) -> tuple[float, bool]:
    candidates = sorted([row for row in clip_windows if row["candidate"]], key=lambda row: float(row["stage1_score"]), reverse=True)[: int(recipe["top_k"])]
    if not candidates:
        return 0.0, False
    best_score = -1e9
    for row in candidates:
        sim = cosine_similarity(np.asarray(row["embedding"], dtype=np.float32), prototype)
        if recipe["method"] == "prototype_only":
            score = sim
        else:
            score = float(row["stage2_score"]) + float(recipe["alpha"]) * (sim - float(recipe["beta"]))
        best_score = max(best_score, score)
    return float(best_score), bool(best_score >= float(recipe["threshold"]))


def evaluate_recipe(records: list[dict[str, Any]], cascade: list[dict[str, Any]], recipe: dict[str, Any], shots: int, seed: int) -> list[dict[str, Any]]:
    windows_by_clip = group_by_clip(records)
    clip_rows = clip_rows_for_support(records, cascade)
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        by_participant[str(row["participant_alias"])].append(row)
    out = []
    for alias, rows in sorted(by_participant.items()):
        try:
            supports = choose_nested_supports(rows, seed=seed)
        except ValueError:
            continue
        support, query = supports[shots]
        support_embeddings = []
        for row in support:
            emb = selected_support_embedding(windows_by_clip[str(row["clip_id"])])
            if emb is not None:
                support_embeddings.append(emb)
        if len(support_embeddings) != shots:
            continue
        prototype = build_prototype(support_embeddings)
        query_ids = {str(row["clip_id"]) for row in query}
        zero_by_clip = {str(row["clip_id"]): bool(row["zero_decision"]) for row in query}
        for clip_id in sorted(query_ids):
            wins = windows_by_clip.get(clip_id, [])
            if not wins:
                continue
            score, decision = adapted_clip_score(wins, prototype, recipe)
            first = wins[0]
            out.append(
                {
                    "participant_alias": alias,
                    "clip_id": clip_id,
                    "label": int(first["label"]),
                    "prompt_group": first.get("prompt_group"),
                    "shots": shots,
                    "support_seed": seed,
                    "zero_decision": zero_by_clip[clip_id],
                    "adapted_decision": decision,
                    "adapted_score": score,
                }
            )
    return out


def search_recipes(development_sets: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    recipe_grid = []
    for top_k in (1, 3):
        for threshold in np.linspace(0.55, 0.95, 9):
            recipe_grid.append({"method": "prototype_only", "alpha": 0.0, "beta": 0.0, "threshold": float(threshold), "top_k": top_k})
        for alpha in (0.25, 0.5, 1.0, 2.0):
            for threshold in np.linspace(0.50, 1.30, 17):
                recipe_grid.append({"method": "base_plus_prototype", "alpha": alpha, "beta": 0.0, "threshold": float(threshold), "top_k": top_k})
    for recipe in recipe_grid:
        rows = []
        for records, cascade in development_sets:
            for seed in SUPPORT_SEEDS:
                for shots in (3, 5):
                    rows.extend(evaluate_recipe(records, cascade, recipe, shots, seed))
        if not rows:
            continue
        baseline = metric_from_decisions([{**row, "decision": row["zero_decision"]} for row in rows], "decision")
        adapted = metric_from_decisions([{**row, "decision": row["adapted_decision"]} for row in rows], "decision")
        safety = fpr_safety_gate(baseline, adapted, max_absolute_fpr=0.02, max_fpr_increase=0.02)
        macro = participant_macro([{**row, "decision": row["adapted_decision"]} for row in rows], "decision")
        candidates.append(
            {
                **recipe,
                "query_rows": len(rows),
                "baseline_fpr": baseline.get("false_positive_rate"),
                "adapted_fpr": adapted.get("false_positive_rate"),
                "adapted_recall": adapted.get("recall"),
                "adapted_f1": adapted.get("f1"),
                "participant_macro_recall": macro.get("recall"),
                "participant_macro_f1": macro.get("f1"),
                "safety_passed": safety["passed"],
                "fpr_increase": safety["fpr_increase"],
            }
        )
    safe = [row for row in candidates if row["safety_passed"]]
    if not safe:
        return {"selected_recipe": "no_adaptation_zero_shot_fallback", "reason": "no_safe_prototype_recipe"}, candidates
    selected = sorted(
        safe,
        key=lambda row: (
            float(row.get("participant_macro_recall") or 0.0),
            float(row.get("participant_macro_f1") or 0.0),
            -float(row.get("adapted_fpr") or 0.0),
        ),
        reverse=True,
    )[0]
    selected["selected_recipe"] = "prototype_personalization"
    return selected, candidates


def summarize_outer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"status": "ok", "conditions": {}}
    for shots in (3, 5):
        subset = [row for row in rows if row["shots"] == shots]
        zero_subset = [{**row, "decision": row["zero_decision"]} for row in subset]
        adapted_subset = [{**row, "decision": row["adapted_decision"]} for row in subset]
        out["conditions"][f"0_for_{shots}"] = {
            "pooled": metric_from_decisions(zero_subset, "decision") if zero_subset else {"n": 0},
            "participant_macro": participant_macro(zero_subset, "decision") if zero_subset else {"n": 0},
        }
        out["conditions"][str(shots)] = {
            "pooled": metric_from_decisions(adapted_subset, "decision") if adapted_subset else {"n": 0},
            "participant_macro": participant_macro(adapted_subset, "decision") if adapted_subset else {"n": 0},
            "paired_delta": paired_delta(subset, "zero_decision", "adapted_decision"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/vigil_final/runs/nested_v2")
    parser.add_argument("--reports", default="finetune/experiments/vigil_final/reports")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_root = Path(args.run_root)
    reports = Path(args.reports)
    all_outer_rows = []
    all_search_rows = []
    selected_by_fold = {}
    for fold in range(5):
        result = read_json(run_root / f"outer_{fold}" / "nested_outer_result.json")
        variant = result["selected_method"]
        theta1 = float(result["theta_1_selection"]["threshold"])
        theta2 = float(result["stage2_selections"][variant.replace("stage2_", "")]["threshold_selection"]["threshold"])
        development_sets = []
        for inner in result["inner_runs"]:
            inner_dir = run_root / f"outer_{fold}" / f"inner_val_{inner['inner_validation_fold']}"
            records, cascade = build_window_records(inner_dir, variant, "val", theta1, theta2, device)
            development_sets.append((records, cascade))
        selected, candidates = search_recipes(development_sets)
        selected_by_fold[str(fold)] = selected
        for row in candidates:
            all_search_rows.append({"outer_fold": fold, **row})
        refit_dir = run_root / f"outer_{fold}" / "refit"
        records, cascade = build_window_records(refit_dir, variant, "test", theta1, theta2, device)
        if selected.get("selected_recipe") == "prototype_personalization":
            recipe = selected
            for seed in SUPPORT_SEEDS:
                for shots in (3, 5):
                    all_outer_rows.extend(evaluate_recipe(records, cascade, recipe, shots, seed))
        else:
            recipe = {"method": "base_plus_prototype", "alpha": 0.0, "beta": 0.0, "threshold": 999.0, "top_k": 3}
            rows = []
            for seed in SUPPORT_SEEDS:
                for shots in (3, 5):
                    evaluated = evaluate_recipe(records, cascade, recipe, shots, seed)
                    for row in evaluated:
                        row["adapted_decision"] = row["zero_decision"]
                    rows.extend(evaluated)
            all_outer_rows.extend(rows)
    summary = summarize_outer(all_outer_rows)
    summary["selected_by_fold"] = selected_by_fold
    summary["learned_personalization_claimed"] = any(v.get("selected_recipe") == "prototype_personalization" for v in selected_by_fold.values())
    write_csv(reports / "development_prototype_search.csv", all_search_rows)
    write_json(reports / "development_selected_prototype_recipe.json", {"selected_by_fold": selected_by_fold})
    write_json(reports / "real_few_shot_summary.json", summary)
    public_rows = [
        {
            "participant_alias": row["participant_alias"],
            "shots": row["shots"],
            "support_seed": row["support_seed"],
            "label": row["label"],
            "prompt_group": row.get("prompt_group"),
            "zero_decision": row["zero_decision"],
            "adapted_decision": row["adapted_decision"],
        }
        for row in all_outer_rows
    ]
    write_csv(reports / "real_few_shot_results.csv", public_rows)
    lines = [
        "# Real Few-Shot Onboarding Report",
        "",
        f"- Device: `{device}`",
        f"- Learned personalization claimed: `{summary['learned_personalization_claimed']}`",
        "- Support clips are positive-only and removed from paired query sets.",
        "- Development pseudo-targets select the recipe; outer-test participants are reporting-only.",
        "",
        "| Condition | Recall | FPR | Precision | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for shots in ("0_for_3", "3", "0_for_5", "5"):
        m = summary["conditions"][shots]["pooled"]
        label = {"0_for_3": "0-shot on 3-shot query", "0_for_5": "0-shot on 5-shot query"}.get(shots, f"{shots}-shot")
        lines.append(f"| {label} | {m.get('recall')} | {m.get('false_positive_rate')} | {m.get('precision')} | {m.get('f1')} |")
    (reports / "REAL_FEW_SHOT_ONBOARDING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": "ok", "rows": len(all_outer_rows), "learned": summary["learned_personalization_claimed"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
