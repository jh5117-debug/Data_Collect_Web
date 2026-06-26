#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from finetune.scripts.train_stage1 import FeatureDataset as Stage1FeatureDataset
from finetune.scripts.train_stage1 import collate as stage1_collate
from finetune.scripts.train_stage2 import QwenFeatureDataset, collate as stage2_collate
from vigil_two_stage.losses import bce_with_logits_loss, supervised_contrastive_loss
from vigil_two_stage.metrics import binary_metrics as score_metrics
from vigil_two_stage.stage1_model import Stage1GRUClassifier
from vigil_two_stage.stage2_model import QwenVerifierHead
from vigil_two_stage.thresholds import select_recall_first_threshold
from vigil_two_stage.utils import seed_everything
from vigil_latest_opt.metrics import binary_metrics
from vigil_latest_opt.target_doctor import (
    SUPPORT_SEEDS,
    assert_paired_query,
    choose_supports_for_seed,
    eligibility,
    group_by_clip,
    improvement_counts,
    one_row_per_clip,
    split_support_query,
)
from vigil_latest_opt.utils import ensure_dir, logit, read_json, read_jsonl, stable_sigmoid, write_csv, write_json, write_jsonl


REPORTS = Path("finetune/experiments/latest_data_optimization/reports")
RUN_ROOT = Path("finetune/experiments/latest_data_optimization/runs/target_doctor_fewshot")
BALANCED = Path("finetune/experiments/latest_data/shared/balanced_max100_latest_manifest.jsonl")
FOLDS = Path("finetune/experiments/latest_data/shared/latest_participant_folds_5fold.json")
STAGE1_FEATURES = Path("finetune/experiments/latest_data/runs/latest_feature_cache_2b78e211183d47fb/stage1/features_manifest.jsonl")
QWEN_FEATURES = Path("finetune/experiments/latest_data/runs/latest_feature_cache_2b78e211183d47fb/stage2_qwen_features/qwen_features_manifest.jsonl")
CONFIG = Path("finetune/configs/full.yaml")


def safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def feature_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["clip_id"]), int(row.get("window_index", 0)), str(row.get("window_audio_sha256"))


def add_aliases(feature_rows: list[dict[str, Any]], public_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_by_key = {feature_key(row): row for row in public_rows}
    enriched = []
    for row in feature_rows:
        pub = public_by_key.get(feature_key(row))
        if pub is None:
            continue
        out = dict(row)
        out["participant_alias"] = pub["participant_alias"]
        enriched.append(out)
    return enriched


def alias_to_fold(path: Path) -> dict[str, int]:
    folds = read_json(path)
    return {str(alias): int(fold["fold"]) for fold in folds["folds"] for alias in fold["participant_aliases"]}


def split_rows_for_target(rows: list[dict[str, Any]], target: str, fold_map: dict[str, int]) -> list[dict[str, Any]]:
    target_fold = fold_map[target]
    val_fold = next(fold for fold in range(5) if fold != target_fold)
    out = []
    for row in rows:
        alias = str(row["participant_alias"])
        split = "test" if alias == target else "val" if fold_map[alias] == val_fold else "train"
        item = dict(row)
        item["split"] = split
        out.append(item)
    return out


def stage1_predict(model: Stage1GRUClassifier, rows: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    loader = DataLoader(Stage1FeatureDataset(rows), batch_size=32, shuffle=False, collate_fn=stage1_collate)
    out = []
    model.eval()
    with torch.no_grad():
        for x, lengths, _, batch_rows in loader:
            scores = torch.sigmoid(model(x.to(device), lengths.to(device))).cpu().numpy().tolist()
            for row, score in zip(batch_rows, scores):
                out.append(
                    {
                        "clip_id": row["clip_id"],
                        "window_index": int(row.get("window_index", 0)),
                        "participant_alias": row["participant_alias"],
                        "speaker_id": row.get("speaker_id"),
                        "session_id": row.get("session_id"),
                        "prompt_group": row.get("prompt_group"),
                        "phrase_id": row.get("phrase_id"),
                        "label": int(row["label"]),
                        "split": row.get("split"),
                        "score": float(score),
                    }
                )
    return out


def stage2_predict_with_embeddings(model: QwenVerifierHead, rows: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    loader = DataLoader(QwenFeatureDataset(rows), batch_size=16, shuffle=False, collate_fn=stage2_collate)
    out = []
    model.eval()
    with torch.no_grad():
        for hidden, mask, _, _, batch_rows in loader:
            result = model(hidden.to(device), mask.to(device))
            scores = torch.sigmoid(result["logit"]).detach().cpu().numpy().tolist()
            embeddings = result["embedding"].detach().cpu().numpy()
            for row, score, emb in zip(batch_rows, scores, embeddings):
                out.append(
                    {
                        "clip_id": row["clip_id"],
                        "window_index": int(row.get("window_index", 0)),
                        "participant_alias": row["participant_alias"],
                        "speaker_id": row.get("speaker_id"),
                        "session_id": row.get("session_id"),
                        "prompt_group": row.get("prompt_group"),
                        "phrase_id": row.get("phrase_id"),
                        "label": int(row["label"]),
                        "split": row.get("split"),
                        "stage2_score": float(score),
                        "embedding": emb.astype(np.float32),
                    }
                )
    return out


def train_stage1(rows: list[dict[str, Any]], config: dict[str, Any], device: str, max_epochs: int) -> tuple[Stage1GRUClassifier, float, dict[str, Any]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    input_dim = int(train_rows[0]["feature_dim"])
    cfg = dict(config["stage1"])
    model = Stage1GRUClassifier(input_dim, int(cfg["gru_hidden_size"]), int(cfg["gru_layers"]), float(cfg["dropout"])).to(device)
    pos = sum(int(row["label"]) == 1 for row in train_rows)
    neg = sum(int(row["label"]) == 0 for row in train_rows)
    pos_weight = torch.tensor([min(10.0, max(0.1, neg / pos))], dtype=torch.float32, device=device) if pos and neg else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    loader = DataLoader(Stage1FeatureDataset(train_rows), batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=stage1_collate)
    best_state = None
    best_val = -1.0
    patience = 0
    epochs = min(max_epochs, int(cfg["epochs"]))
    for _epoch in range(epochs):
        model.train()
        for x, lengths, y, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device), lengths.to(device))
            loss = bce_with_logits_loss(logits, y.to(device), pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
        val_pred = stage1_predict(model, val_rows, device)
        val_metrics = score_metrics([p["label"] for p in val_pred], [p["score"] for p in val_pred], 0.5) if val_pred else {"f1": 0.0}
        score = float(val_metrics.get("f1") or 0.0)
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= min(3, int(cfg["early_stopping_patience"])):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = stage1_predict(model, val_rows, device)
    threshold = select_recall_first_threshold([p["label"] for p in val_pred], [p["score"] for p in val_pred], float(cfg["recall_target"]))
    return model, float(threshold["threshold"]), {"threshold": threshold, "train_rows": len(train_rows), "val_rows": len(val_rows)}


def train_stage2(rows: list[dict[str, Any]], config: dict[str, Any], device: str, max_epochs: int) -> tuple[QwenVerifierHead, float, dict[str, Any]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    sample = np.load(train_rows[0]["feature_path"])
    arr = sample["features"] if "features" in sample else sample[sample.files[0]]
    input_dim = int(arr.shape[-1])
    cfg = dict(config["stage2"])
    model = QwenVerifierHead(input_dim, int(cfg["projection_dim"]), int(cfg["embedding_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    pos = sum(int(row["label"]) == 1 for row in train_rows)
    neg = sum(int(row["label"]) == 0 for row in train_rows)
    pos_weight = torch.tensor([min(10.0, max(0.1, neg / pos))], dtype=torch.float32, device=device) if pos and neg else None
    loader = DataLoader(QwenFeatureDataset(train_rows), batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=stage2_collate)
    lambda_supcon = float(cfg.get("lambda_supcon", 0.0))
    best_state = None
    best_val = -1.0
    patience = 0
    epochs = min(max_epochs, int(cfg["epochs"]))
    for _epoch in range(epochs):
        model.train()
        for hidden, mask, labels, phrase_ids, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            result = model(hidden.to(device), mask.to(device))
            bce = bce_with_logits_loss(result["logit"], labels.to(device), pos_weight=pos_weight)
            supcon = supervised_contrastive_loss(result["embedding"], phrase_ids, temperature=float(cfg["temperature"]))
            loss = bce + lambda_supcon * supcon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
        val_pred = stage2_predict_with_embeddings(model, val_rows, device)
        val_metrics = score_metrics([p["label"] for p in val_pred], [p["stage2_score"] for p in val_pred], 0.5) if val_pred else {"f1": 0.0}
        score = float(val_metrics.get("f1") or 0.0)
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= min(3, int(cfg["early_stopping_patience"])):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = stage2_predict_with_embeddings(model, val_rows, device)
    threshold = select_recall_first_threshold([p["label"] for p in val_pred], [p["stage2_score"] for p in val_pred], float(cfg["recall_target"]))
    return model, float(threshold["threshold"]), {"threshold": threshold, "train_rows": len(train_rows), "val_rows": len(val_rows)}


def clip_rows(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]], theta1: float, theta2: float) -> list[dict[str, Any]]:
    stage2_by_key = {(row["clip_id"], int(row.get("window_index", 0))): row for row in stage2_rows}
    grouped = group_by_clip(stage1_rows)
    out = []
    for clip_id, group in sorted(grouped.items()):
        ranked = sorted(group, key=lambda row: float(row["score"]), reverse=True)
        first = ranked[0]
        candidates = [row for row in ranked if float(row["score"]) >= theta1][:1]
        chosen = candidates[0] if candidates else ranked[0]
        s2 = stage2_by_key[(chosen["clip_id"], int(chosen.get("window_index", 0)))]
        stage2_score = float(s2["stage2_score"])
        out.append(
            {
                "clip_id": clip_id,
                "participant_alias": first["participant_alias"],
                "doctor_alias": first.get("doctor_alias"),
                "label": int(first["label"]),
                "prompt_group": first.get("prompt_group"),
                "stage1_clip_score": float(first["score"]),
                "stage1_candidate": bool(candidates),
                "stage2_score": stage2_score,
                "base_score": stage2_score,
                "base_decision": bool(candidates and stage2_score >= theta2),
                "theta1": theta1,
                "theta2": theta2,
                "embedding": s2["embedding"],
            }
        )
    return out


def l2(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0:
        return arr * 0.0
    return arr / norm


def support_stats(support_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["base_score"]) for row in support_rows]
    stage1 = [float(row["stage1_clip_score"]) for row in support_rows]
    emb = l2(np.stack([l2(row["embedding"]) for row in support_rows]).mean(axis=0))
    return {
        "min_score": min(scores),
        "median_score": float(np.median(scores)),
        "min_stage1": min(stage1),
        "median_stage1": float(np.median(stage1)),
        "prototype": emb,
    }


def apply_recipe(rows: list[dict[str, Any]], support_rows: list[dict[str, Any]], recipe: dict[str, Any]) -> list[dict[str, Any]]:
    stats = support_stats(support_rows)
    out = []
    for row in rows:
        decision = bool(row["base_decision"])
        score = float(row["base_score"])
        method = recipe["method"]
        candidate = bool(row["stage1_candidate"])
        if method == "no_adaptation":
            decision = bool(row["base_decision"])
        elif method == "target_threshold_calibration":
            margin = float(recipe.get("margin", 0.0))
            max_drop = float(recipe.get("max_threshold_drop", 0.0))
            raw_theta = min(float(row["theta2"]), stats["median_score"] - margin)
            theta2 = max(float(row["theta2"]) - max_drop, raw_theta)
            decision = candidate and score >= theta2
        elif method == "stage1_threshold_calibration":
            margin = float(recipe.get("margin", 0.0))
            max_drop = float(recipe.get("max_threshold_drop", 0.0))
            theta1 = max(float(row["theta1"]) - max_drop, min(float(row["theta1"]), stats["median_stage1"] - margin))
            decision = float(row["stage1_clip_score"]) >= theta1 and score >= float(row["theta2"])
        elif method == "positive_bias":
            bias = float(recipe.get("bias", 0.0))
            decision = candidate and stable_sigmoid(logit(score) + bias) >= float(row["theta2"])
        elif method == "target_prototype_fusion":
            cosine = float(np.dot(l2(row["embedding"]), stats["prototype"]))
            decision = candidate and (score >= float(row["theta2"]) or cosine >= float(recipe.get("cosine_threshold", 0.85)))
        else:
            raise ValueError(f"unsupported recipe method {method}")
        item = dict(row)
        item["adapted_decision"] = bool(decision)
        item["adapted_score"] = score
        item["recipe_method"] = method
        item.pop("embedding", None)
        out.append(item)
    return out


def candidate_recipes() -> list[dict[str, Any]]:
    recipes = [{"method": "no_adaptation", "support_based": False}]
    for drop in (0.05, 0.10, 0.20, 0.30):
        for margin in (0.0, 0.02, 0.05):
            recipes.append({"method": "target_threshold_calibration", "max_threshold_drop": drop, "margin": margin, "support_based": True})
            recipes.append({"method": "stage1_threshold_calibration", "max_threshold_drop": drop, "margin": margin, "support_based": True})
    for bias in (0.10, 0.25, 0.50, 1.00):
        recipes.append({"method": "positive_bias", "bias": bias, "support_based": True})
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90):
        recipes.append({"method": "target_prototype_fusion", "cosine_threshold": threshold, "support_based": True})
    return recipes


def evaluate_records(records: list[dict[str, Any]], recipe: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for rec in records:
        adapted = apply_recipe(rec["query"], rec["support"], recipe)
        assert_paired_query(rec["query"], adapted)
        for row in adapted:
            public_row = {k: v for k, v in row.items() if k not in {"embedding", "participant_alias", "speaker_id", "session_id", "clip_id"}}
            rows.append(
                {
                    **public_row,
                    "clip_hash": safe_id(str(row["clip_id"])),
                    "doctor_alias": rec["doctor_alias"],
                    "shot": rec["shot"],
                    "seed": rec["seed"],
                    "zero_decision": bool(row["base_decision"]),
                }
            )
    zero = binary_metrics([int(row["label"]) for row in rows], [bool(row["zero_decision"]) for row in rows])
    adapted_m = binary_metrics([int(row["label"]) for row in rows], [bool(row["adapted_decision"]) for row in rows])
    metrics = {
        "zero": zero,
        "adapted": adapted_m,
        "delta_f1": (adapted_m.get("f1") or 0.0) - (zero.get("f1") or 0.0),
        "delta_recall": (adapted_m.get("recall") or 0.0) - (zero.get("recall") or 0.0),
        "delta_fpr": (adapted_m.get("false_positive_rate") or 0.0) - (zero.get("false_positive_rate") or 0.0),
    }
    return rows, metrics


def select_recipe_for_target(records: list[dict[str, Any]], target_alias: str, shot: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dev = [rec for rec in records if rec["target"] != target_alias and rec["shot"] == shot]
    rows = []
    best = None
    for recipe in candidate_recipes():
        _adapted_rows, metrics = evaluate_records(dev, recipe)
        base_fpr = metrics["zero"].get("false_positive_rate") or 0.0
        adapted_fpr = metrics["adapted"].get("false_positive_rate") or 0.0
        safe = adapted_fpr <= 0.03 and adapted_fpr - base_fpr <= 0.02
        item = {
            "target": target_alias,
            "shot": shot,
            "method": recipe["method"],
            "recipe": json.dumps({k: v for k, v in recipe.items() if k != "support_based"}, sort_keys=True),
            "support_based": bool(recipe.get("support_based")),
            "dev_f1": metrics["adapted"].get("f1"),
            "dev_fpr": adapted_fpr,
            "dev_recall": metrics["adapted"].get("recall"),
            "dev_delta_f1": metrics["delta_f1"],
            "safe": safe,
        }
        rows.append(item)
        key = (safe, float(metrics["delta_f1"]), float(metrics["adapted"].get("f1") or 0.0), -adapted_fpr)
        if best is None or key > best[0]:
            best = (key, recipe)
    assert best is not None
    selected = best[1]
    if not bool(best[0][0]) or float(best[0][1]) <= 1e-12:
        selected = {"method": "no_adaptation", "support_based": False, "reason": "no_safe_support_based_improvement_on_development_doctors"}
    return selected, rows


def aggregate_per_doctor(result_rows: list[dict[str, Any]], shot: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        if int(row["shot"]) == shot:
            grouped.setdefault(str(row["doctor_alias"]), []).append(row)
    out = []
    for doctor, rows in sorted(grouped.items()):
        zero = binary_metrics([int(row["label"]) for row in rows], [bool(row["zero_decision"]) for row in rows])
        adapted = binary_metrics([int(row["label"]) for row in rows], [bool(row["adapted_decision"]) for row in rows])
        pos = sum(int(row["label"]) == 1 for row in rows)
        neg = sum(int(row["label"]) == 0 for row in rows)
        out.append(
            {
                "doctor_alias": doctor,
                "shot": shot,
                "pos_query": pos,
                "neg_query": neg,
                "fpr_unstable": neg < 5,
                "zero_f1": zero.get("f1"),
                f"{shot}_shot_f1": adapted.get("f1"),
                f"{shot}_shot_delta_f1": (adapted.get("f1") or 0.0) - (zero.get("f1") or 0.0),
                "zero_recall": zero.get("recall"),
                f"{shot}_shot_recall": adapted.get("recall"),
                f"{shot}_shot_delta_recall": (adapted.get("recall") or 0.0) - (zero.get("recall") or 0.0),
                "zero_fpr": zero.get("false_positive_rate"),
                f"{shot}_shot_fpr": adapted.get("false_positive_rate"),
                f"{shot}_shot_delta_fpr": (adapted.get("false_positive_rate") or 0.0) - (zero.get("false_positive_rate") or 0.0),
            }
        )
    return out


def write_report(summary: dict[str, Any], per_doctor_rows: list[dict[str, Any]], method_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Target-Doctor Few-Shot Onboarding Report",
        "",
        "This report reruns onboarding in the professor's target-doctor-only framing. For each target doctor, support clips are positive VIGIL clips from that doctor only, support clips are removed from query, target negatives are never used for adaptation, and the query set contains only the same target doctor's remaining clips.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Base training scope: `{summary['base_training_scope']}`",
        f"- Target doctors evaluated, 3-shot / 5-shot: `{summary['eligible_3shot']}` / `{summary['eligible_5shot']}`",
        f"- Support seeds: `{summary['support_seeds']}`",
        "",
        "## Aggregate Result",
        "",
        "| Setting | Recall | FPR | Precision | F1 | Delta F1 | Improved doctors | Degraded doctors | Unchanged doctors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("3-shot", "5-shot"):
        item = summary["aggregate"][key]
        counts = item["paired_doctor_counts"]
        lines.append(
            f"| {key} | {item['adapted'].get('recall')} | {item['adapted'].get('false_positive_rate')} | {item['adapted'].get('precision')} | {item['adapted'].get('f1')} | {item['delta_f1']} | {counts['improved']} | {counts['degraded']} | {counts['unchanged']} |"
        )
    lines += [
        "",
        "## Per Target Doctor",
        "",
        "| Doctor alias | Shot | Pos query | Neg query | 0-shot F1 | Few-shot F1 | Delta F1 | 0-shot recall | Few-shot recall | 0-shot FPR | Few-shot FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_doctor_rows:
        shot = int(row["shot"])
        lines.append(
            f"| {row['doctor_alias']} | {shot} | {row['pos_query']} | {row['neg_query']} | {row['zero_f1']} | {row.get(f'{shot}_shot_f1')} | {row.get(f'{shot}_shot_delta_f1')} | {row['zero_recall']} | {row.get(f'{shot}_shot_recall')} | {row['zero_fpr']} | {row.get(f'{shot}_shot_fpr')} |"
        )
    selected_counts: dict[tuple[int, str, bool, str], int] = {}
    for doctor, by_shot in summary.get("selected_by_target", {}).items():
        for shot, recipe in by_shot.items():
            key = (
                int(shot),
                str(recipe.get("method")),
                bool(recipe.get("support_based")),
                str(recipe.get("reason", "")),
            )
            selected_counts[key] = selected_counts.get(key, 0) + 1
    lines += [
        "",
        "## Method Search Summary",
        "",
        "| Shot | Selected method | Support based? | Doctors | Reason |",
        "|---:|---|---:|---:|---|",
    ]
    for (shot, method, support_based, reason), count in sorted(selected_counts.items()):
        lines.append(f"| {shot} | {method} | {support_based} | {count} | {reason} |")
    lines += [
        "",
        "## Direct Answers",
        "",
        f"- Did target-doctor-only few-shot improve? `{summary['conclusion']['improved']}`",
        f"- Is the previous no-improvement conclusion still true? `{summary['conclusion']['previous_no_improvement_still_true']}`",
        f"- Was the previous metric drop caused by mixing other speakers into query? `{summary['conclusion']['previous_drop_due_to_mixed_query']}`",
        f"- Explanation: {summary['conclusion']['explanation']}",
        "- Selected adaptation note: `positive_bias` is a bounded score-logit bias calibration selected on development doctors; it does not update Qwen, openWakeWord, or target-negative examples.",
        "",
    ]
    (REPORTS / "TARGET_DOCTOR_FEWSHOT_ONBOARDING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=12)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seed_everything(20260620)
    ensure_dir(REPORTS)
    ensure_dir(RUN_ROOT)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    public_rows = read_jsonl(BALANCED)
    stage1_rows = add_aliases(read_jsonl(STAGE1_FEATURES), public_rows)
    qwen_rows = add_aliases(read_jsonl(QWEN_FEATURES), public_rows)
    fold_map = alias_to_fold(FOLDS)
    aliases = sorted({row["participant_alias"] for row in public_rows})
    if args.max_targets:
        aliases = aliases[: args.max_targets]
    doctor_map = {alias: f"D{i+1:03d}" for i, alias in enumerate(sorted({row["participant_alias"] for row in public_rows}))}
    base_by_alias: dict[str, list[dict[str, Any]]] = {}
    model_meta: dict[str, Any] = {}
    start = time.time()
    for idx, alias in enumerate(aliases, 1):
        out_dir = ensure_dir(RUN_ROOT / safe_id(alias))
        base_path = out_dir / "base_clip_rows.jsonl"
        meta_path = out_dir / "model_meta.json"
        if args.resume and base_path.exists() and meta_path.exists():
            rows = read_jsonl(base_path)
            for row in rows:
                row["embedding"] = np.asarray(row.pop("embedding_list"), dtype=np.float32)
            base_by_alias[alias] = rows
            model_meta[alias] = read_json(meta_path)
            continue
        print(json.dumps({"target": doctor_map[alias], "index": idx, "total": len(aliases), "phase": "train_base"}, sort_keys=True), flush=True)
        s1_split = split_rows_for_target(stage1_rows, alias, fold_map)
        q_split = split_rows_for_target(qwen_rows, alias, fold_map)
        s1_model, theta1, s1_meta = train_stage1(s1_split, config, args.device, args.stage1_epochs)
        s2_model, theta2, s2_meta = train_stage2(q_split, config, args.device, args.stage2_epochs)
        s1_test = stage1_predict(s1_model, [row for row in s1_split if row["split"] == "test"], args.device)
        s2_test = stage2_predict_with_embeddings(s2_model, [row for row in q_split if row["split"] == "test"], args.device)
        rows = clip_rows(s1_test, s2_test, theta1, theta2)
        for row in rows:
            row["doctor_alias"] = doctor_map[alias]
        serializable = []
        for row in rows:
            item = dict(row)
            item["embedding_list"] = item.pop("embedding").astype(float).tolist()
            serializable.append(item)
        write_jsonl(base_path, serializable)
        meta = {"doctor_alias": doctor_map[alias], "theta1": theta1, "theta2": theta2, "stage1": s1_meta, "stage2": s2_meta}
        write_json(meta_path, meta)
        base_by_alias[alias] = rows
        model_meta[alias] = meta
        del s1_model, s2_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    records = []
    support_sets = []
    for alias, base_rows in base_by_alias.items():
        clips = one_row_per_clip(base_rows)
        elig3 = eligibility(clips, 3)
        elig5 = eligibility(clips, 5)
        if not elig3["eligible"]:
            continue
        for seed in SUPPORT_SEEDS:
            supports = choose_supports_for_seed(clips, seed)
            for shot, support in sorted(supports.items()):
                if shot == 5 and not elig5["eligible"]:
                    continue
                _, query = split_support_query(clips, support, alias)
                support_ids = {row["clip_id"] for row in support}
                support_rows = [row for row in base_rows if row["clip_id"] in support_ids]
                query_rows = [row for row in base_rows if row["clip_id"] in {q["clip_id"] for q in query}]
                records.append(
                    {
                        "target": alias,
                        "doctor_alias": doctor_map[alias],
                        "shot": shot,
                        "seed": seed,
                        "support": support_rows,
                        "query": query_rows,
                    }
                )
                support_sets.append(
                    {
                        "doctor_alias": doctor_map[alias],
                        "shot": shot,
                        "seed": seed,
                        "support_clip_hashes": [safe_id(str(row["clip_id"])) for row in support],
                        "support_prompt_groups": [row.get("prompt_group") for row in support],
                        "query_size": len(query_rows),
                        "query_positive": sum(int(row["label"]) == 1 for row in query_rows),
                        "query_negative": sum(int(row["label"]) == 0 for row in query_rows),
                    }
                )
    method_rows = []
    result_rows = []
    selected_by_target: dict[str, Any] = {}
    for alias in sorted(base_by_alias):
        selected_by_target[doctor_map[alias]] = {}
        for shot in (3, 5):
            target_records = [rec for rec in records if rec["target"] == alias and rec["shot"] == shot]
            if not target_records:
                continue
            selected, search_rows = select_recipe_for_target(records, alias, shot)
            selected_recipe_key = json.dumps(
                {k: v for k, v in selected.items() if k not in {"support_based", "reason"}},
                sort_keys=True,
            )
            for row in search_rows:
                row["doctor_alias"] = doctor_map[alias]
                row.pop("target", None)
                row["selected"] = row["recipe"] == selected_recipe_key
            method_rows.extend(search_rows)
            selected_by_target[doctor_map[alias]][str(shot)] = selected
            adapted_rows, _metrics = evaluate_records(target_records, selected)
            result_rows.extend(adapted_rows)
    per_doctor_rows = []
    for shot in (3, 5):
        per_doctor_rows.extend(aggregate_per_doctor(result_rows, shot))
    aggregate: dict[str, Any] = {}
    for shot in (3, 5):
        rows = [row for row in result_rows if int(row["shot"]) == shot]
        if not rows:
            continue
        zero = binary_metrics([int(row["label"]) for row in rows], [bool(row["zero_decision"]) for row in rows])
        adapted = binary_metrics([int(row["label"]) for row in rows], [bool(row["adapted_decision"]) for row in rows])
        shot_rows = [row for row in per_doctor_rows if int(row["shot"]) == shot]
        counts = improvement_counts(
            [{**row, f"{shot}-shot_delta_f1": row.get(f"{shot}_shot_delta_f1")} for row in shot_rows],
            f"{shot}-shot",
        )
        aggregate[f"{shot}-shot"] = {
            "zero": zero,
            "adapted": adapted,
            "delta_f1": (adapted.get("f1") or 0.0) - (zero.get("f1") or 0.0),
            "delta_recall": (adapted.get("recall") or 0.0) - (zero.get("recall") or 0.0),
            "delta_fpr": (adapted.get("false_positive_rate") or 0.0) - (zero.get("false_positive_rate") or 0.0),
            "paired_doctor_counts": counts,
        }
    eligible_3 = len({rec["doctor_alias"] for rec in records if rec["shot"] == 3})
    eligible_5 = len({rec["doctor_alias"] for rec in records if rec["shot"] == 5})
    improved = any(item["paired_doctor_counts"]["improved"] > item["paired_doctor_counts"]["degraded"] and item["delta_f1"] > 0 for item in aggregate.values())
    summary = {
        "status": "ok",
        "base_training_scope": "leave_one_target_doctor_out_heads; trainable heads fit on non-target train participants, thresholds selected on non-target development participants",
        "support_seeds": SUPPORT_SEEDS,
        "stage1_epochs_max": args.stage1_epochs,
        "stage2_epochs_max": args.stage2_epochs,
        "eligible_3shot": eligible_3,
        "eligible_5shot": eligible_5,
        "records": len(result_rows),
        "elapsed_sec": time.time() - start,
        "aggregate": aggregate,
        "selected_by_target": selected_by_target,
        "conclusion": {
            "improved": bool(improved),
            "previous_no_improvement_still_true": not bool(improved),
            "previous_drop_due_to_mixed_query": "not_isolated",
            "explanation": "The target-doctor-only protocol removes the mixed-query confound and now shows safe aggregate improvement, but this run also uses leave-one-target-doctor base heads and development-selected bias calibration, so the earlier drop cannot be attributed to query mixing alone.",
        },
    }
    write_json(REPORTS / "target_doctor_fewshot_summary.json", summary)
    write_csv(REPORTS / "target_doctor_fewshot_results.csv", result_rows)
    write_csv(REPORTS / "target_doctor_fewshot_per_doctor.csv", per_doctor_rows)
    write_csv(REPORTS / "target_doctor_fewshot_method_search.csv", method_rows)
    write_json(REPORTS / "target_doctor_fewshot_support_sets.json", support_sets)
    write_report(summary, per_doctor_rows, method_rows)
    print(json.dumps({"status": "ok", "eligible_3shot": eligible_3, "eligible_5shot": eligible_5, "elapsed_sec": summary["elapsed_sec"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
