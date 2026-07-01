#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vigil_latest_opt.metrics import metrics_from_rows, participant_macro
from vigil_latest_opt.utils import read_json, read_jsonl, write_json
from vigil_two_stage.stage1_model import Stage1GRUClassifier
from vigil_two_stage.stage2_model import QwenVerifierHead


def load_npz(path: str) -> np.ndarray:
    data = np.load(path)
    return (data["features"] if "features" in data else data[data.files[0]]).astype(np.float32)


def load_stage1(run_dir: Path, device: str) -> Stage1GRUClassifier:
    cfg = read_json(run_dir / "stage1/model_config.json")
    model = Stage1GRUClassifier(cfg["input_dim"], cfg["gru_hidden_size"], cfg["gru_layers"], cfg["dropout"]).to(device)
    ckpt = torch.load(run_dir / "stage1/checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def load_stage2(run_dir: Path, variant: str, device: str) -> QwenVerifierHead:
    ckpt = torch.load(run_dir / variant / "checkpoint_best.pt", map_location=device)
    model = QwenVerifierHead(int(ckpt["input_dim"]), int(ckpt["config"]["projection_dim"]), int(ckpt["config"]["embedding_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def score_stage1(model: Stage1GRUClassifier, feature_path: str, device: str) -> float:
    arr = load_npz(feature_path)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    lengths = torch.tensor([arr.shape[0]], device=device)
    with torch.no_grad():
        return float(torch.sigmoid(model(x, lengths)).detach().cpu().item())


def score_stage2(model: QwenVerifierHead, feature_path: str, device: str) -> float:
    arr = load_npz(feature_path)
    hidden = torch.from_numpy(arr).unsqueeze(0).to(device)
    mask = torch.ones(1, arr.shape[0], dtype=torch.bool, device=device)
    with torch.no_grad():
        return float(torch.sigmoid(model(hidden, mask)["logit"]).detach().cpu().item())


def group_by_clip(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["clip_id"])].append(row)
    return grouped


def main() -> int:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reports = Path("finetune/experiments/latest_data_optimization/reports")
    selected = read_json(reports / "latest_opt_stage2_selected_config.json")
    folds = read_json("finetune/experiments/latest_data/shared/latest_participant_folds_5fold.json")["folds"]
    full_manifest = read_jsonl("finetune/experiments/latest_data/shared/full_unbalanced_latest_manifest.jsonl")
    alias_by_clip = {row["clip_id"]: row["participant_alias"] for row in full_manifest}
    full_clip_ids = {row["clip_id"] for row in full_manifest}
    stage1_features = [
        {**row, "participant_alias": alias_by_clip[row["clip_id"]]}
        for row in read_jsonl("finetune/experiments/latest_data/runs/latest_feature_cache_2b78e211183d47fb/stage1/features_manifest.jsonl")
        if row["clip_id"] in full_clip_ids
    ]
    qwen_features = {
        (row["clip_id"], int(row.get("window_index", 0))): {**row, "participant_alias": alias_by_clip[row["clip_id"]]}
        for row in read_jsonl("finetune/experiments/latest_data/runs/latest_feature_cache_2b78e211183d47fb/stage2_qwen_features/qwen_features_manifest.jsonl")
        if row["clip_id"] in full_clip_ids
    }
    qwen_cache = read_jsonl("finetune/experiments/latest_data/shared/qwen_transcript_cache_balanced_max100_latest.jsonl")
    qwen_cached_full = [row for row in qwen_cache if row["clip_id"] in full_clip_ids]
    all_stage1 = []
    all_stage2 = []
    fold_rows = []
    run_root = Path("finetune/experiments/latest_data/runs/nested_zero_shot")
    for fold in folds:
        fold_idx = int(fold["fold"])
        aliases = set(fold["participant_aliases"])
        run_dir = run_root / f"fold_{fold_idx}"
        theta1 = float(read_json(run_dir / "stage1/threshold.json")["threshold"])
        theta2 = float(selected["thresholds"][fold_idx])
        stage1 = load_stage1(run_dir, device)
        stage2 = load_stage2(run_dir, selected["variant"], device)
        rows = [row for row in stage1_features if row.get("participant_alias") in aliases]
        scored = []
        for row in rows:
            key = (row["clip_id"], int(row.get("window_index", 0)))
            qrow = qwen_features.get(key)
            if qrow is None:
                continue
            scored.append({**row, "stage1_score": score_stage1(stage1, row["feature_path"], device), "stage2_score": score_stage2(stage2, qrow["feature_path"], device)})
        for clip_id, group in group_by_clip(scored).items():
            ranked = sorted(group, key=lambda row: float(row["stage1_score"]), reverse=True)
            first = ranked[0]
            stage1_decision = float(ranked[0]["stage1_score"]) >= theta1
            candidates = [row for row in ranked if float(row["stage1_score"]) >= theta1][: int(selected["top_k"])]
            stage2_decision = any(float(row["stage2_score"]) >= theta2 for row in candidates)
            base = {
                "clip_id": clip_id,
                "participant_alias": first["participant_alias"],
                "label": int(first["label"]),
                "prompt_group": first.get("prompt_group"),
                "fold": fold_idx,
            }
            all_stage1.append({**base, "decision": stage1_decision})
            all_stage2.append({**base, "decision": stage2_decision})
        fold_rows.append({"fold": fold_idx, "clips": len({row["clip_id"] for row in rows}), "windows": len(rows)})
    balanced = selected["baseline"]
    summary = {
        "status": "partial_full_unbalanced_heads_only",
        "device": device,
        "reason_qwen_exact_full": "full-unbalanced Qwen exact was not rerun; only balanced transcript cache exists for 1346 clips",
        "full_unbalanced": {
            "clips": len({row["clip_id"] for row in full_manifest}),
            "windows": len(full_manifest),
            "qwen_exact_cached_clips": len({row["clip_id"] for row in qwen_cached_full}),
            "stage1_only": {"pooled": metrics_from_rows(all_stage1), "participant_macro": participant_macro(all_stage1)},
            "selected_stage2": {"pooled": metrics_from_rows(all_stage2), "participant_macro": participant_macro(all_stage2)},
            "fold_rows": fold_rows,
        },
        "balanced_reference": {
            "qwen_exact": balanced["qwen_exact"],
            "stage1_only": balanced["stage1_only_recomputed"],
            "selected_stage2": selected["test"],
        },
        "conclusion": "Partial head-only full-unbalanced ablation does not replace the balanced max-100 primary result.",
    }
    write_json(reports / "latest_opt_balanced_vs_full_summary.json", summary)
    lines = [
        "# Latest Optimized Balanced Versus Full-Data Ablation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Full-unbalanced clips/windows evaluated by heads: `{summary['full_unbalanced']['clips']}` / `{summary['full_unbalanced']['windows']}`",
        f"- Full-unbalanced Qwen exact status: {summary['reason_qwen_exact_full']}",
        "",
        "| Dataset | Method | Recall | FPR | Precision | F1 | Participant-macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (
        ("Balanced max-100", balanced["stage1_only_recomputed"]),
        ("Balanced max-100", selected["test"]),
        ("Full unbalanced", summary["full_unbalanced"]["stage1_only"]["pooled"]),
        ("Full unbalanced", summary["full_unbalanced"]["selected_stage2"]["pooled"]),
    ):
        method = "Stage1 only" if metrics is balanced["stage1_only_recomputed"] or metrics is summary["full_unbalanced"]["stage1_only"]["pooled"] else "Selected Stage2"
        macro = None
        if name == "Full unbalanced":
            macro = summary["full_unbalanced"]["stage1_only" if method == "Stage1 only" else "selected_stage2"]["participant_macro"].get("f1")
        lines.append(f"| {name} | {method} | {metrics.get('recall')} | {metrics.get('false_positive_rate')} | {metrics.get('precision')} | {metrics.get('f1')} | {macro} |")
    lines.append("\nPrimary result remains balanced max-100; this ablation is partial because full Qwen exact transcript cache was not generated.")
    (reports / "LATEST_OPT_BALANCED_VS_FULL_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": summary["status"], "clips": summary["full_unbalanced"]["clips"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
