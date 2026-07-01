#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from vigil_latest_opt.cascade import apply_threshold, clip_score_rows, stage1_only_rows
from vigil_latest_opt.metrics import group_metrics, metrics_from_rows
from vigil_latest_opt.thresholds import detailed_metrics, threshold_for_recall_target, threshold_for_safe_f1
from vigil_latest_opt.utils import read_json, read_jsonl, write_csv, write_json


VARIANTS = ("stage2_bce", "stage2_bce_supcon")
RECALL_TARGETS = (0.85, 0.90, 0.92, 0.95)


def fold_dir(run_root: Path, fold: int) -> Path:
    return run_root / f"fold_{fold}"


def theta1_for_fold(run_root: Path, fold: int) -> float:
    return float(read_json(fold_dir(run_root, fold) / "stage1/threshold.json")["threshold"])


def load_split(run_root: Path, fold: int, variant: str, split: str, *, top_k: int, fusion_a: float, fusion_b: float, use_fusion_logit: bool) -> list[dict[str, Any]]:
    root = fold_dir(run_root, fold)
    return clip_score_rows(
        read_jsonl(root / "stage1" / f"{split}_predictions.jsonl"),
        read_jsonl(root / variant / f"{split}_predictions.jsonl"),
        theta1=theta1_for_fold(run_root, fold),
        top_k=top_k,
        fusion_a=fusion_a,
        fusion_b=fusion_b,
        use_fusion_logit=use_fusion_logit,
    )


def summarize_by_key(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["|".join(str(row[key]) for key in keys)].append(row)
    out = {}
    for key, group in grouped.items():
        out[key] = {}
        for metric in ("dev_recall", "dev_false_positive_rate", "dev_precision", "dev_f1", "test_recall", "test_false_positive_rate", "test_precision", "test_f1"):
            vals = [float(row[metric]) for row in group if row.get(metric) is not None]
            out[key][metric] = sum(vals) / len(vals) if vals else None
        out[key]["folds"] = len(group)
    return out


def operating_points(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    decisions_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold in range(5):
        for variant in VARIANTS:
            dev_rows = load_split(run_root, fold, variant, "val", top_k=3, fusion_a=1.0, fusion_b=0.0, use_fusion_logit=False)
            test_rows = load_split(run_root, fold, variant, "test", top_k=3, fusion_a=1.0, fusion_b=0.0, use_fusion_logit=False)
            for target in RECALL_TARGETS:
                selection = threshold_for_recall_target(dev_rows, target)
                threshold = float(selection["threshold"])
                dev_metrics = detailed_metrics(dev_rows, threshold)
                test_metrics = detailed_metrics(test_rows, threshold)
                row = {
                    "fold": fold,
                    "variant": variant,
                    "recall_target": target,
                    "threshold": threshold,
                    "selection_reason": selection["reason"],
                    "selection_split": "val",
                    "dev_recall": dev_metrics.get("recall"),
                    "dev_false_positive_rate": dev_metrics.get("false_positive_rate"),
                    "dev_precision": dev_metrics.get("precision"),
                    "dev_f1": dev_metrics.get("f1"),
                    "dev_p1_recall": dev_metrics.get("p1_recall"),
                    "dev_p2_recall": dev_metrics.get("p2_recall"),
                    "dev_p3_recall": dev_metrics.get("p3_recall"),
                    "dev_p4_false_positive_rate": dev_metrics.get("p4_false_positive_rate"),
                    "dev_hard_negative_false_positive_rate": dev_metrics.get("hard_negative_false_positive_rate"),
                    "dev_stage2_rejection_rate": dev_metrics.get("stage2_rejection_rate"),
                    "test_recall": test_metrics.get("recall"),
                    "test_false_positive_rate": test_metrics.get("false_positive_rate"),
                    "test_precision": test_metrics.get("precision"),
                    "test_f1": test_metrics.get("f1"),
                    "test_p1_recall": test_metrics.get("p1_recall"),
                    "test_p2_recall": test_metrics.get("p2_recall"),
                    "test_p3_recall": test_metrics.get("p3_recall"),
                    "test_p4_false_positive_rate": test_metrics.get("p4_false_positive_rate"),
                    "test_hard_negative_false_positive_rate": test_metrics.get("hard_negative_false_positive_rate"),
                    "test_stage2_rejection_rate": test_metrics.get("stage2_rejection_rate"),
                }
                rows.append(row)
                mode_key = f"{variant}_target_{target:.2f}"
                decisions_by_mode[mode_key].extend(apply_threshold(test_rows, threshold))
    pooled = {
        key: metrics_from_rows(value)
        for key, value in decisions_by_mode.items()
    }
    return rows, {"status": "ok", "selection_split": "val", "summary": summarize_by_key(rows, ("variant", "recall_target")), "pooled_outer_test": pooled}


def candidate_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for top_k in (1, 3, 5, 8):
            configs.append({"variant": variant, "top_k": top_k, "fusion_a": 1.0, "fusion_b": 0.0, "use_fusion_logit": False, "method": "threshold_only"})
        for top_k in (3, 5, 8):
            for a in (0.5, 1.0, 1.5, 2.0):
                for b in (0.0, 0.25, 0.5, 1.0):
                    configs.append({"variant": variant, "top_k": top_k, "fusion_a": a, "fusion_b": b, "use_fusion_logit": True, "method": "stage1_stage2_logit_fusion"})
    return configs


def evaluate_config(run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dev_decisions = []
    test_decisions = []
    fold_rows = []
    thresholds = []
    for fold in range(5):
        dev_rows = load_split(run_root, fold, config["variant"], "val", top_k=int(config["top_k"]), fusion_a=float(config["fusion_a"]), fusion_b=float(config["fusion_b"]), use_fusion_logit=bool(config["use_fusion_logit"]))
        test_rows = load_split(run_root, fold, config["variant"], "test", top_k=int(config["top_k"]), fusion_a=float(config["fusion_a"]), fusion_b=float(config["fusion_b"]), use_fusion_logit=bool(config["use_fusion_logit"]))
        selection = threshold_for_safe_f1(dev_rows, preferred_fpr=0.01, allowed_fpr=0.02)
        threshold = float(selection["threshold"])
        thresholds.append(threshold)
        dev_fold = apply_threshold(dev_rows, threshold)
        test_fold = apply_threshold(test_rows, threshold)
        dev_decisions.extend(dev_fold)
        test_decisions.extend(test_fold)
        dm = metrics_from_rows(dev_fold)
        tm = metrics_from_rows(test_fold)
        fold_rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "selection_reason": selection["reason"],
                "safety_band": selection["safety_band"],
                "dev_recall": dm.get("recall"),
                "dev_false_positive_rate": dm.get("false_positive_rate"),
                "dev_precision": dm.get("precision"),
                "dev_f1": dm.get("f1"),
                "test_recall": tm.get("recall"),
                "test_false_positive_rate": tm.get("false_positive_rate"),
                "test_precision": tm.get("precision"),
                "test_f1": tm.get("f1"),
            }
        )
    dev = metrics_from_rows(dev_decisions)
    test = metrics_from_rows(test_decisions)
    return {
        **config,
        "thresholds": thresholds,
        "dev": dev,
        "test": test,
        "fold_rows": fold_rows,
        "safe_preferred": dev.get("false_positive_rate") is not None and float(dev["false_positive_rate"]) <= 0.01,
        "safe_allowed": dev.get("false_positive_rate") is not None and float(dev["false_positive_rate"]) <= 0.02,
    }


def optimization_search(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluated = [evaluate_config(run_root, config) for config in candidate_configs()]
    public_rows = []
    for item in evaluated:
        public_rows.append(
            {
                "variant": item["variant"],
                "method": item["method"],
                "top_k": item["top_k"],
                "fusion_a": item["fusion_a"],
                "fusion_b": item["fusion_b"],
                "use_fusion_logit": item["use_fusion_logit"],
                "safe_preferred": item["safe_preferred"],
                "safe_allowed": item["safe_allowed"],
                "dev_recall": item["dev"].get("recall"),
                "dev_false_positive_rate": item["dev"].get("false_positive_rate"),
                "dev_precision": item["dev"].get("precision"),
                "dev_f1": item["dev"].get("f1"),
                "test_recall": item["test"].get("recall"),
                "test_false_positive_rate": item["test"].get("false_positive_rate"),
                "test_precision": item["test"].get("precision"),
                "test_f1": item["test"].get("f1"),
            }
        )
    selectable = [item for item in evaluated if item["safe_preferred"]] or [item for item in evaluated if item["safe_allowed"]]
    selected = sorted(
        selectable,
        key=lambda item: (
            item["dev"].get("f1") if item["dev"].get("f1") is not None else -1.0,
            item["dev"].get("recall") if item["dev"].get("recall") is not None else -1.0,
            -(item["dev"].get("false_positive_rate") if item["dev"].get("false_positive_rate") is not None else 1.0),
            item["dev"].get("precision") if item["dev"].get("precision") is not None else -1.0,
        ),
        reverse=True,
    )[0]
    safe_mode = sorted(evaluated, key=lambda item: ((item["dev"].get("false_positive_rate") if item["dev"].get("false_positive_rate") is not None else 1.0), -(item["dev"].get("recall") or 0.0)))[0]
    high_recall = sorted(
        [item for item in evaluated if item["safe_allowed"]],
        key=lambda item: (item["dev"].get("recall") or 0.0, item["dev"].get("f1") or 0.0, -(item["dev"].get("false_positive_rate") or 1.0)),
        reverse=True,
    )[0]
    return public_rows, {"status": "ok", "selected_config": selected, "safe_mode": safe_mode, "balanced_mode": selected, "high_recall_mode": high_recall}


def baseline_summary(run_root: Path) -> dict[str, Any]:
    stage1 = []
    qwen_summary = read_json(Path("finetune/experiments/latest_data/reports/latest_nested_zero_shot_summary.json"))["methods"]["qwen_exact"]
    qwen = {metric: value.get("mean") for metric, value in qwen_summary.items() if isinstance(value, dict)}
    for fold in range(5):
        root = fold_dir(run_root, fold)
        stage1.extend(stage1_only_rows(read_jsonl(root / "stage1/test_predictions.jsonl"), theta1_for_fold(run_root, fold)))
    return {"qwen_exact": qwen, "stage1_only_recomputed": metrics_from_rows(stage1)}


def make_plot(rows: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [row["dev_false_positive_rate"] or 0.0 for row in rows]
        ys = [row["dev_recall"] or 0.0 for row in rows]
        plt.figure(figsize=(6, 4))
        plt.scatter(xs, ys, s=18, alpha=0.65)
        plt.xlabel("Development FPR")
        plt.ylabel("Development recall")
        plt.title("Stage2 threshold/config trade-off")
        plt.grid(True, alpha=0.25)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
    except Exception:
        out.write_bytes(b"")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="finetune/experiments/latest_data/runs/nested_zero_shot")
    parser.add_argument("--reports", default="finetune/experiments/latest_data_optimization/reports")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    op_rows, op_summary = operating_points(run_root)
    search_rows, search_summary = optimization_search(run_root)
    baseline = baseline_summary(run_root)
    write_csv(reports / "latest_opt_stage2_operating_points.csv", op_rows)
    write_json(reports / "latest_opt_stage2_operating_points.json", op_summary)
    write_csv(reports / "latest_opt_stage2_recall_search.csv", search_rows)
    selected_public = {
        key: value
        for key, value in search_summary["selected_config"].items()
        if key not in {"fold_rows"}
    }
    selected_public["baseline"] = baseline
    write_json(reports / "latest_opt_stage2_selected_config.json", selected_public)
    make_plot(search_rows, reports / "plots/stage2_threshold_tradeoff.png")

    selected = search_summary["selected_config"]
    lines = [
        "# Latest Optimized Stage 2 Operating Points",
        "",
        "Thresholds are selected from fold validation predictions only. Outer-test rows are used once for reporting.",
        "",
        "| Variant | Target recall | Dev recall | Dev FPR | Test recall | Test FPR | Test F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(op_summary["summary"].items()):
        variant, target = key.split("|")
        lines.append(
            f"| {variant} | {float(target):.2f} | {item['dev_recall']:.6f} | {item['dev_false_positive_rate']:.6f} | "
            f"{item['test_recall']:.6f} | {item['test_false_positive_rate']:.6f} | {item['test_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Meeting Modes",
            "",
            f"- Safe mode: `{search_summary['safe_mode']['variant']}`, top_k `{search_summary['safe_mode']['top_k']}`, method `{search_summary['safe_mode']['method']}`.",
            f"- Balanced mode: `{selected['variant']}`, top_k `{selected['top_k']}`, method `{selected['method']}`, dev F1 `{selected['dev'].get('f1')}`, test F1 `{selected['test'].get('f1')}`.",
            f"- High-recall mode: `{search_summary['high_recall_mode']['variant']}`, top_k `{search_summary['high_recall_mode']['top_k']}`, method `{search_summary['high_recall_mode']['method']}`.",
        ]
    )
    (reports / "LATEST_OPT_STAGE2_OPERATING_POINT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    s = selected
    recall_lines = [
        "# Latest Stage 2 Recall/FPR Optimization",
        "",
        "Development-only search evaluated threshold-only, top-K expansion, BCE versus SupCon, and Stage1/Stage2 logit fusion.",
        "",
        f"- Selected variant: `{s['variant']}`",
        f"- Selected method: `{s['method']}`",
        f"- Selected top_k: `{s['top_k']}`",
        f"- Fusion a/b/logit: `{s['fusion_a']}` / `{s['fusion_b']}` / `{s['use_fusion_logit']}`",
        f"- Development recall/FPR/F1: `{s['dev'].get('recall')}` / `{s['dev'].get('false_positive_rate')}` / `{s['dev'].get('f1')}`",
        f"- Outer-test recall/FPR/F1: `{s['test'].get('recall')}` / `{s['test'].get('false_positive_rate')}` / `{s['test'].get('f1')}`",
        f"- Qwen exact outer-test F1: `{baseline['qwen_exact'].get('f1')}`",
        f"- Stage1-only recomputed outer-test F1: `{baseline['stage1_only_recomputed'].get('f1')}`",
        "",
        "The selected configuration is deployment-safe if its development FPR is <= 0.02 and outer-test FPR is reported without tuning.",
    ]
    (reports / "LATEST_OPT_STAGE2_RECALL_REPORT.md").write_text("\n".join(recall_lines) + "\n", encoding="utf-8")
    print({"status": "ok", "selected": {k: s[k] for k in ("variant", "method", "top_k", "fusion_a", "fusion_b")}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
