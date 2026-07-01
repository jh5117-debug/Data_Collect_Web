from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SUPPORT_SEEDS = [20260620, 20260621, 20260622, 20260623, 20260624]
METHOD_ORDER = {
    "zero_shot": 0,
    "stage2_positive_bias": 1,
    "stage2_cosine_prototype": 2,
    "stage2_finetune_bias_only": 3,
    "stage2_finetune_head": 4,
    "stage1_finetune_bias_only": 5,
    "stage1_finetune_head": 6,
    "stage1_stage2_combined": 7,
}


@dataclass(frozen=True)
class FewShotRecord:
    target: str
    doctor_alias: str
    shot: int
    seed: int
    support: list[dict[str, Any]]
    query: list[dict[str, Any]]
    source_replay_stage2: list[dict[str, Any]]
    source_replay_stage1: list[dict[str, Any]]


def stable_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(value: float, eps: float = 1e-6) -> float:
    value = min(1.0 - eps, max(eps, float(value)))
    return math.log(value / (1.0 - value))


def l2_normalize(vector: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("cannot normalize empty or non-finite vector")
    return arr / norm


def build_prototype(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        raise ValueError("support rows are required")
    return l2_normalize(np.stack([l2_normalize(row["embedding"]) for row in rows]).mean(axis=0))


def cosine_similarity(row: dict[str, Any], prototype: np.ndarray) -> float:
    return float(np.dot(l2_normalize(row["embedding"]), l2_normalize(prototype)))


def binary_metrics(labels: list[int], decisions: list[bool]) -> dict[str, Any]:
    if len(labels) != len(decisions):
        raise ValueError("labels and decisions must have equal length")
    tp = tn = fp = fn = 0
    for label, decision in zip(labels, decisions):
        y = int(label)
        pred = bool(decision)
        if pred and y == 1:
            tp += 1
        elif pred and y == 0:
            fp += 1
        elif not pred and y == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "n": len(labels),
        "positive": tp + fn,
        "negative": tn + fp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "f1": f1,
        "specificity": specificity,
    }


def metrics_from_rows(rows: list[dict[str, Any]], decision_key: str) -> dict[str, Any]:
    return binary_metrics([int(row["label"]) for row in rows], [bool(row[decision_key]) for row in rows])


def clip_key(row: dict[str, Any]) -> str:
    return str(row["clip_id"])


def one_row_per_clip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_clip[clip_key(row)].append(row)
    return [sorted(group, key=lambda item: int(item.get("window_index", 0)))[0] for _, group in sorted(by_clip.items())]


def eligibility(rows: list[dict[str, Any]], shots: int) -> dict[str, Any]:
    clips = one_row_per_clip(rows)
    positives = sum(int(row["label"]) == 1 for row in clips)
    negatives = sum(int(row["label"]) == 0 for row in clips)
    return {
        "eligible": positives >= shots + 1 and negatives >= 1,
        "stable_fpr": negatives >= 5,
        "positive_clips": positives,
        "negative_clips": negatives,
        "required_positive_clips": shots + 1,
    }


def _prompt_cover_order(positives: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        by_prompt[str(row.get("prompt_group", ""))].append(row)
    for group in by_prompt.values():
        group.sort(key=clip_key)
        rng.shuffle(group)
    ordered = []
    for prompt in ("P1_vigil_only", "P2_phrase_plus_vigil", "P3_vigil_plus_phrase"):
        if by_prompt.get(prompt):
            ordered.append(by_prompt[prompt].pop(0))
    rest = [row for group in by_prompt.values() for row in group]
    rest.sort(key=clip_key)
    rng.shuffle(rest)
    ordered.extend(rest)
    seen = set()
    unique = []
    for row in ordered:
        key = clip_key(row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def choose_supports_for_seed(rows: list[dict[str, Any]], seed: int) -> dict[int, list[dict[str, Any]]]:
    clips = one_row_per_clip(rows)
    positives = [row for row in clips if int(row["label"]) == 1]
    negatives = [row for row in clips if int(row["label"]) == 0]
    if len(positives) < 4 or not negatives:
        raise ValueError("target doctor is not eligible for 3-shot onboarding")
    ordered = _prompt_cover_order(positives, random.Random(seed))
    supports: dict[int, list[dict[str, Any]]] = {3: ordered[:3]}
    if len(positives) >= 6:
        supports[5] = ordered[:5]
    return supports


def split_support_query(rows: list[dict[str, Any]], support: list[dict[str, Any]], target: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_rows = [row for row in one_row_per_clip(rows) if str(row.get("participant_alias")) == str(target)]
    support_ids = {clip_key(row) for row in support}
    if len(support_ids) != len(support):
        raise ValueError("support contains duplicate clips")
    if any(str(row.get("participant_alias")) != str(target) for row in support):
        raise ValueError("support must contain only the target doctor")
    if any(int(row["label"]) != 1 for row in support):
        raise ValueError("target negatives are not allowed in support")
    target_ids = {clip_key(row) for row in target_rows}
    if not support_ids <= target_ids:
        raise ValueError("support contains clips outside target doctor")
    query = [row for row in target_rows if clip_key(row) not in support_ids]
    if support_ids & {clip_key(row) for row in query}:
        raise ValueError("support/query overlap")
    if any(str(row.get("participant_alias")) != str(target) for row in query):
        raise ValueError("query contains non-target doctor clips")
    return list(support), query


def assert_paired_query(zero_rows: list[dict[str, Any]], adapted_rows: list[dict[str, Any]]) -> None:
    if {clip_key(row) for row in zero_rows} != {clip_key(row) for row in adapted_rows}:
        raise ValueError("zero-shot and few-shot query sets are not identical")


def load_base_by_alias(run_root: Path | str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(run_root).glob("*/base_clip_rows.jsonl")):
        rows = []
        for row in read_jsonl(path):
            item = dict(row)
            item["embedding"] = np.asarray(item.pop("embedding_list"), dtype=np.float32)
            item["base_logit"] = logit(float(item.get("base_score", item.get("stage2_score", 0.0))))
            item["stage1_logit"] = logit(float(item.get("stage1_clip_score", 0.0)))
            item["theta1_logit"] = logit(float(item.get("theta1", 0.5)))
            item["theta2_logit"] = logit(float(item.get("theta2", 0.5)))
            rows.append(item)
        if rows:
            out[str(rows[0]["participant_alias"])] = rows
    return out


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"embedding", "participant_alias", "speaker_id", "session_id", "clip_id"}
    }


def source_replay_for_target(
    base_by_alias: dict[str, list[dict[str, Any]]],
    target: str,
    seed: int,
    *,
    positives: int = 24,
    negatives: int = 48,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [row for alias, group in base_by_alias.items() if alias != target for row in one_row_per_clip(group)]
    if any(str(row.get("participant_alias")) == str(target) for row in rows):
        raise ValueError("source replay contains target doctor rows")
    rng = random.Random(seed)
    pos = [row for row in rows if int(row["label"]) == 1]
    neg = [row for row in rows if int(row["label"]) == 0]
    hard_neg = [row for row in neg if str(row.get("prompt_group", "")).startswith("P4")]
    pos = sorted(pos, key=clip_key)
    neg_pool = sorted(hard_neg or neg, key=clip_key)
    rng.shuffle(pos)
    rng.shuffle(neg_pool)
    return pos[:positives], neg_pool[:negatives]


def build_records(base_by_alias: dict[str, list[dict[str, Any]]], *, max_targets: int = 0) -> list[FewShotRecord]:
    records: list[FewShotRecord] = []
    aliases = sorted(base_by_alias)
    if max_targets:
        aliases = aliases[:max_targets]
    for alias in aliases:
        clips = one_row_per_clip(base_by_alias[alias])
        if not eligibility(clips, 3)["eligible"]:
            continue
        for seed in SUPPORT_SEEDS:
            supports = choose_supports_for_seed(clips, seed)
            replay_pos, replay_neg = source_replay_for_target(base_by_alias, alias, seed)
            for shot, support in sorted(supports.items()):
                if shot == 5 and not eligibility(clips, 5)["eligible"]:
                    continue
                _, query = split_support_query(clips, support, alias)
                support_ids = {clip_key(row) for row in support}
                query_ids = {clip_key(row) for row in query}
                support_rows = [row for row in base_by_alias[alias] if clip_key(row) in support_ids]
                query_rows = [row for row in base_by_alias[alias] if clip_key(row) in query_ids]
                records.append(
                    FewShotRecord(
                        target=alias,
                        doctor_alias=str(clips[0]["doctor_alias"]),
                        shot=shot,
                        seed=seed,
                        support=support_rows,
                        query=query_rows,
                        source_replay_stage2=[*replay_pos, *replay_neg],
                        source_replay_stage1=replay_neg,
                    )
                )
    return records


def fit_bias(logits: list[float], labels: list[int], *, lr: float, steps: int, l2: float = 1.0) -> float:
    b = 0.0
    y = np.asarray(labels, dtype=np.float32)
    x = np.asarray(logits, dtype=np.float32)
    for _ in range(int(steps)):
        p = np.asarray([sigmoid(v) for v in x + b], dtype=np.float32)
        grad = float(np.mean(p - y)) + float(l2) * b
        b -= float(lr) * grad
    return float(b)


def fit_linear_adapter(
    embeddings: np.ndarray,
    base_logits: np.ndarray,
    labels: list[int],
    *,
    lr: float,
    steps: int,
    l2: float = 10.0,
) -> tuple[np.ndarray, float]:
    x = np.asarray(embeddings, dtype=np.float32)
    logits = np.asarray(base_logits, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    w = np.zeros(x.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(int(steps)):
        delta = x @ w + b
        p = np.asarray([sigmoid(v) for v in logits + delta], dtype=np.float32)
        err = p - y
        grad_w = (x.T @ err) / max(1, len(y)) + float(l2) * w
        grad_b = float(np.mean(err)) + float(l2) * b
        w -= float(lr) * grad_w.astype(np.float32)
        b -= float(lr) * grad_b
    return w, float(b)


def method_grid(method: str) -> list[dict[str, Any]]:
    if method == "zero_shot":
        return [{"method": method, "changed_stage": "none", "support_based": False}]
    if method == "stage2_cosine_prototype":
        return [
            {"method": method, "alpha": alpha, "beta": beta, "changed_stage": "stage2", "support_based": True}
            for alpha in (0.25, 0.5, 1.0, 2.0, 4.0)
            for beta in (0.0, 0.25, 0.5)
        ]
    if method == "stage2_positive_bias":
        return [
            {"method": method, "max_bias": max_bias, "margin": margin, "quantile": quantile, "changed_stage": "stage2", "support_based": True}
            for max_bias in (0.25, 0.5, 1.0, 2.0)
            for margin in (0.0, 0.25, 0.5)
            for quantile in ("min", "q20", "median")
        ]
    if method == "stage2_finetune_bias_only":
        return [
            {"method": method, "steps": steps, "lr": lr, "changed_stage": "stage2", "support_based": True}
            for steps in (5, 10, 25, 50)
            for lr in (1e-4, 3e-4, 1e-3)
        ]
    if method == "stage2_finetune_head":
        return [
            {"method": method, "steps": steps, "lr": lr, "changed_stage": "stage2", "support_based": True}
            for steps in (5, 10, 25)
            for lr in (1e-4, 3e-4, 1e-3)
        ]
    if method == "stage1_finetune_bias_only":
        return [
            {"method": method, "steps": steps, "lr": lr, "changed_stage": "stage1", "support_based": True}
            for steps in (5, 10, 25, 50)
            for lr in (1e-4, 3e-4, 1e-3)
        ]
    if method == "stage1_finetune_head":
        return [
            {"method": method, "steps": steps, "lr": lr, "changed_stage": "stage1", "support_based": True}
            for steps in (5, 10, 25)
            for lr in (1e-4, 3e-4, 1e-3)
        ]
    raise ValueError(f"unsupported method grid: {method}")


def _support_quantile(values: list[float], quantile: str) -> float:
    if quantile == "min":
        return min(values)
    if quantile == "q20":
        return float(np.quantile(values, 0.2))
    if quantile == "median":
        return float(np.median(values))
    raise ValueError(f"unsupported quantile: {quantile}")


def apply_recipe(record: FewShotRecord, recipe: dict[str, Any]) -> list[dict[str, Any]]:
    method = str(recipe["method"])
    support = record.support
    replay2 = record.source_replay_stage2
    replay1 = record.source_replay_stage1
    prototype = build_prototype(support) if method == "stage2_cosine_prototype" else None
    stage2_bias = 0.0
    stage1_bias = 0.0
    stage2_w: np.ndarray | None = None
    stage2_b = 0.0
    stage1_w = 0.0
    stage1_b = 0.0
    if method == "stage2_positive_bias":
        support_logits = [float(row["base_logit"]) for row in support]
        theta = float(support[0]["theta2_logit"])
        q = _support_quantile(support_logits, str(recipe["quantile"]))
        stage2_bias = min(float(recipe["max_bias"]), max(0.0, theta - q + float(recipe["margin"])))
    elif method == "stage2_finetune_bias_only":
        train = [*support, *replay2]
        stage2_bias = fit_bias([float(row["base_logit"]) for row in train], [int(row["label"]) for row in train], lr=float(recipe["lr"]), steps=int(recipe["steps"]))
    elif method == "stage2_finetune_head":
        train = [*support, *replay2]
        stage2_w, stage2_b = fit_linear_adapter(
            np.stack([row["embedding"] for row in train]),
            np.asarray([float(row["base_logit"]) for row in train], dtype=np.float32),
            [int(row["label"]) for row in train],
            lr=float(recipe["lr"]),
            steps=int(recipe["steps"]),
        )
    elif method == "stage1_finetune_bias_only":
        train = [*support, *replay1]
        stage1_bias = fit_bias([float(row["stage1_logit"]) for row in train], [int(row["label"]) for row in train], lr=float(recipe["lr"]), steps=int(recipe["steps"]))
    elif method == "stage1_finetune_head":
        train = [*support, *replay1]
        x = np.asarray([float(row["stage1_clip_score"]) - float(row["theta1"]) for row in train], dtype=np.float32)
        y = np.asarray([int(row["label"]) for row in train], dtype=np.float32)
        base = np.asarray([float(row["stage1_logit"]) for row in train], dtype=np.float32)
        for _ in range(int(recipe["steps"])):
            p = np.asarray([sigmoid(v) for v in base + stage1_w * x + stage1_b], dtype=np.float32)
            err = p - y
            stage1_w -= float(recipe["lr"]) * (float(np.mean(err * x)) + 10.0 * stage1_w)
            stage1_b -= float(recipe["lr"]) * (float(np.mean(err)) + 10.0 * stage1_b)
    out = []
    for row in record.query:
        candidate = bool(row["stage1_candidate"])
        stage2_pass = bool(float(row["base_logit"]) >= float(row["theta2_logit"]))
        adapted_score = float(row["base_score"])
        if method == "zero_shot":
            decision = bool(row["base_decision"])
        elif method == "stage2_cosine_prototype":
            sim = cosine_similarity(row, prototype)
            adapted_logit = float(row["base_logit"]) + float(recipe["alpha"]) * (sim - float(recipe["beta"]))
            adapted_score = sigmoid(adapted_logit)
            decision = candidate and adapted_logit >= float(row["theta2_logit"])
        elif method in {"stage2_positive_bias", "stage2_finetune_bias_only"}:
            adapted_logit = float(row["base_logit"]) + stage2_bias
            adapted_score = sigmoid(adapted_logit)
            decision = candidate and adapted_logit >= float(row["theta2_logit"])
        elif method == "stage2_finetune_head":
            assert stage2_w is not None
            adapted_logit = float(row["base_logit"]) + float(np.dot(row["embedding"], stage2_w) + stage2_b)
            adapted_score = sigmoid(adapted_logit)
            decision = candidate and adapted_logit >= float(row["theta2_logit"])
        elif method == "stage1_finetune_bias_only":
            adapted_stage1 = float(row["stage1_logit"]) + stage1_bias >= float(row["theta1_logit"])
            decision = adapted_stage1 and stage2_pass
        elif method == "stage1_finetune_head":
            adapted_stage1 = float(row["stage1_logit"]) + stage1_w * (float(row["stage1_clip_score"]) - float(row["theta1"])) + stage1_b >= float(row["theta1_logit"])
            decision = adapted_stage1 and stage2_pass
        else:
            raise ValueError(f"unsupported method: {method}")
        out.append(
            {
                **sanitize_row(row),
                "clip_hash": stable_hash(clip_key(row)),
                "doctor_alias": record.doctor_alias,
                "shot": record.shot,
                "support_seed": record.seed,
                "method": method,
                "changed_stage": recipe.get("changed_stage", "none"),
                "support_based": bool(recipe.get("support_based")),
                "zero_decision": bool(row["base_decision"]),
                "adapted_decision": bool(decision),
                "adapted_score": float(adapted_score),
                "stage1_miss": bool(int(row["label"]) == 1 and not bool(row["stage1_candidate"])),
                "stage2_reject": bool(int(row["label"]) == 1 and bool(row["stage1_candidate"]) and not stage2_pass),
                "final_false_accept": bool(int(row["label"]) == 0 and decision),
            }
        )
    return out


def evaluate_records(records: list[FewShotRecord], recipe: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        adapted = apply_recipe(record, recipe)
        zero_rows = [{**row, "clip_id": row["clip_hash"]} for row in adapted]
        assert_paired_query(zero_rows, [{**row, "clip_id": row["clip_hash"]} for row in adapted])
        rows.extend(adapted)
    return rows


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    zero = metrics_from_rows(rows, "zero_decision")
    adapted = metrics_from_rows(rows, "adapted_decision")
    return {
        "zero": zero,
        "adapted": adapted,
        "delta_f1": (adapted.get("f1") or 0.0) - (zero.get("f1") or 0.0),
        "delta_recall": (adapted.get("recall") or 0.0) - (zero.get("recall") or 0.0),
        "delta_fpr": (adapted.get("false_positive_rate") or 0.0) - (zero.get("false_positive_rate") or 0.0),
        "stage1_misses": sum(bool(row.get("stage1_miss")) for row in rows),
        "stage2_rejects": sum(bool(row.get("stage2_reject")) for row in rows),
        "final_false_accepts": sum(bool(row.get("final_false_accept")) for row in rows),
    }


def safety_pass(summary: dict[str, Any]) -> bool:
    zero_fpr = summary.get("zero", {}).get("false_positive_rate") or 0.0
    adapted_fpr = summary.get("adapted", {}).get("false_positive_rate") or 0.0
    return adapted_fpr - zero_fpr <= 0.02 + 1e-12 and adapted_fpr <= 0.03 + 1e-12


def select_recipe(records: list[FewShotRecord], method: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    search = []
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for recipe in method_grid(method):
        rows = evaluate_records(records, recipe)
        summary = summarize_prediction_rows(rows)
        safe = safety_pass(summary)
        changed = any(bool(row["zero_decision"]) != bool(row["adapted_decision"]) for row in rows)
        item = {
            **recipe,
            "safe": safe,
            "changed_outputs": changed,
            "query_rows": len(rows),
            "zero_f1": summary.get("zero", {}).get("f1"),
            "adapted_f1": summary.get("adapted", {}).get("f1"),
            "adapted_recall": summary.get("adapted", {}).get("recall"),
            "adapted_fpr": summary.get("adapted", {}).get("false_positive_rate"),
            "delta_f1": summary.get("delta_f1"),
            "delta_recall": summary.get("delta_recall"),
            "delta_fpr": summary.get("delta_fpr"),
        }
        search.append(item)
        key = (
            bool(safe),
            float(summary.get("delta_f1") or 0.0),
            float(summary.get("delta_recall") or 0.0),
            -METHOD_ORDER.get(method, 99),
            bool(changed),
        )
        if best is None or key > best[0]:
            best = (key, recipe)
    assert best is not None
    selected = dict(best[1])
    selected["selection_safe"] = bool(best[0][0])
    if not selected["selection_safe"]:
        selected = {"method": "zero_shot", "changed_stage": "none", "support_based": False, "selection_safe": False, "fallback_reason": f"{method}_unsafe_on_dev"}
    return selected, search


def aggregate_per_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["doctor_alias"]), int(row["shot"]), int(row["support_seed"]), str(row["method"]))].append(row)
    out = []
    for (doctor, shot, seed, method), group in sorted(grouped.items()):
        summary = summarize_prediction_rows(group)
        out.append(flat_summary(doctor, shot, method, summary, support_seed=seed, rows=group))
    return out


def aggregate_per_doctor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["doctor_alias"]), int(row["shot"]), str(row["method"]))].append(row)
    out = []
    for (doctor, shot, method), group in sorted(grouped.items()):
        summary = summarize_prediction_rows(group)
        out.append(flat_summary(doctor, shot, method, summary, rows=group))
    return out


def aggregate_per_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["shot"]), str(row["method"]))].append(row)
    out = []
    for (shot, method), group in sorted(grouped.items(), key=lambda item: (item[0][0], METHOD_ORDER.get(item[0][1], 99))):
        summary = summarize_prediction_rows(group)
        doctor_rows = aggregate_per_doctor(group)
        deltas = [float(row["delta_f1"]) for row in doctor_rows if row.get("delta_f1") is not None]
        improved = sum(delta > 1e-12 for delta in deltas)
        degraded = sum(delta < -1e-12 for delta in deltas)
        unchanged = sum(abs(delta) <= 1e-12 for delta in deltas)
        item = flat_summary("ALL", shot, method, summary, rows=group)
        item.update(
            {
                "doctor_macro_f1": float(np.mean([row["f1"] for row in doctor_rows if row.get("f1") is not None])) if doctor_rows else None,
                "mean_paired_delta_f1": float(np.mean(deltas)) if deltas else None,
                "median_paired_delta_f1": float(np.median(deltas)) if deltas else None,
                "improved_doctors": improved,
                "degraded_doctors": degraded,
                "unchanged_doctors": unchanged,
                "safety_pass_rate": float(np.mean([row["safety_pass"] for row in doctor_rows])) if doctor_rows else None,
            }
        )
        out.append(item)
    return out


def flat_summary(
    doctor: str,
    shot: int,
    method: str,
    summary: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    support_seed: int | None = None,
) -> dict[str, Any]:
    zero = summary.get("zero", {})
    adapted = summary.get("adapted", {})
    label = "unchanged"
    delta = float(summary.get("delta_f1") or 0.0)
    if delta > 1e-12:
        label = "improved"
    elif delta < -1e-12:
        label = "degraded"
    return {
        "doctor_alias": doctor,
        "shot": shot,
        "support_seed": support_seed,
        "method": method,
        "positive_query_count": adapted.get("positive"),
        "negative_query_count": adapted.get("negative"),
        "zero_precision": zero.get("precision"),
        "zero_recall": zero.get("recall"),
        "zero_fpr": zero.get("false_positive_rate"),
        "zero_f1": zero.get("f1"),
        "precision": adapted.get("precision"),
        "recall": adapted.get("recall"),
        "fpr": adapted.get("false_positive_rate"),
        "f1": adapted.get("f1"),
        "delta_f1": summary.get("delta_f1"),
        "delta_recall": summary.get("delta_recall"),
        "delta_fpr": summary.get("delta_fpr"),
        "change_label": label,
        "safety_pass": safety_pass(summary),
        "fpr_unstable": (adapted.get("negative") or 0) < 5,
        "stage1_misses": summary.get("stage1_misses"),
        "stage2_rejects": summary.get("stage2_rejects"),
        "final_false_accepts": summary.get("final_false_accepts"),
        "changed_stage": next((row.get("changed_stage") for row in rows if row.get("method") == method), None),
    }


def method_changed_stage(method: str) -> str:
    if method.startswith("stage1_stage2"):
        return "stage1+stage2"
    if method.startswith("stage1"):
        return "stage1"
    if method.startswith("stage2"):
        return "stage2"
    return "none"
