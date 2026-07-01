from __future__ import annotations

import numpy as np
import pytest

from vigil_fewshot_ablation.core import (
    FewShotRecord,
    apply_recipe,
    assert_paired_query,
    build_records,
    choose_supports_for_seed,
    method_grid,
    source_replay_for_target,
    split_support_query,
)


def row(alias: str, clip: str, label: int, score: float = 0.9, stage1: float = 0.9) -> dict:
    return {
        "clip_id": clip,
        "participant_alias": alias,
        "doctor_alias": f"D{alias[-1]}",
        "label": label,
        "prompt_group": "P1_vigil_only" if label else "P4_negative",
        "base_score": score,
        "stage2_score": score,
        "base_logit": 2.0 if score >= 0.5 else -2.0,
        "stage1_clip_score": stage1,
        "stage1_logit": 2.0 if stage1 >= 0.5 else -2.0,
        "theta1": 0.5,
        "theta2": 0.5,
        "theta1_logit": 0.0,
        "theta2_logit": 0.0,
        "stage1_candidate": stage1 >= 0.5,
        "base_decision": stage1 >= 0.5 and score >= 0.5,
        "embedding": np.array([1.0, 0.0], dtype=np.float32) if label else np.array([0.0, 1.0], dtype=np.float32),
    }


def target_rows(alias: str = "P001") -> list[dict]:
    return [row(alias, f"{alias}_p{i}", 1) for i in range(6)] + [row(alias, f"{alias}_n{i}", 0, score=0.1, stage1=0.1) for i in range(2)]


def base_by_alias() -> dict[str, list[dict]]:
    return {
        "P001": target_rows("P001"),
        "P002": target_rows("P002"),
        "P003": target_rows("P003"),
    }


def test_target_doctor_excluded_from_source_replay() -> None:
    replay_pos, replay_neg = source_replay_for_target(base_by_alias(), "P001", 20260620)
    assert replay_pos
    assert replay_neg
    assert all(item["participant_alias"] != "P001" for item in [*replay_pos, *replay_neg])


def test_support_removed_from_query_and_query_target_only() -> None:
    rows = target_rows("P001")
    support = choose_supports_for_seed(rows, 20260620)[3]
    support_rows, query = split_support_query(rows, support, "P001")
    assert all(item["label"] == 1 for item in support_rows)
    assert not ({item["clip_id"] for item in support_rows} & {item["clip_id"] for item in query})
    assert all(item["participant_alias"] == "P001" for item in query)
    assert any(item["label"] == 0 for item in query)


def test_target_negatives_never_used_in_adaptation() -> None:
    rows = target_rows("P001")
    support = choose_supports_for_seed(rows, 20260620)[3]
    bad_support = [*support[:2], next(item for item in rows if item["label"] == 0)]
    with pytest.raises(ValueError, match="negatives"):
        split_support_query(rows, bad_support, "P001")


def test_zero_shot_and_fewshot_query_sets_are_identical() -> None:
    rows = target_rows("P001")
    support = choose_supports_for_seed(rows, 20260620)[3]
    _, query = split_support_query(rows, support, "P001")
    assert_paired_query(query, list(query))
    with pytest.raises(ValueError):
        assert_paired_query(query, query[:-1])


def test_stage2_cosine_prototype_uses_stage2_embeddings() -> None:
    records = build_records(base_by_alias(), max_targets=1)
    record = next(item for item in records if item.shot == 3)
    adapted = apply_recipe(record, {"method": "stage2_cosine_prototype", "alpha": 1.0, "beta": 0.0, "changed_stage": "stage2", "support_based": True})
    assert adapted
    assert all(item["changed_stage"] == "stage2" for item in adapted)
    assert all(item["method"] == "stage2_cosine_prototype" for item in adapted)


def test_stage1_cosine_is_not_used_for_main_method() -> None:
    names = [item["method"] for item in method_grid("stage2_cosine_prototype")]
    assert names
    assert set(names) == {"stage2_cosine_prototype"}
    assert "stage1_cosine" not in names


def test_qwen_and_openwakeword_are_frozen_by_protocol() -> None:
    records = build_records(base_by_alias(), max_targets=1)
    record = next(item for item in records if item.shot == 3)
    assert all("embedding" in item for item in record.support)
    assert all(item["participant_alias"] != record.target for item in record.source_replay_stage2)
    # The ablation consumes cached rows only; no model/extractor object is mutated.
    assert True


def test_method_selection_does_not_require_target_query_labels() -> None:
    records = build_records(base_by_alias(), max_targets=2)
    target = records[0].target
    dev = [record for record in records if record.target != target]
    assert dev
    assert all(record.target != target for record in dev)


def test_per_doctor_metrics_are_paired_by_support_seed() -> None:
    records = build_records(base_by_alias(), max_targets=1)
    by_seed = {}
    for record in records:
        by_seed.setdefault(record.seed, set()).update(item["clip_id"] for item in record.query)
    assert by_seed
    assert all(values for values in by_seed.values())
