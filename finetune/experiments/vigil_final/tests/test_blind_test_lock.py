from __future__ import annotations

import pytest

from vigil_final.blind_test import BlindTestLock, reject_known_participants, validate_lock
from vigil_final.final_bundle import validate_public_bundle_manifest


def lock():
    return BlindTestLock(
        code_commit="abc123",
        selected_method="stage2_bce",
        stage1_threshold=0.1,
        stage2_threshold=0.2,
        balanced_dataset_checksum="sha",
        fold_checksum="fold",
        onboarding_recipe={"method": "none"},
        inference_stride=0.25,
        top_k=3,
        locked_date="2026-06-25",
    ).to_json()


def test_blind_test_lock_checksum_validation_fields():
    validate_lock(lock())
    bad = lock()
    bad.pop("fold_checksum")
    with pytest.raises(ValueError):
        validate_lock(bad)


def test_blind_test_evaluator_rejects_known_participants():
    reject_known_participants({"P999"}, {"P001"})
    with pytest.raises(ValueError):
        reject_known_participants({"P001"}, {"P001"})


def test_final_model_bundle_contains_no_identity_or_qwen_weights():
    validate_public_bundle_manifest({"include_qwen_weights": False, "selected_method": "stage2"})
    with pytest.raises(ValueError):
        validate_public_bundle_manifest({"include_qwen_weights": False, "email": "x@y.com"})
    with pytest.raises(ValueError):
        validate_public_bundle_manifest({"include_qwen_weights": True})
