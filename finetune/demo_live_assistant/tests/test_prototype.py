from __future__ import annotations

import numpy as np
import pytest

from model_runtime import AssistantModelRuntime
from prototype import PrototypeCalibrationError, build_prototype, cosine_similarity, l2_normalize


def test_l2_normalize_and_cosine_similarity() -> None:
    vec = l2_normalize([3.0, 4.0])
    assert np.allclose(vec, [0.6, 0.8])
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_build_prototype_records_pairwise_similarity() -> None:
    calibration = build_prototype([
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.8, 0.2], dtype=np.float32),
        np.array([0.9, 0.1], dtype=np.float32),
    ])

    assert calibration.support_count == 3
    assert calibration.embedding.shape == (2,)
    assert np.linalg.norm(calibration.embedding) == pytest.approx(1.0)
    assert calibration.pairwise_mean_similarity is not None
    assert calibration.pairwise_mean_similarity > 0.9


def test_build_prototype_rejects_bad_embeddings() -> None:
    with pytest.raises(PrototypeCalibrationError):
        build_prototype([])
    with pytest.raises(PrototypeCalibrationError):
        build_prototype([[0.0, 0.0]])
    with pytest.raises(PrototypeCalibrationError):
        build_prototype([[1.0, 0.0], [1.0, 0.0, 0.0]])


def test_runtime_builds_real_few_shot_calibration_from_support_embeddings(tmp_path) -> None:
    runtime = AssistantModelRuntime(force_mock=True)
    runtime.mode = "real"
    runtime.inference = object()
    paths = [tmp_path / f"clip_{index}.webm" for index in range(3)]
    for path in paths:
        path.write_bytes(b"fake")

    def fake_embedding(path):
        index = int(path.stem.split("_")[-1])
        return {
            "score": 0.8 + index * 0.01,
            "embedding": np.array([1.0, index * 0.1], dtype=np.float32),
        }

    runtime._stage2_embedding_and_score = fake_embedding
    calibration = runtime.build_support_calibration(
        [
            {
                "clip_id": path.stem,
                "prompt_group": "P1_vigil_only",
                "transcript": "VIGIL",
                "audio_path": str(path),
            }
            for path in paths
        ]
    )

    assert calibration["calibration_status"] == "ok"
    assert calibration["method"] == "few_shot_qwen_stage2_prototype"
    assert calibration["calibration_active"] is True
    assert calibration["prototype_embedding_dim"] == 2
    assert calibration["stage2_weights_updated"] is False
