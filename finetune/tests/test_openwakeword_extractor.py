from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_extractor_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "extract_openwakeword_features.py"
    spec = importlib.util.spec_from_file_location("extract_openwakeword_features", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openwakeword_normalizes_batch_embeddings():
    module = load_extractor_module()
    output = np.zeros((1, 16, 96), dtype=np.float32)
    features = module.OfficialOpenWakeWordExtractor._normalize_output(output)
    assert features.shape == (16, 96)
    assert features.dtype == np.float32


def test_openwakeword_normalizes_single_vector():
    module = load_extractor_module()
    output = np.zeros((96,), dtype=np.float32)
    features = module.OfficialOpenWakeWordExtractor._normalize_output(output)
    assert features.shape == (1, 96)


def test_openwakeword_rejects_ragged_list_output():
    module = load_extractor_module()
    with pytest.raises(RuntimeError, match="ragged/object"):
        module.OfficialOpenWakeWordExtractor._normalize_output([np.zeros((2, 96)), np.zeros((3, 96))])


def test_openwakeword_rejects_empty_output():
    module = load_extractor_module()
    with pytest.raises(RuntimeError, match="empty"):
        module.OfficialOpenWakeWordExtractor._normalize_output(np.asarray([]))

