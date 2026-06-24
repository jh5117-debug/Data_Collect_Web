from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_librispeech_runner_uses_shared_extractor() -> None:
    source = _read("finetune/benchmarks/asr/src/qwen_runner.py")
    assert "from vigil_two_stage.qwen_text_result import extract_qwen_text" in source
    assert "extract_qwen_text(" in source
    assert "return str(result)" not in source
    assert "def _extract_text" not in source


def test_vigil_baseline_uses_shared_extractor() -> None:
    source = _read("finetune/scripts/run_qwen_text_baseline.py")
    assert "from vigil_two_stage.qwen_text_result import extract_qwen_text" in source
    assert "extract_qwen_text(" in source
    assert "return str(result)" not in source
    assert "def _extract_text" not in source


def test_demo_uses_shared_extractor() -> None:
    source = _read("finetune/demo/inference.py")
    assert "from vigil_two_stage.qwen_text_result import extract_qwen_text" in source
    assert "extract_qwen_text(" in source
    assert "return str(result)" not in source
    assert "def _extract_text" not in source
