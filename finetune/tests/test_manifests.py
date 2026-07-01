import json

from vigil_two_stage.manifests import qwen_asr_text
from vigil_two_stage.utils import stable_json


def test_qwen_manifest_uses_transcript_only():
    assert qwen_asr_text("visual") == "language English<asr_text>visual"
    assert "positive" not in qwen_asr_text("VIGIL")
    assert "negative" not in qwen_asr_text("visual")


def test_manifest_serialization_is_deterministic():
    a = stable_json({"b": 1, "a": 2})
    b = stable_json({"a": 2, "b": 1})
    assert a == b
    assert json.loads(a) == {"a": 2, "b": 1}
