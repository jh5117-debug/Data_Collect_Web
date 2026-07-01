from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import pytest

from vigil_two_stage.qwen_text_result import QwenTextExtractionError, extract_qwen_text


def test_plain_string() -> None:
    extracted = extract_qwen_text("  THE QUICK BROWN FOX  ")
    assert extracted.text == "THE QUICK BROWN FOX"
    assert extracted.extraction_path == "$"
    assert extracted.result_type == "builtins.str"


def test_bytes() -> None:
    assert extract_qwen_text(b"hello").text == "hello"


def test_dict_with_text() -> None:
    extracted = extract_qwen_text({"text": "hello"})
    assert extracted.text == "hello"
    assert extracted.extraction_path == "$['text']"


def test_nested_dict() -> None:
    extracted = extract_qwen_text({"output": {"text": "nested transcript"}})
    assert extracted.text == "nested transcript"
    assert extracted.extraction_path == "$['output']['text']"


@dataclass
class DataclassResult:
    language: str
    text: str


def test_list_of_dataclass_result_objects() -> None:
    extracted = extract_qwen_text([DataclassResult("English", "dataclass text")])
    assert extracted.text == "dataclass text"
    assert extracted.extraction_path == "$[0].text"
    assert extracted.result_type.endswith(".DataclassResult")


def test_tuple_of_result_objects() -> None:
    extracted = extract_qwen_text((DataclassResult("English", "tuple text"),))
    assert extracted.text == "tuple text"
    assert extracted.extraction_path == "$[0].text"


class StandardResult:
    def __init__(self, text: str):
        self.language = "English"
        self.text = text


def test_standard_class_with_text() -> None:
    assert extract_qwen_text(StandardResult("standard text")).text == "standard text"


class TranscriptResult:
    transcript = "transcript attr"


def test_object_with_transcript() -> None:
    assert extract_qwen_text(TranscriptResult()).text == "transcript attr"


class HypothesisResult:
    hypothesis = "hypothesis attr"


def test_object_with_hypothesis() -> None:
    assert extract_qwen_text(HypothesisResult()).text == "hypothesis attr"


def test_named_tuple() -> None:
    Result = namedtuple("Result", ["language", "text"])
    extracted = extract_qwen_text(Result("English", "named tuple text"))
    assert extracted.text == "named tuple text"
    assert extracted.extraction_path == "$.text"


class PydanticLike:
    def model_dump(self) -> dict[str, str]:
        return {"text": "model dump text"}


def test_pydantic_like_model_dump() -> None:
    assert extract_qwen_text(PydanticLike()).text == "model dump text"


def test_empty_list_raises() -> None:
    with pytest.raises(QwenTextExtractionError, match="empty"):
        extract_qwen_text([])


def test_unsupported_object_raises() -> None:
    with pytest.raises(QwenTextExtractionError, match="unsupported Qwen result object"):
        extract_qwen_text(object())


def test_python_object_repr_is_not_accepted() -> None:
    with pytest.raises(QwenTextExtractionError, match="object repr"):
        extract_qwen_text({"text": "<Fake object at 0xabc123>"})


def test_structured_result_repr_is_not_accepted() -> None:
    with pytest.raises(QwenTextExtractionError, match="structured result repr"):
        extract_qwen_text("ASRTranscription(language='English', text='THE REAL TEXT', time_stamps=None)")


def test_field_label_repr_is_not_accepted() -> None:
    with pytest.raises(QwenTextExtractionError, match="field labels"):
        extract_qwen_text("language='English', text='THE REAL TEXT'")


def test_cycle_detection() -> None:
    value: dict[str, object] = {}
    value["text"] = value
    with pytest.raises(QwenTextExtractionError, match="cycle"):
        extract_qwen_text(value)


def test_nested_valid_result() -> None:
    assert extract_qwen_text([{"output": [StandardResult("deep text")]}]).text == "deep text"


def test_extraction_path_is_recorded() -> None:
    assert extract_qwen_text([{"output": {"text": "path text"}}]).extraction_path == "$[0]['output']['text']"


class FakeASRResult:
    def __init__(self):
        self.language = "English"
        self.text = "THE QUICK BROWN FOX"

    def __repr__(self) -> str:
        return "FakeASRResult(language='English', text='THE QUICK BROWN FOX')"


def test_qwen_style_result_shape_uses_text_field_not_repr() -> None:
    extracted = extract_qwen_text([FakeASRResult()])
    assert extracted.text == "THE QUICK BROWN FOX"
    assert extracted.text != repr(FakeASRResult())
    assert extracted.extraction_path == "$[0].text"
    assert extracted.result_type.endswith(".FakeASRResult")

