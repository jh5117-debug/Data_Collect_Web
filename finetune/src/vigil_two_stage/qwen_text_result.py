from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


TEXT_FIELD_NAMES = ("text", "transcript", "hypothesis", "prediction", "output")

_OBJECT_AT_RE = re.compile(r"<[^>]+ object at 0x[0-9a-fA-F]+>")
_RESULT_REPR_PREFIX_RE = re.compile(r"^[A-Za-z_][\w.]*Result\s*\(")
_ASR_REPR_PREFIX_RE = re.compile(r"^[\w.]*ASR[A-Za-z_]*\s*\(")
_FIELD_LABEL_REPR_RE = re.compile(r"\blanguage\s*=.+\btext\s*=", re.DOTALL)


class QwenTextExtractionError(ValueError):
    """Raised when a Qwen ASR result does not expose a supported transcript field."""


@dataclass(frozen=True)
class ExtractedQwenText:
    text: str
    extraction_path: str
    result_type: str


def _type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _safe_repr(value: object, limit: int = 500) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _validate_text(text: str, *, path: str, result_type: str) -> str:
    text = text.strip()
    if not text:
        raise QwenTextExtractionError(f"empty Qwen transcript at {path} from {result_type}")
    if _OBJECT_AT_RE.search(text):
        raise QwenTextExtractionError(f"Qwen transcript looks like Python object repr at {path}: {_safe_repr(text)}")
    if _RESULT_REPR_PREFIX_RE.search(text) or _ASR_REPR_PREFIX_RE.search(text):
        raise QwenTextExtractionError(f"Qwen transcript looks like structured result repr at {path}: {_safe_repr(text)}")
    if _FIELD_LABEL_REPR_RE.search(text):
        raise QwenTextExtractionError(f"Qwen transcript contains result field labels at {path}: {_safe_repr(text)}")
    return text


def _candidate_paths(value: object) -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        return [(f"[{name!r}]", value[name]) for name in TEXT_FIELD_NAMES if name in value]
    out: list[tuple[str, object]] = []
    for name in TEXT_FIELD_NAMES:
        if hasattr(value, name):
            try:
                out.append((f".{name}", getattr(value, name)))
            except Exception:
                continue
    if not out and hasattr(value, "_asdict"):
        try:
            mapping = value._asdict()
        except Exception:
            mapping = None
        if isinstance(mapping, Mapping):
            out.extend((f"._asdict()[{name!r}]", mapping[name]) for name in TEXT_FIELD_NAMES if name in mapping)
    if not out and hasattr(value, "model_dump"):
        try:
            mapping = value.model_dump()
        except Exception:
            mapping = None
        if isinstance(mapping, Mapping):
            out.extend((f".model_dump()[{name!r}]", mapping[name]) for name in TEXT_FIELD_NAMES if name in mapping)
    if not out and dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            if field.name in TEXT_FIELD_NAMES:
                out.append((f".{field.name}", getattr(value, field.name)))
    return out


def extract_qwen_text(result: object, *, max_depth: int = 8) -> ExtractedQwenText:
    seen: set[int] = set()

    def visit(value: object, path: str, depth: int, source_type: str | None = None) -> ExtractedQwenText:
        if depth > max_depth:
            raise QwenTextExtractionError(f"Qwen text extraction exceeded max depth {max_depth} at {path}")
        result_type = _type_name(value)
        if isinstance(value, str):
            return ExtractedQwenText(
                text=_validate_text(value, path=path, result_type=result_type),
                extraction_path=path,
                result_type=source_type or result_type,
            )
        if isinstance(value, bytes):
            decoded = value.decode("utf-8", errors="replace")
            return ExtractedQwenText(
                text=_validate_text(decoded, path=path, result_type=result_type),
                extraction_path=path,
                result_type=source_type or result_type,
            )
        obj_id = id(value)
        if obj_id in seen:
            raise QwenTextExtractionError(f"cycle detected while extracting Qwen text at {path} from {result_type}")
        seen.add(obj_id)
        try:
            if isinstance(value, Mapping):
                last_error: QwenTextExtractionError | None = None
                for suffix, child in _candidate_paths(value):
                    try:
                        return visit(child, path + suffix, depth + 1, source_type=result_type)
                    except QwenTextExtractionError as exc:
                        last_error = exc
                        continue
                if last_error is not None:
                    raise last_error
                supported = ", ".join(TEXT_FIELD_NAMES)
                keys = ", ".join(str(key) for key in list(value.keys())[:20])
                raise QwenTextExtractionError(
                    f"unsupported Qwen mapping result at {path}: type={result_type}; "
                    f"supported fields={supported}; keys={keys}; repr={_safe_repr(value)}"
                )
            if isinstance(value, tuple) and hasattr(value, "_asdict"):
                last_error: QwenTextExtractionError | None = None
                for suffix, child in _candidate_paths(value):
                    try:
                        return visit(child, path + suffix, depth + 1, source_type=result_type)
                    except QwenTextExtractionError as exc:
                        last_error = exc
                        continue
                if last_error is not None:
                    raise last_error
            if isinstance(value, (list, tuple)):
                if not value:
                    raise QwenTextExtractionError(f"empty Qwen result sequence at {path}: type={result_type}")
                last_error: QwenTextExtractionError | None = None
                for index, child in enumerate(value):
                    try:
                        return visit(child, f"{path}[{index}]", depth + 1)
                    except QwenTextExtractionError as exc:
                        last_error = exc
                        continue
                if last_error is not None:
                    raise last_error
                raise QwenTextExtractionError(
                    f"no supported transcript in Qwen result sequence at {path}: type={result_type}; repr={_safe_repr(value)}"
                )
            last_error: QwenTextExtractionError | None = None
            for suffix, child in _candidate_paths(value):
                try:
                    return visit(child, path + suffix, depth + 1, source_type=result_type)
                except QwenTextExtractionError as exc:
                    last_error = exc
                    continue
            if last_error is not None:
                raise last_error
            supported = ", ".join(TEXT_FIELD_NAMES)
            attrs = ", ".join(name for name in dir(value) if not name.startswith("_"))[:300]
            raise QwenTextExtractionError(
                f"unsupported Qwen result object at {path}: type={result_type}; "
                f"supported fields={supported}; attrs={attrs}; repr={_safe_repr(value)}"
            )
        finally:
            seen.discard(obj_id)

    return visit(result, "$", 0)
