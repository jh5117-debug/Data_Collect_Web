from __future__ import annotations

from typing import Any


def feature_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["clip_id"]), int(row.get("window_index", 0)), str(row.get("window_audio_sha256"))
