from __future__ import annotations

from typing import Any


FORBIDDEN_BUNDLE_KEYS = {"name", "email", "speaker_hash", "account_id", "session_id", "participant_alias_map"}


def validate_public_bundle_manifest(manifest: dict[str, Any]) -> None:
    found = sorted(key for key in manifest if key in FORBIDDEN_BUNDLE_KEYS)
    if found:
        raise ValueError(f"bundle manifest contains private identity fields: {found}")
    if manifest.get("include_qwen_weights"):
        raise ValueError("final bundle must not include Qwen weights")
