from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trigger import bounded_positive_bias


def _safe_id(value: str) -> str:
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in value):
        raise ValueError("invalid id")
    return value


def _display_name(name: str) -> str:
    words = name.strip().split()
    return words[0][:40] if words else "Local user"


def _name_hash(name: str) -> str:
    return hashlib.sha256(name.strip().encode("utf-8")).hexdigest()[:16]


@dataclass
class ProfileStore:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def profile_dir(self, profile_id: str) -> Path:
        return self.root / "profiles" / _safe_id(profile_id)

    def create_profile(self, name: str) -> dict[str, Any]:
        profile_id = uuid.uuid4().hex
        profile = {
            "profile_id": profile_id,
            "display_name": _display_name(name),
            "name_sha256_16": _name_hash(name),
        }
        directory = self.profile_dir(profile_id)
        (directory / "clips").mkdir(parents=True, exist_ok=True)
        (directory / "sessions").mkdir(parents=True, exist_ok=True)
        (directory / "profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        return {"profile_id": profile_id, "display_name": profile["display_name"]}

    def require_profile(self, profile_id: str) -> Path:
        directory = self.profile_dir(profile_id)
        if not (directory / "profile.json").exists():
            raise FileNotFoundError(profile_id)
        return directory

    def save_clip(self, profile_id: str, filename: str, data: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
        directory = self.require_profile(profile_id)
        clip_id = uuid.uuid4().hex
        suffix = Path(filename or "clip.webm").suffix.lower()
        if suffix not in {".webm", ".wav", ".m4a", ".ogg", ".mp3"}:
            suffix = ".webm"
        audio_path = directory / "clips" / f"{clip_id}{suffix}"
        meta_path = directory / "clips" / f"{clip_id}.json"
        audio_path.write_bytes(data)
        meta = {
            "clip_id": clip_id,
            "profile_id": profile_id,
            "audio_filename": audio_path.name,
            "prompt_group": metadata.get("prompt_group") or "P1_vigil_only",
            "transcript": metadata.get("transcript") or "",
            "is_positive": bool(metadata.get("is_positive", True)),
            "accepted": bool(metadata.get("accepted", True)),
            "size_bytes": len(data),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return {
            "clip_id": clip_id,
            "duration": None,
            "playback_url": f"/api/onboarding/clip/{clip_id}/audio?profile_id={profile_id}",
            **meta,
        }

    def clip_paths(self, profile_id: str, clip_id: str) -> tuple[Path, Path]:
        directory = self.require_profile(profile_id)
        clip_id = _safe_id(clip_id)
        meta_path = directory / "clips" / f"{clip_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(clip_id)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if Path(meta["audio_filename"]).name != meta["audio_filename"]:
            raise ValueError("unsafe audio path")
        audio_path = directory / "clips" / meta["audio_filename"]
        if audio_path.parent.resolve() != (directory / "clips").resolve():
            raise ValueError("unsafe audio path")
        return meta_path, audio_path

    def delete_clip(self, profile_id: str, clip_id: str) -> dict[str, Any]:
        meta_path, audio_path = self.clip_paths(profile_id, clip_id)
        if audio_path.exists():
            audio_path.unlink()
        meta_path.unlink()
        return {"deleted": True, "clip_id": clip_id}

    def accepted_positive_clips(self, profile_id: str) -> list[dict[str, Any]]:
        directory = self.require_profile(profile_id)
        clips = []
        for meta_path in sorted((directory / "clips").glob("*.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("accepted") and meta.get("is_positive"):
                clips.append(meta)
        return clips

    def calibrate(self, profile_id: str, support_scores: list[float], threshold: float) -> dict[str, Any]:
        clips = self.accepted_positive_clips(profile_id)
        if len(clips) < 3:
            return {
                "calibration_status": "need_more_positive_clips",
                "support_count": len(clips),
                "method": None,
                "calibration_active": False,
                "bias": 0.0,
                "warnings": ["At least 3 accepted positive VIGIL clips are required."],
            }
        usable_scores = support_scores[: len(clips)] if support_scores else [0.75] * len(clips)
        bias = bounded_positive_bias(usable_scores, threshold)
        calibration = {
            "calibration_status": "ok",
            "support_count": len(clips),
            "method": "bounded_positive_bias_demo",
            "calibration_active": True,
            "bias": bias,
            "threshold": threshold,
            "warnings": [],
        }
        directory = self.require_profile(profile_id)
        (directory / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        return calibration

    def calibration(self, profile_id: str) -> dict[str, Any]:
        directory = self.require_profile(profile_id)
        path = directory / "calibration.json"
        if not path.exists():
            return {"calibration_active": False, "bias": 0.0, "method": None}
        return json.loads(path.read_text(encoding="utf-8"))


def clear_local_data(root: Path) -> None:
    root = root.resolve()
    if root.name != "local_data" or "demo_live_assistant" not in str(root):
        raise RuntimeError(f"refusing to clear unsafe path: {root}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
