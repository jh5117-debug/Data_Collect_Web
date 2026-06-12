from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from ..config import settings


CONTENT_TYPE_EXTENSIONS = {
    "audio/webm": ".webm",
    "audio/webm;codecs=opus": ".webm",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


class LocalStorageBackend:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.local_storage_root
        self.root.mkdir(parents=True, exist_ok=True)

    def absolute_path(self, relative_path: str) -> Path:
        return self.root / relative_path

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def raw_audio_path(
        self,
        participant_id: str,
        session_id: str,
        clip_id: str,
        extension: str,
        clip_type: str,
    ) -> Path:
        base = "calibration" if clip_type == "calibration" else "raw_audio"
        return self.root / base / participant_id / session_id / f"{clip_id}{extension}"

    def processed_wav_path(self, participant_id: str, session_id: str, clip_id: str) -> Path:
        return self.root / "processed_wav" / participant_id / session_id / f"{clip_id}.wav"

    def segment_path(
        self, participant_id: str, session_id: str, parent_clip_id: str, segment_index: int
    ) -> Path:
        filename = f"{parent_clip_id}_seg{segment_index:03d}.wav"
        return self.root / "segments" / participant_id / session_id / filename

    def export_path(self, file_name: str) -> Path:
        return self.root / "exports" / file_name

    async def save_upload(self, upload: UploadFile, destination: Path) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with destination.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                handle.write(chunk)
        return total


class S3StorageBackend:
    """Placeholder for future S3 or MinIO support."""

    def __init__(self) -> None:
        raise NotImplementedError("S3 storage is not implemented in this MVP.")


def get_storage_backend() -> LocalStorageBackend:
    if settings.storage_backend != "local":
        raise NotImplementedError("Only local storage is implemented in this MVP.")
    return LocalStorageBackend()


def infer_extension(content_type: str | None, filename: str | None) -> str:
    if content_type in CONTENT_TYPE_EXTENSIONS:
        return CONTENT_TYPE_EXTENSIONS[content_type]

    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix

    return f".{uuid4().hex}.bin"
