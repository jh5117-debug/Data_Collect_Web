from pathlib import Path
import json
from typing import Protocol
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
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


class StorageBackend(Protocol):
    def absolute_path(self, relative_path: str) -> Path: ...
    def relative(self, path: Path | str) -> str: ...
    def raw_audio_path(
        self,
        participant_id: str,
        session_id: str,
        clip_id: str,
        extension: str,
        clip_type: str,
    ) -> Path | str: ...
    def processed_wav_path(self, participant_id: str, session_id: str, clip_id: str) -> Path | str: ...
    def segment_path(self, participant_id: str, session_id: str, parent_clip_id: str, segment_index: int) -> Path | str: ...
    def export_path(self, file_name: str) -> Path: ...
    async def save_upload(self, upload: UploadFile, destination: Path | str) -> int: ...
    def exists(self, relative_path: str) -> bool: ...
    def download_bytes(self, relative_path: str) -> bytes: ...
    def delete(self, relative_path: str) -> bool: ...


class LocalStorageBackend:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.local_storage_root
        self.root.mkdir(parents=True, exist_ok=True)

    def absolute_path(self, relative_path: str) -> Path:
        return self.root / relative_path

    def relative(self, path: Path | str) -> str:
        if isinstance(path, str):
            return path
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

    async def save_upload(self, upload: UploadFile, destination: Path | str) -> int:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with destination.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                handle.write(chunk)
        return total

    def exists(self, relative_path: str) -> bool:
        return self.absolute_path(relative_path).exists()

    def download_bytes(self, relative_path: str) -> bytes:
        return self.absolute_path(relative_path).read_bytes()

    def delete(self, relative_path: str) -> bool:
        path = self.absolute_path(relative_path)
        try:
            path.relative_to(self.root)
        except ValueError:
            return False
        if not path.exists() or not path.is_file():
            return False
        path.unlink()
        remove_empty_parents(path.parent, self.root)
        return True


class S3StorageBackend:
    """Placeholder for future S3 or MinIO support."""

    def __init__(self) -> None:
        raise NotImplementedError("S3 storage is not implemented in this MVP.")


class SupabaseStorageBackend:
    def __init__(self) -> None:
        if not settings.supabase_url:
            raise RuntimeError("SUPABASE_URL is required when STORAGE_BACKEND=supabase")
        if not settings.supabase_secret_key:
            raise RuntimeError("SUPABASE_SECRET_KEY is required when STORAGE_BACKEND=supabase")
        self.supabase_url = settings.supabase_url.rstrip("/")
        self.bucket = settings.supabase_storage_bucket
        self.secret_key = settings.supabase_secret_key
        self.root = settings.local_storage_root
        self.root.mkdir(parents=True, exist_ok=True)

    def absolute_path(self, relative_path: str) -> Path:
        return self.root / relative_path

    def relative(self, path: Path | str) -> str:
        if isinstance(path, Path):
            return path.as_posix()
        return path

    def raw_audio_path(
        self,
        participant_id: str,
        session_id: str,
        clip_id: str,
        extension: str,
        clip_type: str,
    ) -> str:
        base = "calibration" if clip_type == "calibration" else "raw_audio"
        return f"{base}/{participant_id}/{session_id}/{clip_id}{extension}"

    def processed_wav_path(self, participant_id: str, session_id: str, clip_id: str) -> str:
        return f"processed_wav/{participant_id}/{session_id}/{clip_id}.wav"

    def segment_path(self, participant_id: str, session_id: str, parent_clip_id: str, segment_index: int) -> str:
        filename = f"{parent_clip_id}_seg{segment_index:03d}.wav"
        return f"segments/{participant_id}/{session_id}/{filename}"

    def export_path(self, file_name: str) -> Path:
        return self.root / "exports" / file_name

    async def save_upload(self, upload: UploadFile, destination: Path | str) -> int:
        relative_path = self.relative(destination)
        data = await upload.read()
        self._request(
            "POST",
            f"/storage/v1/object/{self.bucket}/{quote(relative_path, safe='/')}",
            data=data,
            headers={
                "content-type": upload.content_type or "application/octet-stream",
                "x-upsert": "true",
            },
        )
        return len(data)

    def exists(self, relative_path: str) -> bool:
        try:
            self._request(
                "HEAD",
                f"/storage/v1/object/{self.bucket}/{quote(relative_path, safe='/')}",
            )
        except HTTPError as exc:
            if exc.code == 404:
                return False
            raise
        except FileNotFoundError:
            return False
        return True

    def download_bytes(self, relative_path: str) -> bytes:
        try:
            return self._request(
                "GET",
                f"/storage/v1/object/{self.bucket}/{quote(relative_path, safe='/')}",
            )
        except HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(relative_path) from exc
            raise

    def delete(self, relative_path: str) -> bool:
        if not self.exists(relative_path):
            return False
        payload = json.dumps({"prefixes": [relative_path]}).encode("utf-8")
        try:
            self._request(
                "DELETE",
                f"/storage/v1/object/{self.bucket}",
                data=payload,
                headers={"content-type": "application/json"},
            )
        except HTTPError as exc:
            if exc.code == 404:
                return False
            raise
        return True

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        request = Request(
            f"{self.supabase_url}{path}",
            data=data,
            method=method,
            headers={
                "apikey": self.secret_key,
                "authorization": f"Bearer {self.secret_key}",
                **(headers or {}),
            },
        )
        with urlopen(request, timeout=60) as response:
            return response.read()


def remove_empty_parents(path: Path, root: Path) -> None:
    while path != root and path.exists():
        try:
            path.rmdir()
        except OSError:
            return
        path = path.parent


def get_storage_backend() -> StorageBackend:
    if settings.storage_backend == "supabase":
        return SupabaseStorageBackend()
    if settings.storage_backend != "local":
        raise NotImplementedError("STORAGE_BACKEND must be local or supabase.")
    return LocalStorageBackend()


def infer_extension(content_type: str | None, filename: str | None) -> str:
    if content_type in CONTENT_TYPE_EXTENSIONS:
        return CONTENT_TYPE_EXTENSIONS[content_type]

    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix

    return f".{uuid4().hex}.bin"
