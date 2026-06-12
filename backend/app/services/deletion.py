from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Clip, Participant, RecordingSession, Segment
from .storage import get_storage_backend


def delete_clip_and_files(db: Session, clip: Clip) -> list[str]:
    storage = get_storage_backend()
    deleted_files: list[str] = []

    segments = db.execute(select(Segment).where(Segment.parent_clip_id == clip.clip_id)).scalars().all()
    relative_paths = [clip.raw_audio_path, clip.processed_wav_path] + [
        segment.segment_audio_path for segment in segments
    ]
    for relative_path in relative_paths:
        if not relative_path:
            continue
        path = storage.absolute_path(relative_path)
        try:
            path.relative_to(storage.root)
        except ValueError:
            continue
        if path.exists() and path.is_file():
            path.unlink()
            deleted_files.append(relative_path)
            remove_empty_parents(path.parent, storage.root)

    for segment in segments:
        db.delete(segment)
    db.delete(clip)
    return deleted_files


def remove_empty_parents(path: Path, root: Path) -> None:
    while path != root and path.exists():
        try:
            path.rmdir()
        except OSError:
            return
        path = path.parent


def delete_session_clips_and_files(db: Session, session: RecordingSession) -> list[str]:
    clips = db.execute(select(Clip).where(Clip.session_id == session.session_id)).scalars().all()
    deleted_files: list[str] = []
    for clip in clips:
        deleted_files.extend(delete_clip_and_files(db, clip))
    return deleted_files


def delete_generated_exports() -> list[str]:
    storage = get_storage_backend()
    exports_dir = storage.root / "exports"
    deleted_files: list[str] = []
    if not exports_dir.exists():
        return deleted_files

    for path in exports_dir.glob("*.zip"):
        if not path.is_file():
            continue
        try:
            path.relative_to(storage.root)
        except ValueError:
            continue
        path.unlink()
        deleted_files.append(str(path.relative_to(storage.root)))
    remove_empty_parents(exports_dir, storage.root)
    return deleted_files


def delete_session_and_files(db: Session, session: RecordingSession, *, delete_empty_participant: bool = True) -> list[str]:
    participant_id = session.participant_id
    session_id = session.session_id
    deleted_files = delete_session_clips_and_files(db, session)
    db.delete(session)

    if delete_empty_participant:
        remaining_sessions = db.execute(
            select(RecordingSession)
            .where(
                RecordingSession.participant_id == participant_id,
                RecordingSession.session_id != session_id,
            )
            .limit(1)
        ).scalars().first()
        if not remaining_sessions:
            participant = db.get(Participant, participant_id)
            if participant:
                db.delete(participant)

    return deleted_files
