from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Clip, EmailLoginCode, Participant, RecordingSession, Segment, UserAccount, UserSessionToken
from ..schemas import AccountSessionOut, AdminClientOut, AdminClipOut, DeleteClipOut, ExportOut, FlaggedClipOut, SummaryOut
from ..services.deletion import delete_clip_and_files, delete_generated_exports, delete_session_and_files
from ..services.export import create_export_zip
from ..services.email_auth import normalize_account_identifier
from ..services.prompt_groups import contains_exact_vigil
from ..services.qc import flags_from_json, flags_to_json
from ..services.storage import get_storage_backend

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _is_positive_clip(clip: Clip) -> bool:
    return bool(clip.contains_vigil and clip.wake_intent and not clip.is_negative)


def _clip_effective_flags(clip: Clip) -> list[str]:
    flags = set(flags_from_json(clip.auto_qc_flags))
    if not clip.normalized_transcript:
        flags.add("missing_transcript")
    if clip.prompt_group in {"P2_phrase_plus_vigil", "P3_vigil_plus_phrase"} and not contains_exact_vigil(
        clip.normalized_transcript
    ):
        flags.add("transcript_missing_vigil")
    if clip.prompt_group == "P4_negative" and contains_exact_vigil(clip.normalized_transcript):
        flags.add("negative_transcript_contains_vigil")

    storage = get_storage_backend()
    relative_path = clip.processed_wav_path or clip.raw_audio_path
    if not relative_path or not storage.exists(relative_path):
        flags.add("missing_audio_file")
    return sorted(flags)


def _clip_effective_status(clip: Clip) -> str:
    flags = _clip_effective_flags(clip)
    if clip.auto_qc_status == "auto_rejected":
        return "auto_rejected"
    if flags or clip.auto_qc_status == "flagged_for_review":
        return "flagged_for_review"
    return "auto_accepted"


def _clip_flag_string(clip: Clip) -> str:
    return flags_to_json(_clip_effective_flags(clip))


@router.get("/summary", response_model=SummaryOut)
def admin_summary(db: Session = Depends(get_db)) -> SummaryOut:
    accounts = db.execute(select(func.count()).select_from(UserAccount)).scalar_one()
    sessions = db.execute(
        select(func.count())
        .select_from(RecordingSession)
        .join(Participant, RecordingSession.participant_id == Participant.participant_id)
        .join(UserAccount, Participant.user_email == UserAccount.email)
    ).scalar_one()
    submitted_sessions = db.execute(
        select(func.count())
        .select_from(RecordingSession)
        .join(Participant, RecordingSession.participant_id == Participant.participant_id)
        .join(UserAccount, Participant.user_email == UserAccount.email)
        .where(RecordingSession.status == "submitted")
    ).scalar_one()
    total_clips = db.execute(
        select(func.count())
        .select_from(Clip)
        .join(Participant, Clip.participant_id == Participant.participant_id)
        .join(UserAccount, Participant.user_email == UserAccount.email)
    ).scalar_one()
    total_segments = db.execute(
        select(func.count())
        .select_from(Segment)
        .join(Participant, Segment.participant_id == Participant.participant_id)
        .join(UserAccount, Participant.user_email == UserAccount.email)
    ).scalar_one()
    auto_accepted = db.execute(
        select(func.count())
        .select_from(Clip)
        .join(Participant, Clip.participant_id == Participant.participant_id)
        .join(UserAccount, Participant.user_email == UserAccount.email)
        .where(Clip.auto_qc_status == "auto_accepted")
    ).scalar_one()
    flagged = db.execute(
        select(func.count())
        .select_from(Clip)
        .join(Participant, Clip.participant_id == Participant.participant_id)
        .join(UserAccount, Participant.user_email == UserAccount.email)
        .where(Clip.auto_qc_status == "flagged_for_review")
    ).scalar_one()
    rejected = db.execute(
        select(func.count())
        .select_from(Clip)
        .join(Participant, Clip.participant_id == Participant.participant_id)
        .join(UserAccount, Participant.user_email == UserAccount.email)
        .where(Clip.auto_qc_status == "auto_rejected")
    ).scalar_one()
    clips = (
        db.execute(
            select(Clip)
            .join(Participant, Clip.participant_id == Participant.participant_id)
            .join(UserAccount, Participant.user_email == UserAccount.email)
        )
        .scalars()
        .all()
    )
    prompt_group_counts: dict[str, int] = {
        "P1_vigil_only": 0,
        "P2_phrase_plus_vigil": 0,
        "P3_vigil_plus_phrase": 0,
        "P4_negative": 0,
        "legacy": 0,
    }
    for clip in clips:
        prompt_group_counts[clip.prompt_group or "legacy"] = prompt_group_counts.get(clip.prompt_group or "legacy", 0) + 1

    return SummaryOut(
        batch_id=settings.default_batch_id,
        participants=accounts,
        sessions=sessions,
        submitted_sessions=submitted_sessions,
        total_clips=total_clips,
        total_segments=total_segments,
        positive_clips=sum(1 for clip in clips if _is_positive_clip(clip)),
        negative_clips=sum(1 for clip in clips if clip.is_negative),
        auto_accepted=auto_accepted,
        flagged=flagged,
        rejected=rejected,
        prompt_group_counts=prompt_group_counts,
    )


@router.get("/flagged", response_model=list[FlaggedClipOut])
def flagged_clips(db: Session = Depends(get_db)) -> list[FlaggedClipOut]:
    clips = db.execute(select(Clip).order_by(Clip.created_at_utc.desc()).limit(500)).scalars().all()
    return [
        flagged_clip_out(clip)
        for clip in clips
        if _clip_effective_status(clip) in {"flagged_for_review", "auto_rejected"}
    ][:100]


@router.get("/clients", response_model=list[AdminClientOut])
def admin_clients(db: Session = Depends(get_db)) -> list[AdminClientOut]:
    accounts = db.execute(select(UserAccount).order_by(UserAccount.created_at_utc.desc())).scalars().all()

    rows: list[AdminClientOut] = []
    for account in accounts:
        email = account.email
        participant_ids = db.execute(
            select(Participant.participant_id).where(Participant.user_email == email)
        ).scalars().all()
        session_ids = []
        if participant_ids:
            session_ids = db.execute(
                select(RecordingSession.session_id).where(RecordingSession.participant_id.in_(participant_ids))
            ).scalars().all()
        clip_count = (
            db.execute(select(func.count()).select_from(Clip).where(Clip.session_id.in_(session_ids))).scalar_one()
            if session_ids
            else 0
        )
        segment_count = (
            db.execute(select(func.count()).select_from(Segment).where(Segment.session_id.in_(session_ids))).scalar_one()
            if session_ids
            else 0
        )
        positive_clip_count = (
            db.execute(
                select(func.count())
                .select_from(Clip)
                .where(
                    Clip.session_id.in_(session_ids),
                    Clip.contains_vigil.is_(True),
                    Clip.wake_intent.is_(True),
                    Clip.is_negative.is_(False),
                )
            ).scalar_one()
            if session_ids
            else 0
        )
        negative_clip_count = (
            db.execute(
                select(func.count())
                .select_from(Clip)
                .where(Clip.session_id.in_(session_ids), Clip.is_negative.is_(True))
            ).scalar_one()
            if session_ids
            else 0
        )
        submitted_session_count = (
            db.execute(
                select(func.count())
                .select_from(RecordingSession)
                .where(RecordingSession.session_id.in_(session_ids), RecordingSession.status == "submitted")
            ).scalar_one()
            if session_ids
            else 0
        )
        rows.append(
            AdminClientOut(
                email=email,
                verified=account.verified,
                account_created_at_utc=account.created_at_utc,
                last_login_at_utc=account.last_login_at_utc,
                participant_count=len(participant_ids),
                session_count=len(session_ids),
                submitted_session_count=submitted_session_count,
                clip_count=clip_count,
                positive_clip_count=positive_clip_count,
                negative_clip_count=negative_clip_count,
                segment_count=segment_count,
            )
        )
    return rows


def flagged_clip_out(clip: Clip) -> FlaggedClipOut:
    return FlaggedClipOut(
        clip_id=clip.clip_id,
        participant_id=clip.participant_id,
        session_id=clip.session_id,
        prompt_id=clip.prompt_id,
        prompt_group=clip.prompt_group,
        prompt_title=clip.prompt_title,
        transcript=clip.transcript,
        normalized_transcript=clip.normalized_transcript,
        contains_vigil=clip.contains_vigil,
        wake_intent=clip.wake_intent,
        is_negative=clip.is_negative,
        clip_type=clip.clip_type,
        duration_sec=clip.duration_sec,
        auto_qc_status=_clip_effective_status(clip),
        auto_qc_flags=_clip_flag_string(clip),
        segmentation_status=clip.segmentation_status,
        detected_segment_count=clip.detected_segment_count,
        expected_segment_count=clip.expected_segment_count,
        created_at_utc=clip.created_at_utc,
    )


def admin_clip_out(clip: Clip, user_email: str | None) -> AdminClipOut:
    return AdminClipOut(
        clip_id=clip.clip_id,
        participant_id=clip.participant_id,
        user_email=user_email,
        session_id=clip.session_id,
        prompt_id=clip.prompt_id,
        prompt_group=clip.prompt_group,
        prompt_title=clip.prompt_title,
        transcript=clip.transcript,
        normalized_transcript=clip.normalized_transcript,
        contains_vigil=clip.contains_vigil,
        wake_intent=clip.wake_intent,
        is_negative=clip.is_negative,
        clip_type=clip.clip_type,
        raw_audio_path=clip.raw_audio_path,
        processed_wav_path=clip.processed_wav_path,
        duration_sec=clip.duration_sec,
        file_size_bytes=clip.file_size_bytes,
        auto_qc_status=_clip_effective_status(clip),
        auto_qc_flags=_clip_flag_string(clip),
        segmentation_status=clip.segmentation_status,
        detected_segment_count=clip.detected_segment_count,
        expected_segment_count=clip.expected_segment_count,
        created_at_utc=clip.created_at_utc,
    )


@router.get("/clips", response_model=list[AdminClipOut])
def admin_clips(db: Session = Depends(get_db)) -> list[AdminClipOut]:
    rows = (
        db.execute(
            select(Clip, Participant.user_email)
            .join(Participant, Clip.participant_id == Participant.participant_id)
            .order_by(Clip.created_at_utc.desc())
            .limit(500)
        )
        .all()
    )
    return [admin_clip_out(clip, user_email) for clip, user_email in rows]


@router.get("/clients/{email}/sessions", response_model=list[AccountSessionOut])
def admin_client_sessions(email: str, db: Session = Depends(get_db)) -> list[AccountSessionOut]:
    normalized = normalize_account_identifier(email)
    sessions = (
        db.execute(
            select(RecordingSession)
            .join(Participant, RecordingSession.participant_id == Participant.participant_id)
            .where(Participant.user_email == normalized)
            .order_by(RecordingSession.created_at_utc.desc())
        )
        .scalars()
        .all()
    )
    output: list[AccountSessionOut] = []
    for session in sessions:
        clip_count = db.execute(
            select(func.count()).select_from(Clip).where(Clip.session_id == session.session_id)
        ).scalar_one()
        positive_clip_count = db.execute(
            select(func.count())
            .select_from(Clip)
            .where(
                Clip.session_id == session.session_id,
                Clip.contains_vigil.is_(True),
                Clip.wake_intent.is_(True),
                Clip.is_negative.is_(False),
            )
        ).scalar_one()
        negative_clip_count = db.execute(
            select(func.count()).select_from(Clip).where(Clip.session_id == session.session_id, Clip.is_negative.is_(True))
        ).scalar_one()
        output.append(
            AccountSessionOut(
                session_id=session.session_id,
                batch_id=session.batch_id,
                status=session.status,
                created_at_utc=session.created_at_utc,
                submitted_at_utc=session.submitted_at_utc,
                clip_count=clip_count,
                positive_clip_count=positive_clip_count,
                negative_clip_count=negative_clip_count,
            )
        )
    return output


@router.get("/clients/{email}/clips", response_model=list[AdminClipOut])
def admin_client_clips(email: str, db: Session = Depends(get_db)) -> list[AdminClipOut]:
    normalized = normalize_account_identifier(email)
    rows = (
        db.execute(
            select(Clip, Participant.user_email)
            .join(Participant, Clip.participant_id == Participant.participant_id)
            .where(Participant.user_email == normalized)
            .order_by(Clip.created_at_utc.desc())
        )
        .all()
    )
    return [admin_clip_out(clip, user_email) for clip, user_email in rows]


@router.get("/sessions/{session_id}/clips", response_model=list[AdminClipOut])
def admin_session_clips(session_id: str, db: Session = Depends(get_db)) -> list[AdminClipOut]:
    session = db.get(RecordingSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    rows = (
        db.execute(
            select(Clip, Participant.user_email)
            .join(Participant, Clip.participant_id == Participant.participant_id)
            .where(Clip.session_id == session_id)
            .order_by(Clip.created_at_utc)
        )
        .all()
    )
    return [admin_clip_out(clip, user_email) for clip, user_email in rows]


@router.delete("/clients/{email}")
def delete_admin_client(email: str, db: Session = Depends(get_db)) -> dict[str, object]:
    normalized = normalize_account_identifier(email)
    account = db.get(UserAccount, normalized)
    participants = db.execute(select(Participant).where(Participant.user_email == normalized)).scalars().all()
    deleted_files: list[str] = []
    deleted_sessions = 0
    deleted_participants = 0

    for participant in participants:
        sessions = db.execute(
            select(RecordingSession).where(RecordingSession.participant_id == participant.participant_id)
        ).scalars().all()
        for session in sessions:
            deleted_files.extend(delete_session_and_files(db, session, delete_empty_participant=False))
            deleted_sessions += 1
        db.delete(participant)
        deleted_participants += 1

    if account:
        db.delete(account)
    if not account and not participants:
        raise HTTPException(status_code=404, detail="client not found")

    for login_code in db.execute(
        select(EmailLoginCode).where(EmailLoginCode.email == normalized)
    ).scalars().all():
        db.delete(login_code)
    for session_token in db.execute(
        select(UserSessionToken).where(UserSessionToken.email == normalized)
    ).scalars().all():
        db.delete(session_token)

    deleted_files.extend(delete_generated_exports())
    db.commit()
    return {
        "status": "deleted",
        "email": normalized,
        "deleted_participants": deleted_participants,
        "deleted_sessions": deleted_sessions,
        "deleted_files": deleted_files,
    }


@router.delete("/clients/{email}/sessions")
def delete_admin_client_sessions(email: str, db: Session = Depends(get_db)) -> dict[str, object]:
    normalized = normalize_account_identifier(email)
    account = db.get(UserAccount, normalized)
    participants = db.execute(select(Participant).where(Participant.user_email == normalized)).scalars().all()
    if not account and not participants:
        raise HTTPException(status_code=404, detail="account not found")

    deleted_files: list[str] = []
    deleted_sessions = 0
    for participant in participants:
        sessions = db.execute(
            select(RecordingSession).where(RecordingSession.participant_id == participant.participant_id)
        ).scalars().all()
        for session in sessions:
            deleted_files.extend(delete_session_and_files(db, session))
            deleted_sessions += 1
    deleted_files.extend(delete_generated_exports())
    db.commit()
    return {
        "status": "deleted",
        "email": normalized,
        "deleted_sessions": deleted_sessions,
        "deleted_files": deleted_files,
    }


@router.delete("/sessions/{session_id}")
def delete_admin_session(session_id: str, db: Session = Depends(get_db)) -> dict[str, object]:
    session = db.get(RecordingSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    deleted_files = delete_session_and_files(db, session)
    deleted_files.extend(delete_generated_exports())
    db.commit()
    return {
        "status": "deleted",
        "session_id": session_id,
        "deleted_files": deleted_files,
    }


@router.delete("/clips/{clip_id}", response_model=DeleteClipOut)
def delete_clip(clip_id: str, db: Session = Depends(get_db)) -> DeleteClipOut:
    clip = db.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="clip_id not found")

    deleted_files = delete_clip_and_files(db, clip)
    deleted_files.extend(delete_generated_exports())
    db.commit()
    return DeleteClipOut(status="deleted", clip_id=clip_id, deleted_files=deleted_files)


@router.get("/clips/{clip_id}/audio")
def admin_clip_audio(clip_id: str, db: Session = Depends(get_db)):
    clip = db.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="clip_id not found")
    storage = get_storage_backend()
    relative_path = clip.processed_wav_path or clip.raw_audio_path
    if not relative_path:
        raise HTTPException(status_code=404, detail="audio file not found")
    try:
        content = storage.download_bytes(relative_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="audio file not found") from None
    media_type = "audio/wav" if relative_path.endswith(".wav") else "audio/webm"
    return Response(
        content=content,
        media_type=media_type,
        headers={"content-disposition": f'inline; filename="{relative_path.split("/")[-1]}"'},
    )


@router.post("/export", response_model=ExportOut)
def export_dataset(db: Session = Depends(get_db)) -> ExportOut:
    zip_path, file_name = create_export_zip(db)
    return ExportOut(
        status="created",
        file_name=file_name,
        download_path=f"/api/admin/export/download/{file_name}",
    )


@router.get("/export/download/{file_name}")
def download_export(file_name: str):
    if "/" in file_name or "\\" in file_name or not file_name.endswith(".zip"):
        raise HTTPException(status_code=400, detail="invalid export file name")
    path = settings.local_storage_root / "exports" / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="export file not found")
    return FileResponse(path, media_type="application/zip", filename=file_name)
