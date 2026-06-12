from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Clip, EmailLoginCode, Participant, RecordingSession, Segment, UserAccount, UserSessionToken
from ..schemas import AccountSessionOut, AdminClientOut, AdminClipOut, DeleteClipOut, ExportOut, FlaggedClipOut, SummaryOut
from ..services.deletion import delete_clip_and_files, delete_session_clips_and_files
from ..services.export import create_export_zip
from ..services.email_auth import normalize_email
from ..services.storage import get_storage_backend

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/summary", response_model=SummaryOut)
def admin_summary(db: Session = Depends(get_db)) -> SummaryOut:
    accounts = db.execute(select(func.count()).select_from(UserAccount)).scalar_one()
    participants = db.execute(select(func.count()).select_from(Participant)).scalar_one()
    sessions = db.execute(select(func.count()).select_from(RecordingSession)).scalar_one()
    submitted_sessions = db.execute(
        select(func.count()).select_from(RecordingSession).where(RecordingSession.status == "submitted")
    ).scalar_one()
    total_clips = db.execute(select(func.count()).select_from(Clip)).scalar_one()
    total_segments = db.execute(select(func.count()).select_from(Segment)).scalar_one()
    auto_accepted = db.execute(
        select(func.count()).select_from(Clip).where(Clip.auto_qc_status == "auto_accepted")
    ).scalar_one()
    flagged = db.execute(
        select(func.count()).select_from(Clip).where(Clip.auto_qc_status == "flagged_for_review")
    ).scalar_one()
    rejected = db.execute(
        select(func.count()).select_from(Clip).where(Clip.auto_qc_status == "auto_rejected")
    ).scalar_one()

    return SummaryOut(
        batch_id=settings.default_batch_id,
        participants=accounts or participants,
        sessions=sessions,
        submitted_sessions=submitted_sessions,
        total_clips=total_clips,
        total_segments=total_segments,
        auto_accepted=auto_accepted,
        flagged=flagged,
        rejected=rejected,
    )


@router.get("/flagged", response_model=list[FlaggedClipOut])
def flagged_clips(db: Session = Depends(get_db)) -> list[Clip]:
    return (
        db.execute(
            select(Clip)
            .where(or_(Clip.auto_qc_status == "flagged_for_review", Clip.auto_qc_status == "auto_rejected"))
            .order_by(Clip.created_at_utc.desc())
            .limit(100)
        )
        .scalars()
        .all()
    )


@router.get("/clients", response_model=list[AdminClientOut])
def admin_clients(db: Session = Depends(get_db)) -> list[AdminClientOut]:
    accounts = db.execute(select(UserAccount).order_by(UserAccount.created_at_utc.desc())).scalars().all()
    account_emails = {account.email for account in accounts}
    participant_emails = {
        email
        for email in db.execute(select(Participant.user_email).where(Participant.user_email.is_not(None))).scalars().all()
        if email
    }
    all_emails = sorted(account_emails | participant_emails)

    rows: list[AdminClientOut] = []
    for email in all_emails:
        account = db.get(UserAccount, email)
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
                verified=account.verified if account else False,
                account_created_at_utc=account.created_at_utc if account else None,
                last_login_at_utc=account.last_login_at_utc if account else None,
                participant_count=len(participant_ids),
                session_count=len(session_ids),
                submitted_session_count=submitted_session_count,
                clip_count=clip_count,
                segment_count=segment_count,
            )
        )
    return rows


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
    return [
        AdminClipOut(
            clip_id=clip.clip_id,
            participant_id=clip.participant_id,
            user_email=user_email,
            session_id=clip.session_id,
            prompt_id=clip.prompt_id,
            clip_type=clip.clip_type,
            raw_audio_path=clip.raw_audio_path,
            processed_wav_path=clip.processed_wav_path,
            duration_sec=clip.duration_sec,
            file_size_bytes=clip.file_size_bytes,
            auto_qc_status=clip.auto_qc_status,
            auto_qc_flags=clip.auto_qc_flags,
            segmentation_status=clip.segmentation_status,
            detected_segment_count=clip.detected_segment_count,
            expected_segment_count=clip.expected_segment_count,
            created_at_utc=clip.created_at_utc,
        )
        for clip, user_email in rows
    ]


@router.get("/clients/{email}/sessions", response_model=list[AccountSessionOut])
def admin_client_sessions(email: str, db: Session = Depends(get_db)) -> list[AccountSessionOut]:
    normalized = normalize_email(email)
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
        output.append(
            AccountSessionOut(
                session_id=session.session_id,
                batch_id=session.batch_id,
                status=session.status,
                created_at_utc=session.created_at_utc,
                submitted_at_utc=session.submitted_at_utc,
                clip_count=clip_count,
            )
        )
    return output


@router.get("/clients/{email}/clips", response_model=list[AdminClipOut])
def admin_client_clips(email: str, db: Session = Depends(get_db)) -> list[AdminClipOut]:
    normalized = normalize_email(email)
    rows = (
        db.execute(
            select(Clip, Participant.user_email)
            .join(Participant, Clip.participant_id == Participant.participant_id)
            .where(Participant.user_email == normalized)
            .order_by(Clip.created_at_utc.desc())
        )
        .all()
    )
    return [
        AdminClipOut(
            clip_id=clip.clip_id,
            participant_id=clip.participant_id,
            user_email=user_email,
            session_id=clip.session_id,
            prompt_id=clip.prompt_id,
            clip_type=clip.clip_type,
            raw_audio_path=clip.raw_audio_path,
            processed_wav_path=clip.processed_wav_path,
            duration_sec=clip.duration_sec,
            file_size_bytes=clip.file_size_bytes,
            auto_qc_status=clip.auto_qc_status,
            auto_qc_flags=clip.auto_qc_flags,
            segmentation_status=clip.segmentation_status,
            detected_segment_count=clip.detected_segment_count,
            expected_segment_count=clip.expected_segment_count,
            created_at_utc=clip.created_at_utc,
        )
        for clip, user_email in rows
    ]


@router.delete("/clients/{email}")
def delete_admin_client(email: str, db: Session = Depends(get_db)) -> dict[str, object]:
    normalized = normalize_email(email)
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
            deleted_files.extend(delete_session_clips_and_files(db, session))
            db.delete(session)
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

    db.commit()
    return {
        "status": "deleted",
        "email": normalized,
        "deleted_participants": deleted_participants,
        "deleted_sessions": deleted_sessions,
        "deleted_files": deleted_files,
    }


@router.delete("/clips/{clip_id}", response_model=DeleteClipOut)
def delete_clip(clip_id: str, db: Session = Depends(get_db)) -> DeleteClipOut:
    clip = db.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="clip_id not found")

    deleted_files = delete_clip_and_files(db, clip)
    db.commit()
    return DeleteClipOut(status="deleted", clip_id=clip_id, deleted_files=deleted_files)


@router.get("/clips/{clip_id}/audio")
def admin_clip_audio(clip_id: str, db: Session = Depends(get_db)):
    clip = db.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="clip_id not found")
    storage = get_storage_backend()
    relative_path = clip.processed_wav_path or clip.raw_audio_path
    path = storage.absolute_path(relative_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio file not found")
    media_type = "audio/wav" if relative_path.endswith(".wav") else "audio/webm"
    return FileResponse(path, media_type=media_type, filename=path.name)


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
