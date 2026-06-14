from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Clip, Participant, RecordingSession, UserAccount
from ..schemas import AccountClipOut, AccountSessionOut, AuthCodeRequest, AuthCodeRequestOut, AuthCodeVerify, AuthVerifyOut, DeleteClipOut, NameLoginRequest
from ..services.deletion import delete_clip_and_files, delete_generated_exports
from ..services.email_auth import create_login_code, create_session_token, normalize_account_identifier, normalize_email, send_login_code, verify_login_code, verify_session_token
from ..services.storage import get_storage_backend

router = APIRouter(prefix="/api/auth", tags=["auth"])


def validate_account_name(name: str) -> str:
    normalized = normalize_account_identifier(name)
    if len(normalized) < 2:
        raise HTTPException(status_code=400, detail="name must be at least 2 characters")
    if len(normalized) > 80:
        raise HTTPException(status_code=400, detail="name must be 80 characters or fewer")
    if "/" in normalized or "\\" in normalized:
        raise HTTPException(status_code=400, detail="name cannot contain slashes")
    return normalized


@router.post("/name-login", response_model=AuthVerifyOut)
def name_login(payload: NameLoginRequest, db: Session = Depends(get_db)) -> AuthVerifyOut:
    account_name = validate_account_name(payload.name)
    now = datetime.now(UTC).replace(tzinfo=None)
    account = db.get(UserAccount, account_name)
    if not account:
        account = UserAccount(email=account_name, verified=True, last_login_at_utc=now)
        db.add(account)
    else:
        account.verified = True
        account.last_login_at_utc = now
    db.commit()

    token, expires_at = create_session_token(db, account_name)
    return AuthVerifyOut(status="verified", email=account_name, name=account_name, auth_token=token, expires_at_utc=expires_at)


@router.post("/request-code", response_model=AuthCodeRequestOut)
def request_code(payload: AuthCodeRequest, db: Session = Depends(get_db)) -> AuthCodeRequestOut:
    email = normalize_email(payload.email)
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="valid email is required")

    code = create_login_code(db, email)
    email_sent = send_login_code(email, code)
    return AuthCodeRequestOut(
        status="sent" if email_sent else "dev_code",
        dev_code=None if email_sent else code,
    )


@router.post("/verify-code", response_model=AuthVerifyOut)
def verify_code(payload: AuthCodeVerify, db: Session = Depends(get_db)) -> AuthVerifyOut:
    email = normalize_email(payload.email)
    if not verify_login_code(db, email, payload.code):
        raise HTTPException(status_code=400, detail="invalid or expired code")
    token, expires_at = create_session_token(db, email)
    return AuthVerifyOut(status="verified", email=email, name=email, auth_token=token, expires_at_utc=expires_at)


def require_account_token(email: str, x_auth_token: str | None, db: Session) -> str:
    normalized = normalize_account_identifier(email)
    if not verify_session_token(db, normalized, x_auth_token):
        raise HTTPException(status_code=401, detail="login required")
    return normalized


def require_account_token_from_header_or_query(
    email: str, x_auth_token: str | None, token: str | None, db: Session
) -> str:
    return require_account_token(email, x_auth_token or token, db)


@router.get("/accounts/{email}/sessions", response_model=list[AccountSessionOut])
def account_sessions(
    email: str,
    x_auth_token: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> list[AccountSessionOut]:
    normalized = require_account_token(email, x_auth_token, db)
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


@router.get("/accounts/{email}/sessions/{session_id}/clips", response_model=list[AccountClipOut])
def account_session_clips(
    email: str,
    session_id: str,
    x_auth_token: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> list[AccountClipOut]:
    normalized = require_account_token(email, x_auth_token, db)
    session = (
        db.execute(
            select(RecordingSession)
            .join(Participant, RecordingSession.participant_id == Participant.participant_id)
            .where(RecordingSession.session_id == session_id, Participant.user_email == normalized)
        )
        .scalars()
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    clips = db.execute(select(Clip).where(Clip.session_id == session_id).order_by(Clip.created_at_utc)).scalars().all()
    return [
        AccountClipOut(
            clip_id=clip.clip_id,
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
            file_size_bytes=clip.file_size_bytes,
            auto_qc_status=clip.auto_qc_status,
            auto_qc_flags=clip.auto_qc_flags,
            segmentation_status=clip.segmentation_status,
            detected_segment_count=clip.detected_segment_count,
            expected_segment_count=clip.expected_segment_count,
            created_at_utc=clip.created_at_utc,
        )
        for clip in clips
    ]


@router.delete("/accounts/{email}/clips/{clip_id}", response_model=DeleteClipOut)
def delete_account_clip(
    email: str,
    clip_id: str,
    x_auth_token: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> DeleteClipOut:
    normalized = require_account_token(email, x_auth_token, db)
    clip = (
        db.execute(
            select(Clip)
            .join(Participant, Clip.participant_id == Participant.participant_id)
            .where(Clip.clip_id == clip_id, Participant.user_email == normalized)
        )
        .scalars()
        .first()
    )
    if not clip:
        raise HTTPException(status_code=404, detail="clip not found")
    deleted_files = delete_clip_and_files(db, clip)
    deleted_files.extend(delete_generated_exports())
    db.commit()
    return DeleteClipOut(status="deleted", clip_id=clip_id, deleted_files=deleted_files)


@router.get("/accounts/{email}/clips/{clip_id}/audio")
def account_clip_audio(
    email: str,
    clip_id: str,
    token: str | None = Query(default=None),
    x_auth_token: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    normalized = require_account_token_from_header_or_query(email, x_auth_token, token, db)
    clip = (
        db.execute(
            select(Clip)
            .join(Participant, Clip.participant_id == Participant.participant_id)
            .where(Clip.clip_id == clip_id, Participant.user_email == normalized)
        )
        .scalars()
        .first()
    )
    if not clip:
        raise HTTPException(status_code=404, detail="clip not found")
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
