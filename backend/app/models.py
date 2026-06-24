from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Participant(Base):
    __tablename__ = "participants"

    participant_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_email: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    english_native_speaker: Mapped[str] = mapped_column(String, nullable=False)
    recording_device_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    sessions: Mapped[list["RecordingSession"]] = relationship(back_populates="participant")


class RecordingSession(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    participant_id: Mapped[str] = mapped_column(ForeignKey("participants.participant_id"), nullable=False)
    batch_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="in_progress", nullable=False)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    submitted_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    participant: Mapped[Participant] = relationship(back_populates="sessions")


class UserAccount(Base):
    __tablename__ = "user_accounts"

    email: Mapped[str] = mapped_column(String, primary_key=True)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    last_login_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class EmailLoginCode(Base):
    __tablename__ = "email_login_codes"

    code_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String, index=True, nullable=False)
    code: Mapped[str] = mapped_column(String, nullable=False)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    expires_at_utc: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class UserSessionToken(Base):
    __tablename__ = "user_session_tokens"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, index=True, nullable=False)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    expires_at_utc: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class Prompt(Base):
    __tablename__ = "prompts"

    prompt_id: Mapped[str] = mapped_column(String, primary_key=True)
    instruction_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_phrase: Mapped[str] = mapped_column(Text, nullable=False)
    display_text: Mapped[str] = mapped_column(Text, nullable=False)
    label_type: Mapped[str] = mapped_column(String, nullable=False)
    recording_mode: Mapped[str] = mapped_column(String, nullable=False)
    target_repetition_count: Mapped[int] = mapped_column(Integer, nullable=False)
    contains_vigil: Mapped[bool] = mapped_column(Boolean, nullable=False)
    wake_intent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    segmentation_required: Mapped[bool] = mapped_column(Boolean, nullable=False)
    expected_transcript: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String, nullable=False)


class Clip(Base):
    __tablename__ = "clips"

    clip_id: Mapped[str] = mapped_column(String, primary_key=True)
    participant_id: Mapped[str] = mapped_column(ForeignKey("participants.participant_id"), nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), nullable=False)
    prompt_id: Mapped[str] = mapped_column(String, nullable=False)
    prompt_group: Mapped[str] = mapped_column(String, default="legacy", nullable=False)
    prompt_title: Mapped[str] = mapped_column(Text, default="", nullable=False)
    transcript: Mapped[str] = mapped_column(Text, default="", nullable=False)
    normalized_transcript: Mapped[str] = mapped_column(Text, default="", nullable=False)
    contains_vigil: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    wake_intent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_negative: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    clip_type: Mapped[str] = mapped_column(String, default="normal", nullable=False)
    raw_audio_path: Mapped[str] = mapped_column(Text, nullable=False)
    processed_wav_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sample_rate_processed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels_processed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    auto_qc_status: Mapped[str] = mapped_column(String, nullable=False)
    auto_qc_flags: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    segmentation_status: Mapped[str] = mapped_column(String, default="not_required", nullable=False)
    detected_segment_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    expected_segment_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    review_status: Mapped[str | None] = mapped_column(String, nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)


class Segment(Base):
    __tablename__ = "segments"

    segment_id: Mapped[str] = mapped_column(String, primary_key=True)
    parent_clip_id: Mapped[str] = mapped_column(ForeignKey("clips.clip_id"), nullable=False)
    participant_id: Mapped[str] = mapped_column(ForeignKey("participants.participant_id"), nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), nullable=False)
    prompt_id: Mapped[str] = mapped_column(String, nullable=False)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    segment_audio_path: Mapped[str] = mapped_column(Text, nullable=False)
    start_time_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_time_sec: Mapped[float] = mapped_column(Float, nullable=False)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False)
    auto_qc_status: Mapped[str] = mapped_column(String, nullable=False)
    auto_qc_flags: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    label_type: Mapped[str] = mapped_column(String, nullable=False)
    contains_vigil: Mapped[bool] = mapped_column(Boolean, nullable=False)
    wake_intent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    expected_transcript: Mapped[str] = mapped_column(Text, nullable=False)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class ExportJob(Base):
    __tablename__ = "export_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, default="queued", nullable=False, index=True)
    phase: Mapped[str] = mapped_column(String, default="queued", nullable=False)
    total_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    current_item: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_name: Mapped[str | None] = mapped_column(String, nullable=True)
    local_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    warning_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    started_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at_utc: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    completed_at_utc: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    export_version: Mapped[str] = mapped_column(String, default="export_job_v1", nullable=False)
