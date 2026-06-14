from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ParticipantCreate(BaseModel):
    user_email: str | None = None
    english_native_speaker: str
    recording_device_type: str


class ParticipantOut(BaseModel):
    participant_id: str


class SessionCreate(BaseModel):
    participant_id: str
    batch_id: str = "vigil_batch_v0_1"


class SessionOut(BaseModel):
    session_id: str


class PromptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prompt_id: str
    instruction_text: str
    target_phrase: str
    display_text: str
    label_type: str
    recording_mode: str
    target_repetition_count: int
    contains_vigil: bool
    wake_intent: bool
    segmentation_required: bool
    expected_transcript: str
    prompt_version: str


class ClipUploadOut(BaseModel):
    status: str
    clip_id: str
    prompt_group: str
    transcript: str
    normalized_transcript: str
    contains_vigil: bool
    wake_intent: bool
    is_negative: bool
    auto_qc_status: str
    auto_qc_flags: list[str]
    segmentation_status: str
    detected_segment_count: int


class SummaryOut(BaseModel):
    batch_id: str
    participants: int
    sessions: int
    submitted_sessions: int
    total_clips: int
    total_segments: int
    positive_clips: int
    negative_clips: int
    auto_accepted: int
    flagged: int
    rejected: int
    prompt_group_counts: dict[str, int]


class FlaggedClipOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    clip_id: str
    participant_id: str
    session_id: str
    prompt_id: str
    prompt_group: str
    prompt_title: str
    transcript: str
    normalized_transcript: str
    contains_vigil: bool
    wake_intent: bool
    is_negative: bool
    clip_type: str
    duration_sec: float | None
    auto_qc_status: str
    auto_qc_flags: str
    segmentation_status: str
    detected_segment_count: int
    expected_segment_count: int
    created_at_utc: datetime


class AdminClientOut(BaseModel):
    email: str | None
    verified: bool
    account_created_at_utc: datetime | None
    last_login_at_utc: datetime | None
    participant_count: int
    session_count: int
    submitted_session_count: int
    clip_count: int
    positive_clip_count: int
    negative_clip_count: int
    segment_count: int


class AdminClipOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    clip_id: str
    participant_id: str
    user_email: str | None
    session_id: str
    prompt_id: str
    prompt_group: str
    prompt_title: str
    transcript: str
    normalized_transcript: str
    contains_vigil: bool
    wake_intent: bool
    is_negative: bool
    clip_type: str
    raw_audio_path: str
    processed_wav_path: str | None
    duration_sec: float | None
    file_size_bytes: int
    auto_qc_status: str
    auto_qc_flags: str
    segmentation_status: str
    detected_segment_count: int
    expected_segment_count: int
    created_at_utc: datetime


class DeleteClipOut(BaseModel):
    status: str
    clip_id: str
    deleted_files: list[str]


class ExportOut(BaseModel):
    status: str
    file_name: str
    download_path: str


class AuthCodeRequest(BaseModel):
    email: str


class AuthCodeRequestOut(BaseModel):
    status: str
    dev_code: str | None = None


class AuthCodeVerify(BaseModel):
    email: str
    code: str


class NameLoginRequest(BaseModel):
    name: str


class AuthVerifyOut(BaseModel):
    status: str
    email: str
    name: str | None = None
    auth_token: str
    expires_at_utc: datetime


class AccountSessionOut(BaseModel):
    session_id: str
    batch_id: str
    status: str
    created_at_utc: datetime
    submitted_at_utc: datetime | None
    clip_count: int
    positive_clip_count: int
    negative_clip_count: int


class AccountClipOut(BaseModel):
    clip_id: str
    session_id: str
    prompt_id: str
    prompt_group: str
    prompt_title: str
    transcript: str
    normalized_transcript: str
    contains_vigil: bool
    wake_intent: bool
    is_negative: bool
    clip_type: str
    duration_sec: float | None
    file_size_bytes: int
    auto_qc_status: str
    auto_qc_flags: str
    segmentation_status: str
    detected_segment_count: int
    expected_segment_count: int
    created_at_utc: datetime
