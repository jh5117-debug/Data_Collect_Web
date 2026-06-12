from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Clip, Participant, Prompt, RecordingSession, Segment, UserAccount
from .metadata_export import model_to_dict, rows_to_csv, rows_to_jsonl
from .prompt_loader import PROMPT_CSV_PATH
from .storage import get_storage_backend


ACCOUNT_FIELDS = ["email", "created_at_utc", "last_login_at_utc", "verified"]
PARTICIPANT_FIELDS = ["participant_id", "user_email", "english_native_speaker", "recording_device_type", "created_at_utc"]
SESSION_FIELDS = ["session_id", "participant_id", "batch_id", "status", "created_at_utc", "submitted_at_utc"]
CLIP_FIELDS = [
    "clip_id",
    "participant_id",
    "session_id",
    "prompt_id",
    "clip_type",
    "raw_audio_path",
    "processed_wav_path",
    "duration_sec",
    "file_size_bytes",
    "sample_rate_processed",
    "channels_processed",
    "auto_qc_status",
    "auto_qc_flags",
    "segmentation_status",
    "detected_segment_count",
    "expected_segment_count",
    "created_at_utc",
    "review_status",
    "review_note",
]
SEGMENT_FIELDS = [
    "segment_id",
    "parent_clip_id",
    "participant_id",
    "session_id",
    "prompt_id",
    "segment_index",
    "segment_audio_path",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "auto_qc_status",
    "auto_qc_flags",
    "label_type",
    "contains_vigil",
    "wake_intent",
    "expected_transcript",
    "created_at_utc",
]
QC_FIELDS = [
    "clip_id",
    "participant_id",
    "session_id",
    "prompt_id",
    "clip_type",
    "auto_qc_status",
    "auto_qc_flags",
    "segmentation_status",
    "detected_segment_count",
    "expected_segment_count",
]


DATASET_README = """# Vigil Recorder Dataset Export

This package contains voice trigger data collected for the Vigil wake-word system.

Vigil is the product name and wake word. The word "visual" appears only when it is
part of a hard negative prompt and is not a product name.

Participants were instructed not to include patient names, MRNs, phone numbers,
addresses, or other patient-identifiable information. This MVP collects clean input
audio for internal research and model development; later research may augment the
clean clips with noise, volume changes, or speed changes.

Audio format:
- Raw browser uploads are preserved exactly as received.
- Processed files are normalized to 16 kHz mono WAV with signed 16-bit samples.
- Repeated prompts keep the full recording and may include derived segment WAVs.

Segmentation logic:
- Repeated prompts use simple energy-based silence segmentation.
- The original full recording is always kept, even when segmentation fails.
- Segment count mismatches are flagged for exception-based review.

QC logic:
- Hard failures include empty audio, duration under 0.3 seconds, and FFmpeg failures.
- Soft flags include suspicious duration, very low volume, clipping, and segment
  count mismatches.
- The coordinator should review flagged or rejected clips, not every clip.

Known limitations:
- This MVP does not use ASR or a trained wake-word model for segmentation.
- The admin page has no login. Add authentication before production use.
- Production deployment should use HTTPS for microphone permission and upload safety.
"""


def create_export_zip(db: Session) -> tuple[Path, str]:
    storage = get_storage_backend()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    export_dir_name = f"vigil_dataset_export_{timestamp}"
    file_name = f"{export_dir_name}.zip"
    zip_path = storage.export_path(file_name)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    participants = db.execute(select(Participant).order_by(Participant.participant_id)).scalars().all()
    accounts = db.execute(select(UserAccount).order_by(UserAccount.email)).scalars().all()
    sessions = db.execute(select(RecordingSession).order_by(RecordingSession.session_id)).scalars().all()
    clips = db.execute(select(Clip).order_by(Clip.clip_id)).scalars().all()
    segments = db.execute(select(Segment).order_by(Segment.segment_id)).scalars().all()

    account_rows = [model_to_dict(row, ACCOUNT_FIELDS) for row in accounts]
    participant_rows = [model_to_dict(row, PARTICIPANT_FIELDS) for row in participants]
    session_rows = [model_to_dict(row, SESSION_FIELDS) for row in sessions]
    clip_rows = [model_to_dict(row, CLIP_FIELDS) for row in clips]
    segment_rows = [model_to_dict(row, SEGMENT_FIELDS) for row in segments]
    qc_rows = [model_to_dict(row, QC_FIELDS) for row in clips]

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        base = export_dir_name
        archive.writestr(f"{base}/README.md", DATASET_README)
        archive.write(PROMPT_CSV_PATH, f"{base}/prompts/prompts_v0_1.csv")
        archive.writestr(f"{base}/metadata/accounts.csv", rows_to_csv(account_rows, ACCOUNT_FIELDS))
        archive.writestr(
            f"{base}/metadata/participants.csv",
            rows_to_csv(participant_rows, PARTICIPANT_FIELDS),
        )
        archive.writestr(f"{base}/metadata/sessions.jsonl", rows_to_jsonl(session_rows))
        archive.writestr(f"{base}/metadata/clips.jsonl", rows_to_jsonl(clip_rows))
        archive.writestr(f"{base}/metadata/segments.jsonl", rows_to_jsonl(segment_rows))
        archive.writestr(f"{base}/metadata/qc_report.csv", rows_to_csv(qc_rows, QC_FIELDS))

        for clip in clips:
            for relative_path in (clip.raw_audio_path, clip.processed_wav_path):
                if not relative_path:
                    continue
                source = storage.absolute_path(relative_path)
                if source.exists():
                    archive.write(source, f"{base}/{relative_path}")
                    if relative_path == clip.raw_audio_path:
                        archive.write(source, f"{base}/by_prompt/{clip.prompt_id}/raw_audio/{Path(relative_path).name}")
                    if relative_path == clip.processed_wav_path:
                        archive.write(source, f"{base}/by_prompt/{clip.prompt_id}/processed_wav/{clip.clip_id}.wav")

        for segment in segments:
            source = storage.absolute_path(segment.segment_audio_path)
            if source.exists():
                archive.write(source, f"{base}/{segment.segment_audio_path}")
                archive.write(
                    source,
                    f"{base}/by_prompt/{segment.prompt_id}/segments/{segment.parent_clip_id}_seg{segment.segment_index:03d}.wav",
                )

    return zip_path, file_name
