from datetime import UTC, datetime
import hashlib
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
    "prompt_group",
    "prompt_title",
    "transcript",
    "normalized_transcript",
    "contains_vigil",
    "wake_intent",
    "is_negative",
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
    "prompt_group",
    "prompt_title",
    "transcript",
    "normalized_transcript",
    "contains_vigil",
    "wake_intent",
    "is_negative",
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

Prompt groups:
- P1_vigil_only: fixed transcript "Vigil".
- P2_phrase_plus_vigil: participant phrase or sentence containing the exact word "Vigil".
- P3_vigil_plus_phrase: participant phrase or sentence containing the exact word "Vigil".
- P4_negative: confusing words or sentences that must not contain the exact word "Vigil".

Audio format:
- Raw browser uploads are preserved exactly as received.
- Online collection does not convert audio or run temporal segmentation.
- Convert raw audio to 16 kHz mono WAV offline before ASR training or Qwen review.

QC logic:
- Online QC only checks upload-level issues such as empty audio and transcript rules.
- Semantic review, audio quality checks, WAV conversion, Qwen ASR, and final train/eval
  manifest generation should run offline after export.

Known limitations:
- This collection deployment does not use ASR or a trained wake-word model online.
- The admin page has no login. Add authentication before production use.
- Production deployment should use HTTPS for microphone permission and upload safety.
- Raw manifests use a simple deterministic account-independent split when account identity
  is available.
"""


PROMPT_GROUP_EXPORTS = [
    "P1_vigil_only",
    "P2_phrase_plus_vigil",
    "P3_vigil_plus_phrase",
    "P4_negative",
    "legacy",
]


def _split_for_clip(clip: Clip, participant_email_by_id: dict[str, str | None]) -> str:
    identity = participant_email_by_id.get(clip.participant_id) or clip.participant_id
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return "eval" if int(digest[:8], 16) % 5 == 0 else "train"


def _manifest_audio_path(clip: Clip) -> str:
    suffix = Path(clip.raw_audio_path).suffix or ".webm"
    return f"audio_raw/{clip.clip_id}{suffix}"


def _qwen_asr_row(clip: Clip) -> dict[str, str]:
    return {
        "audio": _manifest_audio_path(clip),
        "text": f"language English<asr_text>{clip.normalized_transcript or clip.transcript}",
    }


def _kws_row(clip: Clip) -> dict[str, object]:
    return {
        "audio": _manifest_audio_path(clip),
        "transcript": clip.normalized_transcript or clip.transcript,
        "prompt_group": clip.prompt_group,
        "contains_vigil": clip.contains_vigil,
        "wake_intent": clip.wake_intent,
        "is_negative": clip.is_negative,
    }


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
    participant_email_by_id = {participant.participant_id: participant.user_email for participant in participants}
    qwen_train_rows: list[dict[str, object]] = []
    qwen_eval_rows: list[dict[str, object]] = []
    kws_train_rows: list[dict[str, object]] = []
    kws_eval_rows: list[dict[str, object]] = []

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        base = export_dir_name
        archive.writestr(f"{base}/README.md", DATASET_README)
        archive.write(PROMPT_CSV_PATH, f"{base}/prompts/prompts_v0_1.csv")
        for group in PROMPT_GROUP_EXPORTS:
            archive.writestr(f"{base}/by_prompt_group/{group}/.gitkeep", "")
        archive.writestr(f"{base}/metadata/accounts.csv", rows_to_csv(account_rows, ACCOUNT_FIELDS))
        archive.writestr(
            f"{base}/metadata/participants.csv",
            rows_to_csv(participant_rows, PARTICIPANT_FIELDS),
        )
        archive.writestr(f"{base}/metadata/sessions.jsonl", rows_to_jsonl(session_rows))
        archive.writestr(f"{base}/metadata/clips.csv", rows_to_csv(clip_rows, CLIP_FIELDS))
        archive.writestr(f"{base}/metadata/clips.jsonl", rows_to_jsonl(clip_rows))
        archive.writestr(f"{base}/metadata/segments.jsonl", rows_to_jsonl(segment_rows))
        archive.writestr(f"{base}/metadata/qc_report.csv", rows_to_csv(qc_rows, QC_FIELDS))

        for clip in clips:
            if clip.raw_audio_path and storage.exists(clip.raw_audio_path):
                raw_bytes = storage.download_bytes(clip.raw_audio_path)
                raw_name = f"{clip.clip_id}{Path(clip.raw_audio_path).suffix or '.webm'}"
                archive.writestr(f"{base}/{clip.raw_audio_path}", raw_bytes)
                archive.writestr(f"{base}/raw_audio/{raw_name}", raw_bytes)
                archive.writestr(f"{base}/audio_raw/{raw_name}", raw_bytes)
                archive.writestr(f"{base}/by_prompt_group/{clip.prompt_group}/raw_audio/{raw_name}", raw_bytes)
                split = _split_for_clip(clip, participant_email_by_id)
                if split == "eval":
                    qwen_eval_rows.append(_qwen_asr_row(clip))
                    kws_eval_rows.append(_kws_row(clip))
                else:
                    qwen_train_rows.append(_qwen_asr_row(clip))
                    kws_train_rows.append(_kws_row(clip))

            if clip.processed_wav_path and storage.exists(clip.processed_wav_path):
                wav_bytes = storage.download_bytes(clip.processed_wav_path)
                archive.writestr(f"{base}/{clip.processed_wav_path}", wav_bytes)
                archive.writestr(f"{base}/by_prompt_group/{clip.prompt_group}/processed_wav/{clip.clip_id}.wav", wav_bytes)

        for segment in segments:
            if storage.exists(segment.segment_audio_path):
                segment_bytes = storage.download_bytes(segment.segment_audio_path)
                archive.writestr(f"{base}/{segment.segment_audio_path}", segment_bytes)
                archive.writestr(
                    f"{base}/legacy_segments/{segment.prompt_id}/{segment.parent_clip_id}_seg{segment.segment_index:03d}.wav",
                    segment_bytes,
                )

        archive.writestr(f"{base}/qwen_asr/train.jsonl", rows_to_jsonl(qwen_train_rows))
        archive.writestr(f"{base}/qwen_asr/eval.jsonl", rows_to_jsonl(qwen_eval_rows))
        archive.writestr(f"{base}/keyword_spotting/kws_train.jsonl", rows_to_jsonl(kws_train_rows))
        archive.writestr(f"{base}/keyword_spotting/kws_eval.jsonl", rows_to_jsonl(kws_eval_rows))

    return zip_path, file_name
