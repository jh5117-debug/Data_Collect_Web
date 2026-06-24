from datetime import UTC, datetime
from collections.abc import Callable
from dataclasses import dataclass
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
- Each raw clip is written once under audio_raw/ and referenced by manifests.
- by_prompt_group/ contains lightweight clip manifests rather than duplicate audio files.
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


EXPORT_VERSION = "export_job_v1"
ExportProgress = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class ExportZipResult:
    zip_path: Path
    file_name: str
    warning_count: int


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


def _emit_progress(progress: ExportProgress | None, **payload: object) -> None:
    if progress:
        progress(payload)


def _download_progress(processed_items: int, total_items: int) -> float:
    if total_items <= 0:
        return 90.0
    return min(90.0, 5.0 + (processed_items / total_items) * 85.0)


def _group_manifest_row(clip: Clip, audio_path: str) -> dict[str, object]:
    return {
        "clip_id": clip.clip_id,
        "participant_id": clip.participant_id,
        "session_id": clip.session_id,
        "prompt_id": clip.prompt_id,
        "prompt_group": clip.prompt_group,
        "transcript": clip.normalized_transcript or clip.transcript,
        "audio": audio_path,
        "contains_vigil": clip.contains_vigil,
        "wake_intent": clip.wake_intent,
        "is_negative": clip.is_negative,
    }


def create_export_zip(db: Session, progress: ExportProgress | None = None) -> ExportZipResult:
    storage = get_storage_backend()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    export_dir_name = f"vigil_dataset_export_{timestamp}"
    file_name = f"{export_dir_name}.zip"
    zip_path = storage.export_path(file_name)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    _emit_progress(progress, phase="collecting_metadata", progress_percent=2.0, current_item="metadata")
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
    group_rows: dict[str, list[dict[str, object]]] = {group: [] for group in PROMPT_GROUP_EXPORTS}
    warnings: list[dict[str, str]] = []
    total_items = (
        sum(1 for clip in clips if clip.raw_audio_path)
        + sum(1 for clip in clips if clip.processed_wav_path)
        + sum(1 for segment in segments if segment.segment_audio_path)
    )
    processed_items = 0

    _emit_progress(
        progress,
        phase="downloading_audio",
        total_items=total_items,
        processed_items=processed_items,
        progress_percent=5.0,
        current_item="starting audio download",
    )

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        base = export_dir_name
        archive.writestr(f"{base}/README.md", DATASET_README)
        archive.write(PROMPT_CSV_PATH, f"{base}/prompts/prompts_v0_1.csv")
        archive.writestr(f"{base}/audio_raw/.gitkeep", "")
        archive.writestr(f"{base}/processed_wav/.gitkeep", "")
        archive.writestr(f"{base}/segments/.gitkeep", "")
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
            if clip.raw_audio_path:
                processed_items += 1
                _emit_progress(
                    progress,
                    phase="downloading_audio",
                    total_items=total_items,
                    processed_items=processed_items,
                    progress_percent=_download_progress(processed_items, total_items),
                    current_item=clip.raw_audio_path,
                )
                try:
                    raw_bytes = storage.download_bytes(clip.raw_audio_path)
                except FileNotFoundError:
                    warnings.append(
                        {
                            "kind": "missing_raw_audio",
                            "clip_id": clip.clip_id,
                            "path": clip.raw_audio_path,
                        }
                    )
                    continue
                raw_name = f"{clip.clip_id}{Path(clip.raw_audio_path).suffix or '.webm'}"
                canonical_audio_path = f"audio_raw/{raw_name}"
                archive.writestr(f"{base}/{canonical_audio_path}", raw_bytes)
                group_key = clip.prompt_group if clip.prompt_group in PROMPT_GROUP_EXPORTS else "legacy"
                group_rows[group_key].append(
                    _group_manifest_row(clip, canonical_audio_path)
                )
                split = _split_for_clip(clip, participant_email_by_id)
                if split == "eval":
                    qwen_eval_rows.append(_qwen_asr_row(clip))
                    kws_eval_rows.append(_kws_row(clip))
                else:
                    qwen_train_rows.append(_qwen_asr_row(clip))
                    kws_train_rows.append(_kws_row(clip))

            if clip.processed_wav_path:
                processed_items += 1
                _emit_progress(
                    progress,
                    phase="downloading_audio",
                    total_items=total_items,
                    processed_items=processed_items,
                    progress_percent=_download_progress(processed_items, total_items),
                    current_item=clip.processed_wav_path,
                )
                try:
                    wav_bytes = storage.download_bytes(clip.processed_wav_path)
                except FileNotFoundError:
                    warnings.append(
                        {
                            "kind": "missing_processed_wav",
                            "clip_id": clip.clip_id,
                            "path": clip.processed_wav_path,
                        }
                    )
                    continue
                archive.writestr(f"{base}/processed_wav/{clip.clip_id}.wav", wav_bytes)

        for segment in segments:
            if segment.segment_audio_path:
                processed_items += 1
                _emit_progress(
                    progress,
                    phase="downloading_audio",
                    total_items=total_items,
                    processed_items=processed_items,
                    progress_percent=_download_progress(processed_items, total_items),
                    current_item=segment.segment_audio_path,
                )
                try:
                    segment_bytes = storage.download_bytes(segment.segment_audio_path)
                except FileNotFoundError:
                    warnings.append(
                        {
                            "kind": "missing_segment_audio",
                            "segment_id": segment.segment_id,
                            "path": segment.segment_audio_path,
                        }
                    )
                    continue
                archive.writestr(f"{base}/segments/{segment.segment_id}.wav", segment_bytes)

        _emit_progress(
            progress,
            phase="writing_manifests",
            total_items=total_items,
            processed_items=processed_items,
            progress_percent=94.0,
            current_item="manifests",
            warning_count=len(warnings),
        )
        for group in PROMPT_GROUP_EXPORTS:
            archive.writestr(f"{base}/by_prompt_group/{group}/clips.jsonl", rows_to_jsonl(group_rows.get(group, [])))
        archive.writestr(f"{base}/qwen_asr/train.jsonl", rows_to_jsonl(qwen_train_rows))
        archive.writestr(f"{base}/qwen_asr/eval.jsonl", rows_to_jsonl(qwen_eval_rows))
        archive.writestr(f"{base}/keyword_spotting/kws_train.jsonl", rows_to_jsonl(kws_train_rows))
        archive.writestr(f"{base}/keyword_spotting/kws_eval.jsonl", rows_to_jsonl(kws_eval_rows))
        archive.writestr(f"{base}/metadata/export_warnings.jsonl", rows_to_jsonl(warnings))

    _emit_progress(
        progress,
        phase="finalizing",
        total_items=total_items,
        processed_items=processed_items,
        progress_percent=98.0,
        current_item=file_name,
        warning_count=len(warnings),
    )
    return ExportZipResult(zip_path=zip_path, file_name=file_name, warning_count=len(warnings))
