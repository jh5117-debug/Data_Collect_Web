from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Clip, Participant, Prompt, RecordingSession
from ..schemas import ClipUploadOut
from ..services.ids import next_prefixed_id
from ..services.prompt_groups import derive_prompt_group_info, legacy_prompt_group_info
from ..services.qc import flags_to_json
from ..services.storage import get_storage_backend, infer_extension

router = APIRouter(prefix="/api/clips", tags=["clips"])


def _validate_upload_context(
    db: Session, participant_id: str, session_id: str, clip_type: str
) -> tuple[Participant, RecordingSession]:
    if not participant_id:
        raise HTTPException(status_code=400, detail="missing participant_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="missing session_id")
    if clip_type not in {"normal", "calibration"}:
        raise HTTPException(status_code=400, detail="clip_type must be normal or calibration")

    participant = db.get(Participant, participant_id)
    if not participant:
        raise HTTPException(status_code=404, detail="participant_id not found")

    session = db.get(RecordingSession, session_id)
    if not session or session.participant_id != participant_id:
        raise HTTPException(status_code=404, detail="session_id not found for participant")

    return participant, session


def _prompt_for_new_group(group_info) -> Prompt:
    return Prompt(
        prompt_id=group_info.prompt_group,
        instruction_text="Prompt group recording",
        target_phrase=group_info.transcript,
        display_text=group_info.prompt_title,
        label_type="negative" if group_info.is_negative else "positive",
        recording_mode="single",
        target_repetition_count=1,
        contains_vigil=group_info.contains_vigil,
        wake_intent=group_info.wake_intent,
        segmentation_required=False,
        expected_transcript=group_info.normalized_transcript,
        prompt_version="prompt_groups_v1",
    )


@router.post("", response_model=ClipUploadOut)
async def upload_clip(
    audio: UploadFile = File(...),
    participant_id: str = Form(...),
    session_id: str = Form(...),
    prompt_id: str | None = Form(None),
    prompt_group: str | None = Form(None),
    transcript: str | None = Form(None),
    clip_type: str = Form("normal"),
    db: Session = Depends(get_db),
) -> ClipUploadOut:
    _validate_upload_context(db, participant_id, session_id, clip_type)
    active_prompt_id = (prompt_id or "").strip()

    if clip_type == "calibration":
        prompt = db.get(Prompt, "CALIBRATION")
        if not prompt:
            raise HTTPException(status_code=404, detail="prompt_id not found")
        group_info = legacy_prompt_group_info(prompt.prompt_id, prompt.expected_transcript)
    elif prompt_group:
        group_info = derive_prompt_group_info(prompt_group, transcript)
        active_prompt_id = group_info.prompt_group
        prompt = _prompt_for_new_group(group_info)
    else:
        if not active_prompt_id:
            raise HTTPException(status_code=400, detail="missing prompt_group")
        prompt = db.get(Prompt, active_prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail="prompt_id not found")
        group_info = legacy_prompt_group_info(prompt.prompt_id, prompt.expected_transcript)

    storage = get_storage_backend()
    clip_id = next_prefixed_id(db, Clip, "clip_id", "C", 6)

    extension = infer_extension(audio.content_type, audio.filename)
    raw_path = storage.raw_audio_path(participant_id, session_id, clip_id, extension, clip_type)
    file_size = await storage.save_upload(audio, raw_path)
    relative_raw_path = storage.relative(raw_path)

    combined_flags = ["empty_audio"] if file_size == 0 else []
    auto_qc_status = "auto_rejected" if combined_flags else "auto_accepted"
    segmentation_status = "not_required"
    detected_segment_count = 0

    clip = Clip(
        clip_id=clip_id,
        participant_id=participant_id,
        session_id=session_id,
        prompt_id=active_prompt_id or prompt.prompt_id,
        prompt_group=group_info.prompt_group,
        prompt_title=group_info.prompt_title,
        transcript=group_info.transcript,
        normalized_transcript=group_info.normalized_transcript,
        contains_vigil=group_info.contains_vigil,
        wake_intent=group_info.wake_intent,
        is_negative=group_info.is_negative,
        clip_type=clip_type,
        raw_audio_path=relative_raw_path,
        processed_wav_path=None,
        duration_sec=None,
        file_size_bytes=file_size,
        sample_rate_processed=None,
        channels_processed=None,
        auto_qc_status=auto_qc_status,
        auto_qc_flags=flags_to_json(combined_flags),
        segmentation_status=segmentation_status,
        detected_segment_count=detected_segment_count,
        expected_segment_count=0,
        review_status="needs_review" if auto_qc_status != "auto_accepted" else None,
    )
    db.add(clip)
    db.commit()
    return ClipUploadOut(
        status="uploaded",
        clip_id=clip_id,
        prompt_group=clip.prompt_group,
        transcript=clip.transcript,
        normalized_transcript=clip.normalized_transcript,
        contains_vigil=clip.contains_vigil,
        wake_intent=clip.wake_intent,
        is_negative=clip.is_negative,
        auto_qc_status=auto_qc_status,
        auto_qc_flags=combined_flags,
        segmentation_status=segmentation_status,
        detected_segment_count=detected_segment_count,
    )
