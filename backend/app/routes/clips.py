from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Clip, Participant, Prompt, RecordingSession, Segment
from ..schemas import ClipUploadOut
from ..services.audio_processing import convert_to_wav, get_wav_info, load_wav_float
from ..services.ids import next_prefixed_id
from ..services.qc import flags_from_json, flags_to_json, run_audio_qc
from ..services.segmentation import segment_repeated_prompt
from ..services.storage import get_storage_backend, infer_extension

router = APIRouter(prefix="/api/clips", tags=["clips"])


def _validate_upload_context(
    db: Session, participant_id: str, session_id: str, prompt_id: str, clip_type: str
) -> Prompt:
    if not participant_id:
        raise HTTPException(status_code=400, detail="missing participant_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="missing session_id")
    if not prompt_id:
        raise HTTPException(status_code=400, detail="missing prompt_id")
    if clip_type not in {"normal", "calibration"}:
        raise HTTPException(status_code=400, detail="clip_type must be normal or calibration")

    participant = db.get(Participant, participant_id)
    if not participant:
        raise HTTPException(status_code=404, detail="participant_id not found")

    session = db.get(RecordingSession, session_id)
    if not session or session.participant_id != participant_id:
        raise HTTPException(status_code=404, detail="session_id not found for participant")

    prompt = db.get(Prompt, "CALIBRATION" if clip_type == "calibration" else prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="prompt_id not found")
    return prompt


@router.post("", response_model=ClipUploadOut)
async def upload_clip(
    audio: UploadFile = File(...),
    participant_id: str = Form(...),
    session_id: str = Form(...),
    prompt_id: str = Form(...),
    clip_type: str = Form("normal"),
    db: Session = Depends(get_db),
) -> ClipUploadOut:
    prompt = _validate_upload_context(db, participant_id, session_id, prompt_id, clip_type)
    storage = get_storage_backend()
    clip_id = next_prefixed_id(db, Clip, "clip_id", "C", 6)

    extension = infer_extension(audio.content_type, audio.filename)
    raw_path = storage.raw_audio_path(participant_id, session_id, clip_id, extension, clip_type)
    file_size = await storage.save_upload(audio, raw_path)
    relative_raw_path = storage.relative(raw_path)

    expected_segments = prompt.target_repetition_count if prompt.segmentation_required else 0
    processed_path = storage.processed_wav_path(participant_id, session_id, clip_id)
    conversion_ok, conversion_error = convert_to_wav(raw_path, processed_path)

    if not conversion_ok:
        flags = ["ffmpeg_conversion_failed"]
        if file_size == 0:
            flags.append("empty_audio")
        clip = Clip(
            clip_id=clip_id,
            participant_id=participant_id,
            session_id=session_id,
            prompt_id=prompt.prompt_id,
            clip_type=clip_type,
            raw_audio_path=relative_raw_path,
            processed_wav_path=None,
            duration_sec=None,
            file_size_bytes=file_size,
            sample_rate_processed=None,
            channels_processed=None,
            auto_qc_status="auto_rejected",
            auto_qc_flags=flags_to_json(sorted(set(flags))),
            segmentation_status="not_required" if not prompt.segmentation_required else "not_attempted",
            detected_segment_count=0,
            expected_segment_count=expected_segments,
            review_status="needs_review",
            review_note=conversion_error,
        )
        db.add(clip)
        db.commit()
        return ClipUploadOut(
            status="uploaded",
            clip_id=clip_id,
            auto_qc_status=clip.auto_qc_status,
            auto_qc_flags=flags_from_json(clip.auto_qc_flags),
            segmentation_status=clip.segmentation_status,
            detected_segment_count=0,
        )

    wav_info = get_wav_info(processed_path)
    samples, _ = load_wav_float(processed_path)
    hard_flags = ["empty_audio"] if file_size == 0 else []
    qc = run_audio_qc(
        duration_sec=wav_info.duration_sec,
        samples=samples,
        prompt=prompt,
        clip_type=clip_type,
        hard_flags=hard_flags,
    )

    segmentation_status = "not_required"
    detected_segment_count = 0
    segment_flags: list[str] = []
    created_segments: list[dict] = []
    if prompt.segmentation_required:
        segment_result = segment_repeated_prompt(
            processed_path,
            output_path_for_index=lambda index: storage.segment_path(
                participant_id, session_id, clip_id, index
            ),
            expected_count=prompt.target_repetition_count,
        )
        segmentation_status = segment_result.status
        detected_segment_count = len(segment_result.segments)
        segment_flags.extend(segment_result.flags)
        created_segments = segment_result.segments
        if segmentation_status == "failed_no_segments":
            segment_flags.append("segmentation_failed_no_segments")

    combined_flags = sorted(set(qc.flags + segment_flags))
    auto_qc_status = qc.status
    if segment_flags and auto_qc_status == "auto_accepted":
        auto_qc_status = "flagged_for_review"
    if segmentation_status == "failed_no_segments":
        auto_qc_status = "flagged_for_review"

    clip = Clip(
        clip_id=clip_id,
        participant_id=participant_id,
        session_id=session_id,
        prompt_id=prompt.prompt_id,
        clip_type=clip_type,
        raw_audio_path=relative_raw_path,
        processed_wav_path=storage.relative(processed_path),
        duration_sec=wav_info.duration_sec,
        file_size_bytes=file_size,
        sample_rate_processed=wav_info.sample_rate,
        channels_processed=wav_info.channels,
        auto_qc_status=auto_qc_status,
        auto_qc_flags=flags_to_json(combined_flags),
        segmentation_status=segmentation_status,
        detected_segment_count=detected_segment_count,
        expected_segment_count=expected_segments,
        review_status="needs_review" if auto_qc_status != "auto_accepted" else None,
    )
    db.add(clip)

    for segment_data in created_segments:
        segment_id = next_prefixed_id(db, Segment, "segment_id", "SEG", 6)
        segment_flags_for_row: list[str] = []
        if segment_data["duration_sec"] < 0.12:
            segment_flags_for_row.append("segment_too_short")
        segment = Segment(
            segment_id=segment_id,
            parent_clip_id=clip_id,
            participant_id=participant_id,
            session_id=session_id,
            prompt_id=prompt.prompt_id,
            segment_index=segment_data["segment_index"],
            segment_audio_path=storage.relative(segment_data["path"]),
            start_time_sec=segment_data["start_time_sec"],
            end_time_sec=segment_data["end_time_sec"],
            duration_sec=segment_data["duration_sec"],
            auto_qc_status="flagged_for_review" if segment_flags_for_row else "auto_accepted",
            auto_qc_flags=flags_to_json(segment_flags_for_row),
            label_type=prompt.label_type,
            contains_vigil=prompt.contains_vigil,
            wake_intent=prompt.wake_intent,
            expected_transcript=prompt.expected_transcript,
        )
        db.add(segment)

    db.commit()
    return ClipUploadOut(
        status="uploaded",
        clip_id=clip_id,
        auto_qc_status=auto_qc_status,
        auto_qc_flags=combined_flags,
        segmentation_status=segmentation_status,
        detected_segment_count=detected_segment_count,
    )
