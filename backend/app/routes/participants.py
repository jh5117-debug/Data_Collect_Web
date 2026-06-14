from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Participant
from ..schemas import ParticipantCreate, ParticipantOut
from ..services.email_auth import normalize_account_identifier
from ..services.ids import next_prefixed_id

router = APIRouter(prefix="/api/participants", tags=["participants"])


@router.post("", response_model=ParticipantOut)
def create_participant(payload: ParticipantCreate, db: Session = Depends(get_db)) -> ParticipantOut:
    participant_id = next_prefixed_id(db, Participant, "participant_id", "P", 4)
    participant = Participant(
        participant_id=participant_id,
        user_email=normalize_account_identifier(payload.user_email) if payload.user_email else None,
        english_native_speaker=payload.english_native_speaker,
        recording_device_type=payload.recording_device_type,
    )
    db.add(participant)
    db.commit()
    return ParticipantOut(participant_id=participant_id)
