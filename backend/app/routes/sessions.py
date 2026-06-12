from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Participant, RecordingSession
from ..schemas import SessionCreate, SessionOut
from ..services.ids import next_prefixed_id

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)) -> SessionOut:
    participant = db.get(Participant, payload.participant_id)
    if not participant:
        raise HTTPException(status_code=404, detail="participant_id not found")

    session_id = next_prefixed_id(db, RecordingSession, "session_id", "S", 4)
    session = RecordingSession(
        session_id=session_id,
        participant_id=payload.participant_id,
        batch_id=payload.batch_id,
        status="in_progress",
    )
    db.add(session)
    db.commit()
    return SessionOut(session_id=session_id)


@router.post("/{session_id}/submit")
def submit_session(session_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    session = db.get(RecordingSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session_id not found")

    session.status = "submitted"
    session.submitted_at_utc = datetime.now(UTC).replace(tzinfo=None)
    db.commit()
    return {"status": "submitted", "session_id": session_id}
