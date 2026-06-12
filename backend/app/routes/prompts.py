from fastapi import APIRouter, Depends
from sqlalchemy import case, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Prompt
from ..schemas import PromptOut

router = APIRouter(prefix="/api/prompts", tags=["prompts"])
BUG_CHECK_PROMPT_LIMIT = 5


@router.get("", response_model=list[PromptOut])
def list_prompts(db: Session = Depends(get_db)) -> list[Prompt]:
    prompt_order = case(
        (Prompt.prompt_id.like("POS_SINGLE_%"), 1),
        (Prompt.prompt_id.like("POS_REPEAT_%"), 2),
        (Prompt.prompt_id.like("POS_SENT_%"), 3),
        (Prompt.prompt_id.like("NEG_GENERAL_%"), 4),
        (Prompt.prompt_id.like("NEG_HARD_%"), 5),
        else_=99,
    )
    return (
        db.execute(
            select(Prompt)
            .where(Prompt.label_type != "calibration")
            .order_by(prompt_order, Prompt.prompt_id)
            .limit(BUG_CHECK_PROMPT_LIMIT)
        )
        .scalars()
        .all()
    )
