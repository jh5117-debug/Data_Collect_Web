from sqlalchemy import func, select
from sqlalchemy.orm import Session


def next_prefixed_id(db: Session, model: type, field_name: str, prefix: str, width: int) -> str:
    field = getattr(model, field_name)
    count = db.execute(select(func.count()).select_from(model)).scalar_one()
    candidate_number = count + 1

    while True:
        candidate = f"{prefix}{candidate_number:0{width}d}"
        exists = db.get(model, candidate)
        if not exists:
            return candidate
        candidate_number += 1
