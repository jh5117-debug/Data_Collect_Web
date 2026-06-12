from sqlalchemy import func, select
from sqlalchemy.orm import Session


def next_prefixed_id(db: Session, model: type, field_name: str, prefix: str, width: int) -> str:
    field = getattr(model, field_name)
    values = db.execute(select(field)).scalars().all()
    max_number = 0
    for value in values:
        if not isinstance(value, str) or not value.startswith(prefix):
            continue
        suffix = value[len(prefix):]
        if suffix.isdigit():
            max_number = max(max_number, int(suffix))

    if max_number == 0:
        count = db.execute(select(func.count()).select_from(model)).scalar_one()
        candidate_number = count + 1
    else:
        candidate_number = max_number + 1

    while True:
        candidate = f"{prefix}{candidate_number:0{width}d}"
        exists = db.get(model, candidate)
        if not exists:
            return candidate
        candidate_number += 1
