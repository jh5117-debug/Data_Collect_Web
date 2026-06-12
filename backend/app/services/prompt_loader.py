import csv
from pathlib import Path

from sqlalchemy.orm import Session

from ..config import settings
from ..models import Prompt


PROMPT_CSV_PATH = Path(__file__).resolve().parents[1] / "prompts" / "prompts_v0_1.csv"


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def load_prompts(db: Session) -> None:
    with PROMPT_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = Prompt(
                prompt_id=row["prompt_id"],
                instruction_text=row["instruction_text"],
                target_phrase=row["target_phrase"],
                display_text=row["display_text"],
                label_type=row["label_type"],
                recording_mode=row["recording_mode"],
                target_repetition_count=int(row["target_repetition_count"]),
                contains_vigil=_to_bool(row["contains_vigil"]),
                wake_intent=_to_bool(row["wake_intent"]),
                segmentation_required=_to_bool(row["segmentation_required"]),
                expected_transcript=row["expected_transcript"],
                prompt_version=settings.prompt_version,
            )
            db.merge(prompt)

    calibration = Prompt(
        prompt_id="CALIBRATION",
        instruction_text="Please stay silent for one second, then say: This is a microphone test for Vigil data collection.",
        target_phrase="This is a microphone test for Vigil data collection.",
        display_text="This is a microphone test for Vigil data collection.",
        label_type="calibration",
        recording_mode="single",
        target_repetition_count=1,
        contains_vigil=True,
        wake_intent=False,
        segmentation_required=False,
        expected_transcript="This is a microphone test for Vigil data collection.",
        prompt_version=settings.prompt_version,
    )
    db.merge(calibration)
    db.commit()
