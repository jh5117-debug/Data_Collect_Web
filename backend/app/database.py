from collections.abc import Generator
import logging
import time

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

from .config import settings


logger = logging.getLogger(__name__)

is_sqlite = settings.database_url.startswith("sqlite")
is_supabase_transaction_pooler = (
    "pooler.supabase.com:6543" in settings.database_url
    or "pooler.supabase.co:6543" in settings.database_url
)

if is_sqlite:
    connect_args = {"check_same_thread": False}
else:
    connect_args = {"prepare_threshold": None}

engine_kwargs = {
    "connect_args": connect_args,
    "future": True,
}
if not is_sqlite:
    engine_kwargs["pool_pre_ping"] = True
if is_supabase_transaction_pooler:
    engine_kwargs["poolclass"] = NullPool

engine = create_engine(settings.database_url, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(max_attempts: int = 5, retry_delay_seconds: float = 3.0) -> None:
    from . import models  # noqa: F401

    for attempt in range(1, max_attempts + 1):
        try:
            Base.metadata.create_all(bind=engine)
            ensure_sqlite_schema()
            return
        except OperationalError:
            if attempt == max_attempts:
                raise
            logger.warning(
                "Database initialization failed on attempt %s/%s; retrying in %.1fs",
                attempt,
                max_attempts,
                retry_delay_seconds,
                exc_info=True,
            )
            time.sleep(retry_delay_seconds)


def ensure_sqlite_schema() -> None:
    if not settings.database_url.startswith("sqlite"):
        return

    with engine.begin() as connection:
        columns = connection.execute(text("PRAGMA table_info(participants)")).fetchall()
        column_names = {column[1] for column in columns}
        if "user_email" not in column_names:
            connection.execute(text("ALTER TABLE participants ADD COLUMN user_email VARCHAR"))

        clip_columns = connection.execute(text("PRAGMA table_info(clips)")).fetchall()
        clip_column_names = {column[1] for column in clip_columns}
        clip_additions = {
            "prompt_group": "ALTER TABLE clips ADD COLUMN prompt_group VARCHAR NOT NULL DEFAULT 'legacy'",
            "prompt_title": "ALTER TABLE clips ADD COLUMN prompt_title TEXT NOT NULL DEFAULT ''",
            "transcript": "ALTER TABLE clips ADD COLUMN transcript TEXT NOT NULL DEFAULT ''",
            "normalized_transcript": "ALTER TABLE clips ADD COLUMN normalized_transcript TEXT NOT NULL DEFAULT ''",
            "contains_vigil": "ALTER TABLE clips ADD COLUMN contains_vigil BOOLEAN NOT NULL DEFAULT 0",
            "wake_intent": "ALTER TABLE clips ADD COLUMN wake_intent BOOLEAN NOT NULL DEFAULT 0",
            "is_negative": "ALTER TABLE clips ADD COLUMN is_negative BOOLEAN NOT NULL DEFAULT 0",
        }
        for column_name, statement in clip_additions.items():
            if column_name not in clip_column_names:
                connection.execute(text(statement))

        connection.execute(
            text(
                """
                UPDATE clips
                SET
                    prompt_group = COALESCE(NULLIF(prompt_group, ''), 'legacy'),
                    prompt_title = COALESCE(NULLIF(prompt_title, ''), prompt_id),
                    transcript = COALESCE(
                        NULLIF(transcript, ''),
                        (SELECT expected_transcript FROM prompts WHERE prompts.prompt_id = clips.prompt_id),
                        prompt_id
                    ),
                    normalized_transcript = COALESCE(
                        NULLIF(normalized_transcript, ''),
                        (SELECT expected_transcript FROM prompts WHERE prompts.prompt_id = clips.prompt_id),
                        prompt_id
                    ),
                    contains_vigil = CASE
                        WHEN prompt_group != 'legacy' THEN contains_vigil
                        WHEN (SELECT contains_vigil FROM prompts WHERE prompts.prompt_id = clips.prompt_id) = 1 THEN 1
                        ELSE contains_vigil
                    END,
                    wake_intent = CASE
                        WHEN prompt_group != 'legacy' THEN wake_intent
                        WHEN (SELECT wake_intent FROM prompts WHERE prompts.prompt_id = clips.prompt_id) = 1 THEN 1
                        ELSE wake_intent
                    END,
                    is_negative = COALESCE(is_negative, 0)
                """
            )
        )
