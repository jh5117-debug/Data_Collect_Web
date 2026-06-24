from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import ExportJob
from .export import EXPORT_VERSION, create_export_zip


EXPORT_JOB_TERMINAL_STATUSES = {"completed", "completed_with_warnings", "failed"}

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vigil-export")
_scheduled_job_ids: set[str] = set()
_schedule_lock = Lock()


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def create_export_job(db: Session) -> ExportJob:
    now = _utcnow()
    job = ExportJob(
        job_id=str(uuid4()),
        status="queued",
        phase="queued",
        progress_percent=0.0,
        created_at_utc=now,
        updated_at_utc=now,
        export_version=EXPORT_VERSION,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    schedule_export_job(job.job_id)
    return job


def schedule_export_job(job_id: str) -> None:
    with _schedule_lock:
        if job_id in _scheduled_job_ids:
            return
        _scheduled_job_ids.add(job_id)
    try:
        _executor.submit(_run_export_job, job_id)
    except Exception:
        with _schedule_lock:
            _scheduled_job_ids.discard(job_id)
        raise


def recover_export_jobs() -> None:
    queued_job_ids: list[str] = []
    now = _utcnow()
    db = SessionLocal()
    try:
        interrupted_jobs = (
            db.execute(select(ExportJob).where(ExportJob.status == "running"))
            .scalars()
            .all()
        )
        for job in interrupted_jobs:
            job.status = "failed"
            job.phase = "failed"
            job.error_message = "Export worker was interrupted before completion. Start a new export."
            job.updated_at_utc = now
            job.completed_at_utc = now

        queued_jobs = (
            db.execute(select(ExportJob).where(ExportJob.status == "queued").order_by(ExportJob.created_at_utc))
            .scalars()
            .all()
        )
        queued_job_ids = [job.job_id for job in queued_jobs]
        db.commit()
    finally:
        db.close()

    for job_id in queued_job_ids:
        schedule_export_job(job_id)


def _run_export_job(job_id: str) -> None:
    db = SessionLocal()
    db.expire_on_commit = False
    try:
        job = db.get(ExportJob, job_id)
        if not job or job.status in EXPORT_JOB_TERMINAL_STATUSES:
            return

        now = _utcnow()
        job.status = "running"
        job.phase = "collecting_metadata"
        job.started_at_utc = job.started_at_utc or now
        job.updated_at_utc = now
        job.progress_percent = max(job.progress_percent, 1.0)
        db.commit()

        def progress(payload: dict[str, object]) -> None:
            job = db.get(ExportJob, job_id)
            if not job:
                return
            for key in (
                "phase",
                "total_items",
                "processed_items",
                "progress_percent",
                "current_item",
                "warning_count",
            ):
                if key in payload:
                    setattr(job, key, payload[key])
            job.updated_at_utc = _utcnow()
            db.commit()

        result = create_export_zip(db, progress=progress)
        file_size = result.zip_path.stat().st_size if result.zip_path.exists() else None
        finished_at = _utcnow()
        status = "completed_with_warnings" if result.warning_count else "completed"
        job = db.get(ExportJob, job_id)
        if job:
            job.status = status
            job.phase = "completed"
            job.progress_percent = 100.0
            job.current_item = result.file_name
            job.file_name = result.file_name
            job.local_file_path = f"exports/{result.file_name}"
            job.file_size_bytes = file_size
            job.warning_count = result.warning_count
            job.error_message = None
            job.updated_at_utc = finished_at
            job.completed_at_utc = finished_at
            db.commit()
    except Exception as exc:
        failed_at = _utcnow()
        job = db.get(ExportJob, job_id)
        if job:
            job.status = "failed"
            job.phase = "failed"
            job.progress_percent = min(job.progress_percent, 99.0)
            job.error_message = str(exc)[:1000] or exc.__class__.__name__
            job.updated_at_utc = failed_at
            job.completed_at_utc = failed_at
            db.commit()
    finally:
        db.close()
        with _schedule_lock:
            _scheduled_job_ids.discard(job_id)
