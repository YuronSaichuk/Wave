"""FastAPI endpoints for internal wave-core service."""

import logging
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from pydantic import BaseModel

import wave_core
from wave_core.analysis.normalization import normalize_library_features
from wave_core.analysis.pipeline import analyze_and_store_track
from wave_core.storage.db import check_db, get_connection
from wave_core.storage.jobs import (
    JobStatus,
    create_job,
    get_job_by_id,
    update_job,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wave Core API",
    version=wave_core.__version__,
    description="Internal audio analysis and processing API for Wave",
)


class HealthResponse(BaseModel):
    status: str
    version: str
    db: str


class AnalyzeJobRequest(BaseModel):
    file_path: str
    force: bool = False


class AnalyzeJobResponse(BaseModel):
    job_id: UUID
    status: str


class JobResponse(BaseModel):
    id: UUID
    status: str
    progress: float
    error: str | None = None


def _update_job_status(
    job_id: UUID,
    status: str,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    """Safely commit job state changes to the database."""
    try:
        with get_connection() as conn, conn.transaction():
            update_job(
                conn,
                job_id=job_id,
                status=status,
                progress=progress,
                error=error,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to update status for job %s: %s", job_id, exc)


def _process_analyze_job(job_id: UUID, file_path: str, force: bool = False) -> None:
    """Background worker for executing track analysis."""
    try:
        _update_job_status(job_id, status=JobStatus.RUNNING, progress=0.05)
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        def on_progress(p: float, msg: str = "") -> None:
            _update_job_status(job_id, status=JobStatus.RUNNING, progress=p)

        analyze_and_store_track(
            file_path=path,
            force=force,
            progress_callback=on_progress,
        )

        # Trigger auto-normalization check if needed
        try:
            with get_connection() as conn, conn.transaction():
                normalize_library_features(conn=conn, force=False)
        except Exception as norm_err:  # noqa: BLE001
            logger.warning("Auto-normalization check failed after job %s: %s", job_id, norm_err)

        _update_job_status(job_id, status=JobStatus.DONE, progress=1.0)
    except Exception as exc:
        logger.exception("Analysis job %s failed", job_id)
        _update_job_status(job_id, status=JobStatus.FAILED, error=str(exc))


@app.get("/health", response_model=HealthResponse)
def get_health():
    """Health check returning service and DB status."""
    db_status = "ok"
    try:
        check_db()
    except Exception as exc:  # noqa: BLE001
        db_status = f"unreachable ({exc})"

    return HealthResponse(
        status="ok",
        version=wave_core.__version__,
        db=db_status,
    )


@app.post(
    "/analyze",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AnalyzeJobResponse,
)
def create_analyze_job(
    request: AnalyzeJobRequest,
    background_tasks: BackgroundTasks,
):
    """Enqueue an audio file analysis job returning 202 Accepted with job_id."""
    resolved_path = str(Path(request.file_path).resolve())
    with get_connection() as conn, conn.transaction():
        job = create_job(
            conn,
            kind="analyze",
            payload={"file_path": resolved_path, "force": request.force},
        )

    background_tasks.add_task(
        _process_analyze_job,
        job_id=job.id,
        file_path=resolved_path,
        force=request.force,
    )

    return AnalyzeJobResponse(
        job_id=job.id,
        status=job.status,
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: UUID):
    """Get status of an asynchronous job (analyze, render_mix, render_lights)."""
    with get_connection() as conn:
        job = get_job_by_id(conn, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return JobResponse(
            id=job.id,
            status=job.status,
            progress=job.progress,
            error=job.error,
        )
