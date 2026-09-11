from uuid import UUID
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import wave_core
from wave_core.storage.db import check_db

app = FastAPI(
    title="Wave Core API",
    version=wave_core.__version__,
    description="Internal audio analysis and processing API for Wave",
)


class HealthResponse(BaseModel):
    status: str
    version: str
    db: str


class JobResponse(BaseModel):
    id: UUID
    status: str
    progress: float
    error: str | None = None


@app.get("/health", response_model=HealthResponse)
def get_health():
    """Health check returning service and DB status."""
    db_status = "ok"
    try:
        check_db()
    except Exception as exc:
        db_status = f"unreachable ({exc})"

    return HealthResponse(
        status="ok",
        version=wave_core.__version__,
        db=db_status,
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: UUID):
    """Get status of an asynchronous job (analyze, render_mix, render_lights)."""
    # Placeholder stub for M0
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
