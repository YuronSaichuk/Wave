"""Database access and models for background job queue (jobs table)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class JobStatus:
    """Canonical lifecycle states for asynchronous tasks."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class JobRecord:
    """Record model for the jobs database table."""

    id: UUID
    kind: str  # analyze | render_mix | render_lights
    payload: dict[str, Any]
    status: str  # queued | running | done | failed
    progress: float
    error: str | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None


def create_job(
    conn: psycopg.Connection,
    kind: str,
    payload: dict[str, Any],
    job_id: UUID | None = None,
) -> JobRecord:
    """Insert a new job into the queue with 'queued' status.

    Args:
        conn: Open database connection.
        kind: Task category (e.g. 'analyze').
        payload: Metadata dictionary (e.g. {'file_path': '...'}).
        job_id: Optional UUID. Generated automatically if omitted.

    Returns:
        Newly created JobRecord.
    """
    new_id = job_id or uuid4()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO jobs (id, kind, payload, status, progress, created_at)
            VALUES (%s, %s, %s, %s, %s, now())
            RETURNING id, kind, payload, status, progress, error, created_at, finished_at;
            """,
            (new_id, kind, Jsonb(payload), JobStatus.QUEUED, 0.0),
        )
        row = cur.fetchone()
        return JobRecord(**row)


def get_job_by_id(conn: psycopg.Connection, job_id: UUID) -> JobRecord | None:
    """Retrieve a job by its UUID.

    Args:
        conn: Open database connection.
        job_id: Job identifier UUID.

    Returns:
        JobRecord instance or None if not found.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, kind, payload, status, progress, error, created_at, finished_at
            FROM jobs
            WHERE id = %s;
            """,
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return JobRecord(**row)


def update_job(
    conn: psycopg.Connection,
    job_id: UUID,
    status: str,
    progress: float | None = None,
    error: str | None = None,
    payload: dict[str, Any] | None = None,
) -> JobRecord | None:
    """Update job status, progress, and error message.

    If status is 'done' or 'failed', automatically records finished_at timestamp.

    Args:
        conn: Open database connection.
        job_id: Job identifier UUID.
        status: New JobStatus.
        progress: Optional completion ratio in [0.0, 1.0].
        error: Optional failure message string.
        payload: Optional updated payload metadata.

    Returns:
        Updated JobRecord or None if not found.
    """
    is_finished = status in (JobStatus.DONE, JobStatus.FAILED)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status = %s,
                progress = COALESCE(%s, progress),
                error = %s,
                payload = COALESCE(%s, payload),
                finished_at = CASE WHEN %s THEN now() ELSE finished_at END
            WHERE id = %s
            RETURNING id, kind, payload, status, progress, error, created_at, finished_at;
            """,
            (
                status,
                progress,
                error,
                Jsonb(payload) if payload is not None else None,
                is_finished,
                job_id,
            ),
        )
        row = cur.fetchone()
        if not row:
            return None
        return JobRecord(**row)


def delete_job(conn: psycopg.Connection, job_id: UUID) -> bool:
    """Delete a job record from the database.

    Args:
        conn: Open database connection.
        job_id: Job identifier UUID.

    Returns:
        True if record was deleted, False if not found.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM jobs WHERE id = %s RETURNING id;", (job_id,))
        return cur.fetchone() is not None
