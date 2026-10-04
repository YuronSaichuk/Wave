"""Integration and contract tests for wave-core FastAPI endpoints."""

from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from wave_core.analysis.spectral import SAMPLE_RATE
from wave_core.api import app
from wave_core.storage.db import get_connection
from wave_core.storage.features import get_track_features
from wave_core.storage.jobs import JobStatus, create_job
from wave_core.storage.tracks import get_track_by_path

client = TestClient(app)


@pytest.fixture
def synth_wav(tmp_path: Path) -> Path:
    """Create a temporary 2.0-second synthetic sine wave WAV."""
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False, dtype=np.float32)
    # Sine wave 440 Hz
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    wav_path = tmp_path / "test_api_synth.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    return wav_path


def test_get_health():
    """Verify /health endpoint returns status, version, and database connectivity."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["db"] == "ok"


def test_get_job_not_found():
    """Verify /jobs/{job_id} returns 404 for a non-existent job UUID."""
    random_id = uuid4()
    response = client.get(f"/jobs/{random_id}")
    assert response.status_code == 404
    assert f"Job {random_id} not found" in response.json()["detail"]


def test_get_job_invalid_uuid():
    """Verify /jobs/{job_id} returns 422 for an invalid UUID format."""
    response = client.get("/jobs/not-a-valid-uuid")
    assert response.status_code == 422


def test_get_job_existing():
    """Verify /jobs/{job_id} retrieves a created job record."""
    job_id = uuid4()
    with get_connection() as conn, conn.transaction():
        create_job(
            conn,
            kind="analyze",
            payload={"file_path": "/tmp/test.wav"},
            job_id=job_id,
        )

    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(job_id)
    assert data["status"] == JobStatus.QUEUED
    assert data["progress"] == 0.0
    assert data["error"] is None


def test_post_analyze_missing_body():
    """Verify /analyze returns 422 when required payload fields are missing."""
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_post_analyze_success(synth_wav: Path):
    """Verify POST /analyze enqueues and runs background analysis to completion."""
    response = client.post(
        "/analyze",
        json={"file_path": str(synth_wav), "force": True},
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    job_id = UUID(data["job_id"])
    assert data["status"] == JobStatus.QUEUED

    # Background task runs synchronously in TestClient before post() returns
    # Check job completion status via GET /jobs/{job_id}
    job_resp = client.get(f"/jobs/{job_id}")
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["id"] == str(job_id)
    assert job_data["status"] == JobStatus.DONE
    assert job_data["progress"] == 1.0
    assert job_data["error"] is None

    # Verify track and features were persisted in PostgreSQL
    with get_connection() as conn:
        track = get_track_by_path(conn, str(synth_wav.resolve()))
        assert track is not None
        assert track.analyzed_at is not None

        features = get_track_features(conn, track.id)
        assert features is not None
        assert features.bpm is not None
        assert features.camelot is not None
        assert features.timbre_vec is not None
        assert len(features.timbre_vec) == 40


def test_post_analyze_file_not_found(tmp_path: Path):
    """Verify POST /analyze marks job as failed with error when file does not exist."""
    missing_file = tmp_path / "does_not_exist.wav"
    response = client.post(
        "/analyze",
        json={"file_path": str(missing_file)},
    )
    assert response.status_code == 202
    data = response.json()
    job_id = UUID(data["job_id"])

    # Query job status after background task execution
    job_resp = client.get(f"/jobs/{job_id}")
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["status"] == JobStatus.FAILED
    assert job_data["error"] is not None
    assert "Audio file not found" in job_data["error"]


def test_post_analyze_reanalyze_without_force(synth_wav: Path):
    """Verify re-analyzing an existing track with force=False succeeds quickly."""
    # First analysis
    resp1 = client.post("/analyze", json={"file_path": str(synth_wav), "force": True})
    assert resp1.status_code == 202
    job_id1 = UUID(resp1.json()["job_id"])

    job1_data = client.get(f"/jobs/{job_id1}").json()
    assert job1_data["status"] == JobStatus.DONE

    # Second analysis without force
    resp2 = client.post("/analyze", json={"file_path": str(synth_wav), "force": False})
    assert resp2.status_code == 202
    job_id2 = UUID(resp2.json()["job_id"])
    assert job_id2 != job_id1

    job2_data = client.get(f"/jobs/{job_id2}").json()
    assert job2_data["status"] == JobStatus.DONE
    assert job2_data["progress"] == 1.0
    assert job2_data["error"] is None
