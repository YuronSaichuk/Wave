"""Unit and integration tests for batch orchestration and multiprocessing execution (T1.10)."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from wave_core.analysis.batch import (
    BatchSummary,
    _worker_process_track,
    analyze_batch,
)
from wave_core.analysis.spectral import SAMPLE_RATE
from wave_core.cli import app
from wave_core.storage.cache import load_feature_cache
from wave_core.storage.db import get_connection
from wave_core.storage.features import get_track_features
from wave_core.storage.tracks import get_track_by_path

runner = CliRunner()


def _create_synthetic_wav(path: Path, duration_sec: float = 2.0, freq: float = 440.0) -> None:
    """Helper to synthesize a test WAV file with a pure sine wave and clicks."""
    sr = SAMPLE_RATE
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = 0.6 * np.sin(2 * np.pi * freq * t)

    # Add clicks every 0.5s (120 BPM)
    beat_indices = (np.arange(0, duration_sec, 0.5) * sr).astype(int)
    for b in beat_indices:
        if b + 80 < len(audio):
            audio[b : b + 80] += 0.8

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr)


def test_worker_process_track_success(tmp_path: Path):
    """Test worker function processes audio, saves .npz, and returns dictionary."""
    wav_path = tmp_path / "worker_test.wav"
    _create_synthetic_wav(wav_path, duration_sec=2.0)
    cache_path = tmp_path / "cache" / "worker_test.npz"

    dummy_id_str = str(uuid4())
    task_args = (dummy_id_str, str(wav_path), str(cache_path))

    t_id_str, f_path_str, success, error, record_dict, duration, compute_time = _worker_process_track(task_args)

    assert success is True
    assert error is None
    assert t_id_str == dummy_id_str
    assert f_path_str == str(wav_path)
    assert duration > 1.9
    assert compute_time > 0.0

    # Cache file must exist and contain valid arrays
    assert cache_path.exists()
    cache_data = load_feature_cache(cache_path)
    assert "band_energy" in cache_data
    assert "beats" in cache_data

    # Features dictionary must be populated
    assert record_dict is not None
    assert record_dict["bpm"] > 0
    assert len(record_dict["timbre_vec"]) == 40
    assert len(record_dict["band_vec"]) == 8


def test_worker_process_track_error_isolation(tmp_path: Path):
    """Test worker function isolates errors without raising unhandled exceptions."""
    non_existent = tmp_path / "missing.wav"
    cache_path = tmp_path / "cache" / "missing.npz"

    task_args = (str(uuid4()), str(non_existent), str(cache_path))
    _, _, success, error, record_dict, duration, _ = _worker_process_track(task_args)

    assert success is False
    assert error is not None
    assert "not found" in error.lower() or "no such file" in error.lower()
    assert record_dict is None
    assert duration == 0.0


def test_analyze_batch_sequential_and_idempotent(tmp_path: Path):
    """Test sequential batch analysis (workers=1), database persistence, and idempotency."""
    dir_path = tmp_path / "music"
    dir_path.mkdir()

    f1 = dir_path / "track_01.wav"
    f2 = dir_path / "track_02.wav"
    _create_synthetic_wav(f1, duration_sec=2.0, freq=440.0)
    _create_synthetic_wav(f2, duration_sec=2.5, freq=550.0)

    cache_dir = tmp_path / "cache"

    with get_connection() as conn, conn.transaction():
        # First run: analyzes both tracks
        summary1 = analyze_batch(
            target_path=dir_path,
            force=False,
            workers=1,
            cache_dir=cache_dir,
            conn=conn,
        )

        assert isinstance(summary1, BatchSummary)
        assert summary1.total_found == 2
        assert summary1.analyzed_count == 2
        assert summary1.skipped_count == 0
        assert summary1.failed_count == 0

        # Verify tracks and features exist in database
        t1 = get_track_by_path(conn, str(f1))
        t2 = get_track_by_path(conn, str(f2))
        assert t1 is not None and t2 is not None
        assert t1.analyzed_at is not None
        assert t2.analyzed_at is not None

        feat1 = get_track_features(conn, t1.id)
        feat2 = get_track_features(conn, t2.id)
        assert feat1 is not None and feat2 is not None
        assert feat1.bpm > 0 and feat2.bpm > 0

        # Second run with force=False: should skip both tracks
        summary2 = analyze_batch(
            target_path=dir_path,
            force=False,
            workers=1,
            cache_dir=cache_dir,
            conn=conn,
        )

        assert summary2.total_found == 2
        assert summary2.analyzed_count == 0
        assert summary2.skipped_count == 2
        assert summary2.failed_count == 0

        # Third run with force=True: should re-analyze both tracks
        summary3 = analyze_batch(
            target_path=dir_path,
            force=True,
            workers=1,
            cache_dir=cache_dir,
            conn=conn,
        )

        assert summary3.total_found == 2
        assert summary3.analyzed_count == 2
        assert summary3.skipped_count == 0

        # Cleanup
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tracks WHERE id IN (%s, %s);", (t1.id, t2.id))


def test_analyze_batch_error_resilience(tmp_path: Path):
    """Test corrupted files do not abort the entire batch analysis."""
    dir_path = tmp_path / "resilience"
    dir_path.mkdir()

    valid_wav = dir_path / "valid.wav"
    corrupt_wav = dir_path / "corrupted.wav"

    _create_synthetic_wav(valid_wav, duration_sec=2.0)
    # Write garbage bytes that fail audio decoding
    corrupt_wav.write_bytes(b"NOT_A_VALID_AUDIO_HEADER" * 50)

    cache_dir = tmp_path / "cache"

    with get_connection() as conn, conn.transaction():
        summary = analyze_batch(
            target_path=dir_path,
            force=True,
            workers=1,
            cache_dir=cache_dir,
            conn=conn,
        )

        assert summary.total_found == 2
        assert summary.analyzed_count == 1
        assert summary.failed_count == 1
        assert len(summary.failed_items) == 1
        assert "corrupted.wav" in summary.failed_items[0][0]

        # Verify valid track exists in DB
        t_valid = get_track_by_path(conn, str(valid_wav))
        assert t_valid is not None
        assert get_track_features(conn, t_valid.id) is not None

        # Cleanup
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tracks WHERE file_path LIKE %s;", (f"%{tmp_path.name}%",))


def test_analyze_batch_single_file(tmp_path: Path):
    """Test analyzing a single file directly rather than a directory."""
    wav_path = tmp_path / "single_file.wav"
    _create_synthetic_wav(wav_path, duration_sec=2.0)

    with get_connection() as conn, conn.transaction():
        summary = analyze_batch(
            target_path=wav_path,
            force=True,
            workers=1,
            conn=conn,
        )

        assert summary.total_found == 1
        assert summary.analyzed_count == 1
        assert summary.failed_count == 0

        # Cleanup
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tracks WHERE file_path = %s;", (str(wav_path),))


def test_cli_analyze_command(tmp_path: Path):
    """Test wave analyze CLI command."""
    wav_path = tmp_path / "cli_sample.wav"
    _create_synthetic_wav(wav_path, duration_sec=2.0)

    result = runner.invoke(app, ["analyze", str(wav_path), "--force", "--workers", "1"])
    assert result.exit_code == 0
    assert "Wave Audio Analysis Engine" in result.stdout
    assert "Batch Analysis Summary" in result.stdout
    assert "Successfully Analyzed" in result.stdout

    # Clean up test track from database
    with get_connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM tracks WHERE file_path = %s;", (str(wav_path),))
