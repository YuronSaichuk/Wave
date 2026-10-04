"""Tests for feature caching (.npz) and database storage (track_features table)."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf

from wave_core.analysis.pipeline import (
    FullAnalysisResult,
    analyze_and_store_track,
    analyze_track_audio,
    store_track_analysis,
)
from wave_core.analysis.spectral import HOP_LENGTH, SAMPLE_RATE
from wave_core.storage.cache import (
    CACHE_KEYS,
    MAX_CACHE_SIZE_BYTES_4MIN,
    get_cache_path,
    get_cache_size_bytes,
    load_feature_cache,
    save_feature_cache,
)
from wave_core.storage.db import get_connection
from wave_core.storage.features import (
    ANALYSIS_VERSION,
    TrackFeaturesRecord,
    count_analyzed_tracks,
    delete_track_features,
    get_track_features,
    upsert_track_features,
)
from wave_core.storage.tracks import TrackMetadata, upsert_track


def test_save_and_load_feature_cache(tmp_path: Path):
    """Test saving and loading compressed .npz feature cache."""
    n_frames = 500
    cache_file = tmp_path / "test_track.npz"

    band_energy = np.random.randn(8, n_frames).astype(np.float32)
    rms = np.random.rand(n_frames).astype(np.float32)
    onset_env = np.random.rand(n_frames).astype(np.float32)
    beats = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    downbeats = np.array([0.5, 2.5], dtype=np.float32)
    chroma = np.random.rand(12, 60).astype(np.float32)
    mfcc = np.random.randn(20, 60).astype(np.float32)

    saved_path = save_feature_cache(
        path=cache_file,
        band_energy=band_energy,
        rms=rms,
        onset_env=onset_env,
        beats=beats,
        downbeats=downbeats,
        chroma=chroma,
        mfcc=mfcc,
        downsample_features_if_needed=False,
    )
    assert saved_path.exists()
    assert saved_path == cache_file

    loaded = load_feature_cache(cache_file)
    for key in CACHE_KEYS:
        assert key in loaded
        assert loaded[key].dtype == np.float32

    np.testing.assert_allclose(loaded["band_energy"], band_energy, rtol=1e-5)
    np.testing.assert_allclose(loaded["rms"], rms, rtol=1e-5)
    np.testing.assert_allclose(loaded["onset_env"], onset_env, rtol=1e-5)
    np.testing.assert_allclose(loaded["beats"], beats, rtol=1e-5)
    np.testing.assert_allclose(loaded["downbeats"], downbeats, rtol=1e-5)
    np.testing.assert_allclose(loaded["chroma"], chroma, rtol=1e-5)
    np.testing.assert_allclose(loaded["mfcc"], mfcc, rtol=1e-5)


def test_cache_automatic_downsampling(tmp_path: Path):
    """Test that full-rate chroma and mfcc are downsampled to ~5 Hz before saving."""
    n_frames = 800  # full frame rate
    cache_file = tmp_path / "ds_track.npz"

    band_energy = np.zeros((8, n_frames), dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)
    onset_env = np.zeros(n_frames, dtype=np.float32)
    beats = np.array([1.0], dtype=np.float32)
    downbeats = np.array([1.0], dtype=np.float32)
    chroma_full = np.ones((12, n_frames), dtype=np.float32)
    mfcc_full = np.ones((20, n_frames), dtype=np.float32)

    save_feature_cache(
        path=cache_file,
        band_energy=band_energy,
        rms=rms,
        onset_env=onset_env,
        beats=beats,
        downbeats=downbeats,
        chroma=chroma_full,
        mfcc=mfcc_full,
        downsample_features_if_needed=True,
    )

    loaded = load_feature_cache(cache_file)
    assert loaded["chroma"].shape[0] == 12
    assert loaded["chroma"].shape[1] < n_frames
    assert loaded["mfcc"].shape[0] == 20
    assert loaded["mfcc"].shape[1] < n_frames


def test_cache_size_4min_track_under_2mb(tmp_path: Path):
    """Test that compressed .npz size for a 4-minute audio track is <= 2 MB."""
    duration_sec = 240.0  # 4 minutes
    fps = SAMPLE_RATE / HOP_LENGTH
    n_frames = int(duration_sec * fps)  # ~10,336 frames
    n_beats = int(duration_sec * 2)  # ~120 BPM -> 480 beats

    band_energy = np.random.randn(8, n_frames).astype(np.float32)
    rms = np.random.rand(n_frames).astype(np.float32)
    onset_env = np.random.rand(n_frames).astype(np.float32)
    beats = np.linspace(0.5, duration_sec - 0.5, n_beats, dtype=np.float32)
    downbeats = beats[::4]

    chroma_full = np.random.rand(12, n_frames).astype(np.float32)
    mfcc_full = np.random.randn(20, n_frames).astype(np.float32)

    cache_file = tmp_path / "4min_track.npz"
    save_feature_cache(
        path=cache_file,
        band_energy=band_energy,
        rms=rms,
        onset_env=onset_env,
        beats=beats,
        downbeats=downbeats,
        chroma=chroma_full,
        mfcc=mfcc_full,
        downsample_features_if_needed=True,
    )

    size_bytes = get_cache_size_bytes(cache_file)
    assert size_bytes <= MAX_CACHE_SIZE_BYTES_4MIN
    assert size_bytes < 1 * 1024 * 1024


def test_get_cache_path():
    """Test resolving cache path for a UUID."""
    tid = uuid4()
    p = get_cache_path(tid, cache_dir="/tmp/test_cache")
    assert p.name == f"{tid}.npz"


def test_track_features_db_lifecycle():
    """Test upsert, retrieval, count, and deletion of TrackFeaturesRecord in Postgres."""
    test_track_id = uuid4()
    meta = TrackMetadata(
        file_path=f"/music/test_feat_{test_track_id}.mp3",
        file_hash="hash_features_test",
        title="Features Test Track",
        artist="Test Artist",
        album="Test Album",
        duration_sec=120.0,
        sample_rate=22050,
    )

    with get_connection() as conn, conn.transaction():
        # Insert track first
        track_id, _ = upsert_track(conn, meta)

        # Create features record
        timbre = np.random.randn(40).astype(np.float32)
        chroma = np.random.rand(12).astype(np.float32)
        chroma /= np.linalg.norm(chroma)
        band = np.random.randn(8).astype(np.float32)

        sections = [
            {"start": 0.0, "end": 30.0, "label": "intro", "energy": 0.4},
            {"start": 30.0, "end": 90.0, "label": "drop", "energy": 1.0},
            {"start": 90.0, "end": 120.0, "label": "outro", "energy": 0.3},
        ]

        features = TrackFeaturesRecord(
            track_id=track_id,
            bpm=128.0,
            bpm_confidence=0.95,
            beat_grid_path=f"data/cache/{track_id}.npz",
            first_beat_sec=0.25,
            key_pitch=0,
            key_mode=1,
            key_confidence=0.88,
            camelot="8B",
            lufs_integrated=-14.2,
            lufs_range=6.5,
            true_peak_db=-0.8,
            centroid_mean=2100.5,
            centroid_std=450.2,
            rolloff_mean=4200.0,
            rolloff_std=780.0,
            flatness_mean=0.015,
            flatness_std=0.008,
            bandwidth_mean=1800.0,
            bandwidth_std=320.0,
            zcr_mean=0.045,
            zcr_std=0.012,
            timbre_vec=timbre,
            chroma_vec=chroma,
            band_vec=band,
            frames_path=f"data/cache/{track_id}.npz",
            sections=sections,
        )

        initial_count = count_analyzed_tracks(conn)

        # 1. Upsert features
        upsert_track_features(conn, features)
        assert count_analyzed_tracks(conn) == initial_count + 1

        # Check tracks table was updated with analyzed_at and analysis_version
        with conn.cursor() as cur:
            cur.execute("SELECT analyzed_at, analysis_version FROM tracks WHERE id = %s;", (track_id,))
            row = cur.fetchone()
            assert row[0] is not None
            assert row[1] == ANALYSIS_VERSION

        # 2. Retrieve features
        retrieved = get_track_features(conn, track_id)
        assert retrieved is not None
        assert retrieved.track_id == track_id
        assert pytest.approx(retrieved.bpm, 0.01) == 128.0
        assert retrieved.camelot == "8B"
        assert retrieved.key_pitch == 0
        assert retrieved.key_mode == 1
        assert pytest.approx(retrieved.lufs_integrated, 0.01) == -14.2
        assert retrieved.sections == sections

        # Check vectors
        assert retrieved.timbre_vec is not None
        assert len(retrieved.timbre_vec) == 40
        np.testing.assert_allclose(retrieved.timbre_vec, timbre, atol=1e-5)

        assert retrieved.chroma_vec is not None
        assert len(retrieved.chroma_vec) == 12
        np.testing.assert_allclose(retrieved.chroma_vec, chroma, atol=1e-5)

        assert retrieved.band_vec is not None
        assert len(retrieved.band_vec) == 8
        np.testing.assert_allclose(retrieved.band_vec, band, atol=1e-5)

        # 3. Update features (re-upsert)
        features.bpm = 130.0
        features.camelot = "9B"
        upsert_track_features(conn, features)
        updated = get_track_features(conn, track_id)
        assert updated is not None
        assert pytest.approx(updated.bpm, 0.01) == 130.0
        assert updated.camelot == "9B"

        # 4. Delete track features
        deleted = delete_track_features(conn, track_id)
        assert deleted is True
        assert get_track_features(conn, track_id) is None

        with conn.cursor() as cur:
            cur.execute("SELECT analyzed_at, analysis_version FROM tracks WHERE id = %s;", (track_id,))
            row = cur.fetchone()
            assert row[0] is None
            assert row[1] is None

        # Clean up test track
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tracks WHERE id = %s;", (track_id,))


def test_analyze_and_store_end_to_end(tmp_path: Path):
    """Test full analysis and storage pipeline on a synthesized audio file."""
    sr = SAMPLE_RATE
    duration_sec = 4.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    beat_samples = (np.arange(0, duration_sec, 0.5) * sr).astype(int)
    for bs in beat_samples:
        if bs + 100 < len(audio):
            audio[bs : bs + 100] += 0.8

    # Also verify direct analyze_track_audio
    res = analyze_track_audio(audio, sr=sr)
    assert isinstance(res, FullAnalysisResult)
    assert len(res.timbre_vec) == 40
    assert len(res.chroma_vec) == 12
    assert len(res.band_vec) == 8

    audio_file = tmp_path / "synthetic_dance_120bpm.wav"
    sf.write(str(audio_file), audio.astype(np.float32), sr)

    cache_dir = tmp_path / "cache"

    with get_connection() as conn, conn.transaction():
        # Test store_track_analysis directly with a valid track
        dummy_meta = TrackMetadata(
            file_path=f"/dummy/{uuid4()}.wav",
            file_hash="dummy_hash",
            duration_sec=duration_sec,
        )
        dummy_id, _ = upsert_track(conn, dummy_meta)
        c_path, rec = store_track_analysis(conn, dummy_id, res, cache_dir=cache_dir)
        assert c_path.exists()
        assert rec.track_id == dummy_id
        # Clean up dummy track (cascades to track_features)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tracks WHERE id = %s;", (dummy_id,))

        cache_path, record = analyze_and_store_track(
            file_path=audio_file,
            conn=conn,
            cache_dir=cache_dir,
            force=True,
        )

        assert cache_path.exists()
        assert cache_path.name.endswith(".npz")

        cache_data = load_feature_cache(cache_path)
        assert len(cache_data["beats"]) > 0
        assert cache_data["band_energy"].shape[0] == 8
        assert cache_data["chroma"].shape[0] == 12
        assert cache_data["mfcc"].shape[0] == 20

        assert record is not None
        assert record.track_id is not None
        assert record.bpm > 0
        assert record.camelot is not None
        assert len(record.timbre_vec) == 40
        assert len(record.chroma_vec) == 12
        assert len(record.band_vec) == 8
        assert len(record.sections) > 0

        # Verify idempotency without force
        cached_path_2, record_2 = analyze_and_store_track(
            file_path=audio_file,
            conn=conn,
            cache_dir=cache_dir,
            force=False,
        )
        assert cached_path_2 == cache_path
        assert record_2.track_id == record.track_id

        # Clean up created track
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tracks WHERE id = %s;", (record.track_id,))
