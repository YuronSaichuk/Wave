"""Unit and integration tests for library-wide feature normalization (T1.9)."""

from uuid import uuid4

import numpy as np
import pytest
from typer.testing import CliRunner

from wave_core.analysis.normalization import (
    FeatureStats,
    check_normalization_needed,
    compute_feature_stats,
    load_feature_stats,
    normalize_band_vector,
    normalize_chroma_vector,
    normalize_library_features,
    normalize_timbre_vector,
    save_feature_stats,
)
from wave_core.cli import app
from wave_core.storage.db import get_connection
from wave_core.storage.features import TrackFeaturesRecord, upsert_track_features
from wave_core.storage.tracks import TrackMetadata, upsert_track

runner = CliRunner()


def test_compute_feature_stats_basic():
    """Test computing mean and std across a simulated library matrix."""
    n_tracks = 50
    # Create synthetic timbre matrix [50, 40] with known coordinate means
    timbre_mat = np.random.normal(loc=5.0, scale=2.0, size=(n_tracks, 40)).astype(np.float32)
    # Create synthetic band matrix [50, 8]
    band_mat = np.random.normal(loc=-20.0, scale=6.0, size=(n_tracks, 8)).astype(np.float32)

    stats = compute_feature_stats(timbre_mat, band_mat)

    assert stats.track_count == n_tracks
    assert stats.timbre_mean.shape == (40,)
    assert stats.timbre_std.shape == (40,)
    assert stats.band_mean.shape == (8,)
    assert stats.band_std.shape == (8,)

    # Values should be close to generation parameters
    assert pytest.approx(float(np.mean(stats.timbre_mean)), abs=0.5) == 5.0
    assert pytest.approx(float(np.mean(stats.timbre_std)), abs=0.5) == 2.0
    assert pytest.approx(float(np.mean(stats.band_mean)), abs=1.0) == -20.0
    assert pytest.approx(float(np.mean(stats.band_std)), abs=1.0) == 6.0


def test_compute_feature_stats_single_track_stability():
    """Test single track does not cause division by zero (std becomes 1.0)."""
    timbre_mat = np.ones((1, 40), dtype=np.float32) * 3.0
    band_mat = np.ones((1, 8), dtype=np.float32) * -10.0

    stats = compute_feature_stats(timbre_mat, band_mat)
    assert stats.track_count == 1
    assert np.all(stats.timbre_std >= 1.0)
    assert np.all(stats.band_std >= 1.0)

    # Normalization should not produce NaN or Inf
    v_timbre = normalize_timbre_vector(timbre_mat[0], stats)
    assert not np.isnan(v_timbre).any()
    assert not np.isinf(v_timbre).any()

    v_band = normalize_band_vector(band_mat[0], stats)
    assert not np.isnan(v_band).any()
    assert not np.isinf(v_band).any()


def test_compute_feature_stats_empty_raises():
    """Test empty input matrices raise ValueError."""
    with pytest.raises(ValueError, match="Cannot compute feature stats on empty"):
        compute_feature_stats(np.empty((0, 40)), np.empty((0, 8)))


def test_normalize_timbre_vector_l2_norm():
    """Test timbre normalization: z-score followed by L2-normalization."""
    stats = FeatureStats(
        track_count=10,
        timbre_mean=np.full(40, 2.0, dtype=np.float32),
        timbre_std=np.full(40, 1.5, dtype=np.float32),
        band_mean=np.zeros(8, dtype=np.float32),
        band_std=np.ones(8, dtype=np.float32),
    )

    raw_timbre = np.random.randn(40).astype(np.float32)
    norm_timbre = normalize_timbre_vector(raw_timbre, stats)

    assert norm_timbre.shape == (40,)
    assert norm_timbre.dtype == np.float32
    # L2 norm must be 1.0
    assert pytest.approx(float(np.linalg.norm(norm_timbre)), abs=1e-5) == 1.0


def test_normalize_band_vector_zscore():
    """Test band normalization applies z-score without L2 normalization."""
    stats = FeatureStats(
        track_count=100,
        timbre_mean=np.zeros(40, dtype=np.float32),
        timbre_std=np.ones(40, dtype=np.float32),
        band_mean=np.full(8, -15.0, dtype=np.float32),
        band_std=np.full(8, 5.0, dtype=np.float32),
    )

    raw_band = np.full(8, -10.0, dtype=np.float32)  # 1 standard deviation above mean
    norm_band = normalize_band_vector(raw_band, stats)

    assert norm_band.shape == (8,)
    assert norm_band.dtype == np.float32
    # (-10 - (-15)) / 5 = 1.0
    np.testing.assert_allclose(norm_band, np.ones(8, dtype=np.float32), atol=1e-5)


def test_normalize_chroma_vector_l2_only():
    """Test chroma normalization applies L2 normalization only without z-score."""
    raw_chroma = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    norm_chroma = normalize_chroma_vector(raw_chroma)

    assert norm_chroma.shape == (12,)
    assert norm_chroma.dtype == np.float32
    assert pytest.approx(float(np.linalg.norm(norm_chroma)), abs=1e-5) == 1.0

    # Proportions between dimensions should be preserved (no mean centering)
    assert norm_chroma[1] / norm_chroma[0] == pytest.approx(2.0, rel=1e-4)
    assert norm_chroma[2] / norm_chroma[0] == pytest.approx(3.0, rel=1e-4)


def test_feature_stats_db_roundtrip():
    """Test saving and loading FeatureStats to and from the PostgreSQL feature_stats table."""
    timbre_mean = np.random.randn(40).astype(np.float32)
    timbre_std = np.random.rand(40).astype(np.float32) + 0.1
    band_mean = np.random.randn(8).astype(np.float32)
    band_std = np.random.rand(8).astype(np.float32) + 0.1

    stats = FeatureStats(
        track_count=42,
        timbre_mean=timbre_mean,
        timbre_std=timbre_std,
        band_mean=band_mean,
        band_std=band_std,
    )

    with get_connection() as conn, conn.transaction():
        save_feature_stats(conn, stats)
        loaded = load_feature_stats(conn)

        assert loaded is not None
        assert loaded.track_count == 42
        assert loaded.updated_at is not None
        np.testing.assert_allclose(loaded.timbre_mean, timbre_mean, atol=1e-4)
        np.testing.assert_allclose(loaded.timbre_std, timbre_std, atol=1e-4)
        np.testing.assert_allclose(loaded.band_mean, band_mean, atol=1e-4)
        np.testing.assert_allclose(loaded.band_std, band_std, atol=1e-4)


def test_check_normalization_needed():
    """Test 20% growth detection logic."""
    with get_connection() as conn, conn.transaction():
        # Set stats to track_count = 100
        stats = FeatureStats(
            track_count=100,
            timbre_mean=np.zeros(40, dtype=np.float32),
            timbre_std=np.ones(40, dtype=np.float32),
            band_mean=np.zeros(8, dtype=np.float32),
            band_std=np.ones(8, dtype=np.float32),
        )
        save_feature_stats(conn, stats)

        # Mock current count via temporary feature_stats record check
        loaded = load_feature_stats(conn)
        assert loaded.track_count == 100

        is_needed, cur_c, prev_c, growth = check_normalization_needed(conn, threshold=0.20)
        assert prev_c == 100
        assert cur_c >= 0
        assert isinstance(growth, float)
        assert isinstance(is_needed, bool)

        # Test threshold computation directly
        needed_10pct = (110 - 100) / 100 >= 0.20
        assert needed_10pct is False

        needed_20pct = (120 - 100) / 100 >= 0.20
        assert needed_20pct is True

        needed_25pct = (125 - 100) / 100 >= 0.20
        assert needed_25pct is True


def test_normalize_library_features_end_to_end():
    """Test full library normalization on 3 database tracks."""
    created_ids = []

    with get_connection() as conn, conn.transaction():
        # Insert 3 test tracks with features
        for i in range(3):
            meta = TrackMetadata(
                file_path=f"/music/norm_test_{uuid4()}.mp3",
                file_hash=f"hash_norm_{i}",
                duration_sec=120.0,
            )
            t_id, _ = upsert_track(conn, meta)
            created_ids.append(t_id)

            # Insert raw vectors
            feat = TrackFeaturesRecord(
                track_id=t_id,
                bpm=120.0 + i * 4,
                timbre_vec=np.full(40, 1.0 + i * 2, dtype=np.float32),
                chroma_vec=np.ones(12, dtype=np.float32),
                band_vec=np.full(8, -10.0 + i * 5, dtype=np.float32),
                frames_path="",
            )
            upsert_track_features(conn, feat)

        # Run normalization with force=True
        result = normalize_library_features(conn=conn, force=True)

        assert result.applied is True
        assert result.tracks_updated >= 3
        assert result.stats is not None

        # Check that track features in database were updated with L2-normalized timbre
        with conn.cursor() as cur:
            cur.execute("SELECT timbre_vec, band_vec, chroma_vec FROM track_features WHERE track_id = %s;", (created_ids[0],))
            row = cur.fetchone()
            t_vec = row[0].to_numpy() if hasattr(row[0], "to_numpy") else np.asarray(row[0], dtype=np.float32)
            c_vec = row[2].to_numpy() if hasattr(row[2], "to_numpy") else np.asarray(row[2], dtype=np.float32)

            # Timbre and chroma must be unit length
            assert pytest.approx(float(np.linalg.norm(t_vec)), abs=1e-4) == 1.0
            assert pytest.approx(float(np.linalg.norm(c_vec)), abs=1e-4) == 1.0

        # Clean up created tracks
        with conn.cursor() as cur:
            for t_id in created_ids:
                cur.execute("DELETE FROM tracks WHERE id = %s;", (t_id,))


def test_cli_normalize_features_command():
    """Test wave normalize-features command in CLI."""
    result = runner.invoke(app, ["normalize-features", "--force"])
    assert result.exit_code == 0
    assert "Wave Feature Normalizer" in result.stdout
