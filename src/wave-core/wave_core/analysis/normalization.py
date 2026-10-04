"""Library-wide feature normalization (z-score and L2 scaling across tracks)."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from wave_core.storage.cache import load_feature_cache
from wave_core.storage.db import get_connection
from wave_core.storage.features import count_analyzed_tracks


@dataclass(slots=True)
class FeatureStats:
    """Library-wide mean and standard deviation statistics for feature vectors."""

    track_count: int
    timbre_mean: np.ndarray  # shape (40,), float32
    timbre_std: np.ndarray  # shape (40,), float32
    band_mean: np.ndarray  # shape (8,), float32
    band_std: np.ndarray  # shape (8,), float32
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to serializable dictionary for JSONB storage."""
        return {
            "track_count": self.track_count,
            "timbre": {
                "mean": [round(float(x), 6) for x in self.timbre_mean],
                "std": [round(float(x), 6) for x in self.timbre_std],
            },
            "band": {
                "mean": [round(float(x), 6) for x in self.band_mean],
                "std": [round(float(x), 6) for x in self.band_std],
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        track_count: int,
        updated_at: datetime | None = None,
    ) -> "FeatureStats":
        """Instantiate FeatureStats from dictionary representation."""
        timbre_data = data.get("timbre", {})
        band_data = data.get("band", {})
        return cls(
            track_count=track_count,
            timbre_mean=np.array(timbre_data.get("mean", []), dtype=np.float32),
            timbre_std=np.array(timbre_data.get("std", []), dtype=np.float32),
            band_mean=np.array(band_data.get("mean", []), dtype=np.float32),
            band_std=np.array(band_data.get("std", []), dtype=np.float32),
            updated_at=updated_at,
        )


@dataclass(slots=True)
class NormalizationResult:
    """Outcome of library feature normalization."""

    tracks_updated: int
    track_count: int
    previous_track_count: int
    growth_ratio: float
    stats: FeatureStats | None
    applied: bool
    reason: str = ""


def compute_feature_stats(timbre_matrix: np.ndarray, band_matrix: np.ndarray) -> FeatureStats:
    """Compute coordinate-wise mean and standard deviation across track feature matrices.

    Args:
        timbre_matrix: 2D array of shape [num_tracks, 40].
        band_matrix: 2D array of shape [num_tracks, 8].

    Returns:
        FeatureStats instance with float32 mean and std vectors.
    """
    n_tracks = len(timbre_matrix)
    if n_tracks == 0 or len(band_matrix) == 0:
        raise ValueError("Cannot compute feature stats on empty library matrices")

    timbre_mat = np.asarray(timbre_matrix, dtype=np.float32)
    band_mat = np.asarray(band_matrix, dtype=np.float32)

    timbre_mean = np.mean(timbre_mat, axis=0).astype(np.float32)
    timbre_std = np.std(timbre_mat, axis=0).astype(np.float32)
    band_mean = np.mean(band_mat, axis=0).astype(np.float32)
    band_std = np.std(band_mat, axis=0).astype(np.float32)

    # Prevent division by zero if standard deviation is zero (e.g. single track or constant feature)
    timbre_std = np.where(timbre_std < 1e-6, 1.0, timbre_std).astype(np.float32)
    band_std = np.where(band_std < 1e-6, 1.0, band_std).astype(np.float32)

    return FeatureStats(
        track_count=n_tracks,
        timbre_mean=timbre_mean,
        timbre_std=timbre_std,
        band_mean=band_mean,
        band_std=band_std,
    )


def normalize_timbre_vector(timbre_raw: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Normalize 40-dim timbre vector via z-score followed by L2-normalization.

    Pipeline:
        1. z-score across coordinates using library-wide mean and std.
        2. L2-normalization so that ||v||_2 = 1.0 (for cosine distance HNSW index).
    """
    raw = np.asarray(timbre_raw, dtype=np.float32)
    std = np.where(stats.timbre_std < 1e-6, 1.0, stats.timbre_std)
    z = (raw - stats.timbre_mean) / std
    norm = float(np.linalg.norm(z))
    return (z / norm if norm > 1e-8 else z).astype(np.float32)


def normalize_band_vector(band_raw: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Normalize 8-dim energy band vector via library-wide z-score."""
    raw = np.asarray(band_raw, dtype=np.float32)
    std = np.where(stats.band_std < 1e-6, 1.0, stats.band_std)
    z = (raw - stats.band_mean) / std
    return z.astype(np.float32)


def normalize_chroma_vector(chroma_raw: np.ndarray) -> np.ndarray:
    """Normalize 12-dim pitch chroma vector via L2-normalization only (already a distribution)."""
    raw = np.asarray(chroma_raw, dtype=np.float32)
    norm = float(np.linalg.norm(raw))
    return (raw / norm if norm > 1e-8 else raw).astype(np.float32)


def save_feature_stats(conn: psycopg.Connection, stats: FeatureStats) -> None:
    """Upsert feature statistics into the feature_stats database table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO feature_stats (id, updated_at, track_count, stats)
            VALUES (1, now(), %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                updated_at = EXCLUDED.updated_at,
                track_count = EXCLUDED.track_count,
                stats = EXCLUDED.stats;
            """,
            (stats.track_count, Jsonb(stats.to_dict())),
        )


def load_feature_stats(conn: psycopg.Connection) -> FeatureStats | None:
    """Retrieve existing library feature statistics from the feature_stats table."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, updated_at, track_count, stats FROM feature_stats WHERE id = 1;")
        row = cur.fetchone()
        if not row:
            return None
        return FeatureStats.from_dict(
            data=row["stats"],
            track_count=row["track_count"],
            updated_at=row["updated_at"],
        )


def check_normalization_needed(
    conn: psycopg.Connection,
    threshold: float = 0.20,
) -> tuple[bool, int, int, float]:
    """Check whether library feature normalization should run based on track count growth.

    Returns:
        (is_needed, current_track_count, previous_track_count, growth_ratio)
    """
    current_count = count_analyzed_tracks(conn)
    stats = load_feature_stats(conn)
    prev_count = stats.track_count if stats is not None else 0

    if current_count == 0:
        return False, 0, prev_count, 0.0

    if prev_count <= 0:
        return True, current_count, 0, 1.0

    growth_ratio = (current_count - prev_count) / max(1, prev_count)
    is_needed = growth_ratio >= threshold
    return is_needed, current_count, prev_count, growth_ratio


def normalize_library_features(
    conn: psycopg.Connection | None = None,
    force: bool = False,
    growth_threshold: float = 0.20,
) -> NormalizationResult:
    """Compute library-wide stats and re-normalize vectors in track_features table.

    Args:
        conn: Optional active database connection. If None, acquires from pool.
        force: If True, executes normalization regardless of library growth ratio.
        growth_threshold: Minimum library growth ratio (default: 0.20 = 20%) to trigger run.

    Returns:
        NormalizationResult detailing updated tracks and computed stats.
    """

    def _execute(active_conn: psycopg.Connection) -> NormalizationResult:
        needed, cur_count, prev_count, growth = check_normalization_needed(
            active_conn, threshold=growth_threshold
        )
        if not needed and not force:
            existing_stats = load_feature_stats(active_conn)
            return NormalizationResult(
                tracks_updated=0,
                track_count=cur_count,
                previous_track_count=prev_count,
                growth_ratio=growth,
                stats=existing_stats,
                applied=False,
                reason=f"Library growth ({growth * 100:.1f}%) is below {growth_threshold * 100:.0f}% threshold",
            )

        # 1. Fetch all records from track_features
        with active_conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT track_id, frames_path, timbre_vec, chroma_vec, band_vec
                FROM track_features
                ORDER BY track_id;
                """
            )
            rows = cur.fetchall()

        if not rows:
            return NormalizationResult(
                tracks_updated=0,
                track_count=0,
                previous_track_count=prev_count,
                growth_ratio=growth,
                stats=None,
                applied=False,
                reason="No analyzed tracks found in library",
            )

        # 2. Extract raw vectors for each track (preferring .npz cache for exact raw arrays)
        track_ids: list[UUID] = []
        timbre_raw_list: list[np.ndarray] = []
        band_raw_list: list[np.ndarray] = []
        chroma_raw_list: list[np.ndarray] = []

        for row in rows:
            t_id = row["track_id"]
            frames_path_str = row.get("frames_path")
            cache_file = Path(frames_path_str) if frames_path_str else None

            timbre_raw: np.ndarray | None = None
            band_raw: np.ndarray | None = None
            chroma_raw: np.ndarray | None = None

            # Attempt to read exact raw features from .npz cache
            if cache_file and cache_file.exists():
                try:
                    cache_data = load_feature_cache(cache_file)
                    mfcc = cache_data["mfcc"]
                    band_energy = cache_data["band_energy"]
                    chroma = cache_data["chroma"]

                    timbre_raw = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).astype(
                        np.float32
                    )
                    band_raw = np.mean(band_energy, axis=1).astype(np.float32)
                    chroma_raw = np.mean(chroma, axis=1).astype(np.float32)
                except (OSError, KeyError, ValueError):
                    pass

            # Fallback to existing vectors if cache read was not possible
            if timbre_raw is None and row.get("timbre_vec") is not None:
                vec = row["timbre_vec"]
                timbre_raw = vec.to_numpy() if hasattr(vec, "to_numpy") else np.asarray(vec, dtype=np.float32)
            if band_raw is None and row.get("band_vec") is not None:
                vec = row["band_vec"]
                band_raw = vec.to_numpy() if hasattr(vec, "to_numpy") else np.asarray(vec, dtype=np.float32)
            if chroma_raw is None and row.get("chroma_vec") is not None:
                vec = row["chroma_vec"]
                chroma_raw = vec.to_numpy() if hasattr(vec, "to_numpy") else np.asarray(vec, dtype=np.float32)

            if timbre_raw is not None and band_raw is not None and chroma_raw is not None:
                track_ids.append(t_id)
                timbre_raw_list.append(timbre_raw)
                band_raw_list.append(band_raw)
                chroma_raw_list.append(chroma_raw)

        if not track_ids:
            return NormalizationResult(
                tracks_updated=0,
                track_count=len(rows),
                previous_track_count=prev_count,
                growth_ratio=growth,
                stats=None,
                applied=False,
                reason="Failed to extract feature vectors for tracks",
            )

        timbre_matrix = np.vstack(timbre_raw_list)
        band_matrix = np.vstack(band_raw_list)

        # 3. Compute library statistics
        stats = compute_feature_stats(timbre_matrix, band_matrix)

        # 4. Save stats in feature_stats table
        save_feature_stats(active_conn, stats)

        # 5. Normalize vectors for each track and prepare bulk update params
        update_params = []
        for t_id, t_raw, b_raw, c_raw in zip(
            track_ids, timbre_raw_list, band_raw_list, chroma_raw_list, strict=True
        ):
            t_norm = normalize_timbre_vector(t_raw, stats)
            b_norm = normalize_band_vector(b_raw, stats)
            c_norm = normalize_chroma_vector(c_raw)
            update_params.append((t_norm, b_norm, c_norm, t_id))

        # 6. Bulk update track_features in database
        with active_conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE track_features
                SET timbre_vec = %s,
                    band_vec = %s,
                    chroma_vec = %s
                WHERE track_id = %s;
                """,
                update_params,
            )

        return NormalizationResult(
            tracks_updated=len(update_params),
            track_count=len(update_params),
            previous_track_count=prev_count,
            growth_ratio=growth,
            stats=stats,
            applied=True,
            reason="Normalized successfully",
        )

    if conn is not None:
        return _execute(conn)
    else:
        with get_connection() as managed_conn, managed_conn.transaction():
            return _execute(managed_conn)
