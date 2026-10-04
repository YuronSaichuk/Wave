"""Database storage and retrieval for analyzed track features (track_features table)."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

# Algorithm version - bumped whenever the analysis pipeline changes
ANALYSIS_VERSION: int = 1


@dataclass
class TrackFeaturesRecord:
    """Record model for the track_features database table."""

    track_id: UUID
    bpm: float | None = None
    bpm_confidence: float | None = None
    beat_grid_path: str | None = None
    first_beat_sec: float | None = None
    key_pitch: int | None = None  # 0=C .. 11=B
    key_mode: int | None = None  # 0=minor, 1=major
    key_confidence: float | None = None
    camelot: str | None = None
    lufs_integrated: float | None = None
    lufs_range: float | None = None
    true_peak_db: float | None = None
    centroid_mean: float | None = None
    centroid_std: float | None = None
    rolloff_mean: float | None = None
    rolloff_std: float | None = None
    flatness_mean: float | None = None
    flatness_std: float | None = None
    bandwidth_mean: float | None = None
    bandwidth_std: float | None = None
    zcr_mean: float | None = None
    zcr_std: float | None = None
    timbre_vec: np.ndarray | None = None  # shape (40,), float32
    chroma_vec: np.ndarray | None = None  # shape (12,), float32
    band_vec: np.ndarray | None = None  # shape (8,), float32
    frames_path: str = ""
    sections: list[dict[str, Any]] | None = None


def upsert_track_features(conn: psycopg.Connection, record: TrackFeaturesRecord) -> None:
    """Insert or update track features in the database and update tracks.analyzed_at.

    Args:
        conn: Open psycopg database connection.
        record: TrackFeaturesRecord containing extracted metrics and vectors.
    """
    # Ensure vectors are appropriate format or None
    timbre = np.asarray(record.timbre_vec, dtype=np.float32) if record.timbre_vec is not None else None
    chroma = np.asarray(record.chroma_vec, dtype=np.float32) if record.chroma_vec is not None else None
    band = np.asarray(record.band_vec, dtype=np.float32) if record.band_vec is not None else None

    sections_json = Jsonb(record.sections) if record.sections is not None else None

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO track_features (
                track_id, bpm, bpm_confidence, beat_grid_path, first_beat_sec,
                key_pitch, key_mode, key_confidence, camelot,
                lufs_integrated, lufs_range, true_peak_db,
                centroid_mean, centroid_std, rolloff_mean, rolloff_std,
                flatness_mean, flatness_std, bandwidth_mean, bandwidth_std,
                zcr_mean, zcr_std,
                timbre_vec, chroma_vec, band_vec,
                frames_path, sections
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s
            )
            ON CONFLICT (track_id) DO UPDATE SET
                bpm = EXCLUDED.bpm,
                bpm_confidence = EXCLUDED.bpm_confidence,
                beat_grid_path = EXCLUDED.beat_grid_path,
                first_beat_sec = EXCLUDED.first_beat_sec,
                key_pitch = EXCLUDED.key_pitch,
                key_mode = EXCLUDED.key_mode,
                key_confidence = EXCLUDED.key_confidence,
                camelot = EXCLUDED.camelot,
                lufs_integrated = EXCLUDED.lufs_integrated,
                lufs_range = EXCLUDED.lufs_range,
                true_peak_db = EXCLUDED.true_peak_db,
                centroid_mean = EXCLUDED.centroid_mean,
                centroid_std = EXCLUDED.centroid_std,
                rolloff_mean = EXCLUDED.rolloff_mean,
                rolloff_std = EXCLUDED.rolloff_std,
                flatness_mean = EXCLUDED.flatness_mean,
                flatness_std = EXCLUDED.flatness_std,
                bandwidth_mean = EXCLUDED.bandwidth_mean,
                bandwidth_std = EXCLUDED.bandwidth_std,
                zcr_mean = EXCLUDED.zcr_mean,
                zcr_std = EXCLUDED.zcr_std,
                timbre_vec = EXCLUDED.timbre_vec,
                chroma_vec = EXCLUDED.chroma_vec,
                band_vec = EXCLUDED.band_vec,
                frames_path = EXCLUDED.frames_path,
                sections = EXCLUDED.sections;
            """,
            (
                record.track_id,
                record.bpm,
                record.bpm_confidence,
                record.beat_grid_path,
                record.first_beat_sec,
                record.key_pitch,
                record.key_mode,
                record.key_confidence,
                record.camelot,
                record.lufs_integrated,
                record.lufs_range,
                record.true_peak_db,
                record.centroid_mean,
                record.centroid_std,
                record.rolloff_mean,
                record.rolloff_std,
                record.flatness_mean,
                record.flatness_std,
                record.bandwidth_mean,
                record.bandwidth_std,
                record.zcr_mean,
                record.zcr_std,
                timbre,
                chroma,
                band,
                record.frames_path,
                sections_json,
            ),
        )

        # Mark track as analyzed and update algorithm version
        cur.execute(
            """
            UPDATE tracks
            SET analyzed_at = now(),
                analysis_version = %s
            WHERE id = %s;
            """,
            (ANALYSIS_VERSION, record.track_id),
        )


def get_track_features(conn: psycopg.Connection, track_id: UUID) -> TrackFeaturesRecord | None:
    """Retrieve analyzed features for a track by its UUID.

    Args:
        conn: Open psycopg database connection.
        track_id: Primary key UUID of the track.

    Returns:
        TrackFeaturesRecord instance or None if not found.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT track_id, bpm, bpm_confidence, beat_grid_path, first_beat_sec,
                   key_pitch, key_mode, key_confidence, camelot,
                   lufs_integrated, lufs_range, true_peak_db,
                   centroid_mean, centroid_std, rolloff_mean, rolloff_std,
                   flatness_mean, flatness_std, bandwidth_mean, bandwidth_std,
                   zcr_mean, zcr_std,
                   timbre_vec, chroma_vec, band_vec,
                   frames_path, sections
            FROM track_features
            WHERE track_id = %s;
            """,
            (track_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        # Convert pgvector Vector objects to numpy float32 arrays
        for vec_col in ("timbre_vec", "chroma_vec", "band_vec"):
            val = row.get(vec_col)
            if val is not None:
                if hasattr(val, "to_numpy"):
                    row[vec_col] = val.to_numpy().astype(np.float32)
                else:
                    row[vec_col] = np.asarray(val, dtype=np.float32)

        return TrackFeaturesRecord(**row)


def delete_track_features(conn: psycopg.Connection, track_id: UUID) -> bool:
    """Delete a track's feature record and reset tracks.analyzed_at.

    Args:
        conn: Open psycopg database connection.
        track_id: Primary key UUID of the track.

    Returns:
        True if record was deleted, False if not found.
    """
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM track_features WHERE track_id = %s RETURNING track_id;",
            (track_id,),
        )
        deleted = cur.fetchone() is not None

        cur.execute(
            """
            UPDATE tracks
            SET analyzed_at = NULL,
                analysis_version = NULL
            WHERE id = %s;
            """,
            (track_id,),
        )
        return deleted


def count_analyzed_tracks(conn: psycopg.Connection) -> int:
    """Count total analyzed tracks with stored features."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM track_features;")
        row = cur.fetchone()
        return row[0] if row else 0
