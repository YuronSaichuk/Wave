from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row


@dataclass
class TrackMetadata:
    file_path: str
    file_hash: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    duration_sec: float | None = None
    sample_rate: int | None = None


@dataclass
class TrackRecord:
    id: UUID
    file_path: str
    file_hash: str
    title: str | None
    artist: str | None
    album: str | None
    duration_sec: float | None
    sample_rate: int | None
    added_at: datetime
    analyzed_at: datetime | None
    analysis_version: int | None


class UpsertStatus:
    INSERTED = "inserted"
    HASH_CHANGED = "hash_changed"
    METADATA_UPDATED = "metadata_updated"
    UNCHANGED = "unchanged"


def upsert_track(conn: psycopg.Connection, meta: TrackMetadata) -> tuple[UUID, str]:
    """
    Insert or update a track in the database based on its file_path.
    If file_hash changed, resets analyzed_at to NULL.
    Returns (track_id, UpsertStatus).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        # Check existing record
        cur.execute(
            """
            SELECT id, file_hash, title, artist, album, duration_sec, sample_rate, analyzed_at
            FROM tracks
            WHERE file_path = %s;
            """,
            (meta.file_path,),
        )
        existing = cur.fetchone()

        if existing is None:
            new_id = uuid4()
            cur.execute(
                """
                INSERT INTO tracks (id, file_path, file_hash, title, artist, album, duration_sec, sample_rate)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    new_id,
                    meta.file_path,
                    meta.file_hash,
                    meta.title,
                    meta.artist,
                    meta.album,
                    meta.duration_sec,
                    meta.sample_rate,
                ),
            )
            return new_id, UpsertStatus.INSERTED

        track_id = existing["id"]
        hash_changed = existing["file_hash"] != meta.file_hash

        if hash_changed:
            cur.execute(
                """
                UPDATE tracks
                SET file_hash = %s,
                    title = %s,
                    artist = %s,
                    album = %s,
                    duration_sec = %s,
                    sample_rate = %s,
                    analyzed_at = NULL
                WHERE id = %s;
                """,
                (
                    meta.file_hash,
                    meta.title,
                    meta.artist,
                    meta.album,
                    meta.duration_sec,
                    meta.sample_rate,
                    track_id,
                ),
            )
            return track_id, UpsertStatus.HASH_CHANGED

        # Check if metadata changed (e.g. tag updated without audio content change)
        meta_changed = (
            existing["title"] != meta.title
            or existing["artist"] != meta.artist
            or existing["album"] != meta.album
            or existing["duration_sec"] != meta.duration_sec
            or existing["sample_rate"] != meta.sample_rate
        )

        if meta_changed:
            cur.execute(
                """
                UPDATE tracks
                SET title = %s,
                    artist = %s,
                    album = %s,
                    duration_sec = %s,
                    sample_rate = %s
                WHERE id = %s;
                """,
                (
                    meta.title,
                    meta.artist,
                    meta.album,
                    meta.duration_sec,
                    meta.sample_rate,
                    track_id,
                ),
            )
            return track_id, UpsertStatus.METADATA_UPDATED

        return track_id, UpsertStatus.UNCHANGED


def get_track_by_id(conn: psycopg.Connection, track_id: UUID) -> TrackRecord | None:
    """Retrieve a single track by its primary key UUID."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, file_path, file_hash, title, artist, album,
                   duration_sec, sample_rate, added_at, analyzed_at, analysis_version
            FROM tracks
            WHERE id = %s;
            """,
            (track_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return TrackRecord(**row)


def get_track_by_path(conn: psycopg.Connection, file_path: str) -> TrackRecord | None:
    """Retrieve a track by file path."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, file_path, file_hash, title, artist, album,
                   duration_sec, sample_rate, added_at, analyzed_at, analysis_version
            FROM tracks
            WHERE file_path = %s;
            """,
            (file_path,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return TrackRecord(**row)


def count_tracks(conn: psycopg.Connection) -> int:
    """Count total records in tracks table."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM tracks;")
        row = cur.fetchone()
        return row[0] if row else 0
