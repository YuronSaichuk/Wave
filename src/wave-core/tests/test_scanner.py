import os
from pathlib import Path
from uuid import uuid4

import pytest

from wave_core.analysis.scanner import (
    calculate_quick_hash,
    extract_metadata,
    find_audio_files,
    parse_filename_fallback,
    scan_directory,
)
from wave_core.storage.db import get_connection
from wave_core.storage.tracks import (
    TrackMetadata,
    UpsertStatus,
    count_tracks,
    get_track_by_id,
    get_track_by_path,
    upsert_track,
)


def test_parse_filename_fallback():
    artist, title = parse_filename_fallback(Path("Daft Punk - One More Time.mp3"))
    assert artist == "Daft Punk"
    assert title == "One More Time"

    artist, title = parse_filename_fallback(Path("JustATrack.wav"))
    assert artist is None
    assert title == "JustATrack"

    artist, title = parse_filename_fallback(Path("The Prodigy — Breathe.flac"))
    assert artist == "The Prodigy"
    assert title == "Breathe"

    artist, title = parse_filename_fallback(Path("Deadmau5 – Strobe.ogg"))
    assert artist == "Deadmau5"
    assert title == "Strobe"


def test_calculate_quick_hash_small_file(tmp_path: Path):
    sample = tmp_path / "small.wav"
    content = b"RIFF" + b"\x00" * 1024
    sample.write_bytes(content)

    hash1 = calculate_quick_hash(sample)
    hash2 = calculate_quick_hash(sample)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_calculate_quick_hash_large_file_structure(tmp_path: Path):
    large = tmp_path / "large.bin"
    # Create 2.5 MB file
    chunk_1mb = b"A" * (1024 * 1024)
    middle_500k = b"M" * (512 * 1024)
    last_1mb = b"Z" * (1024 * 1024)

    large.write_bytes(chunk_1mb + middle_500k + last_1mb)
    initial_hash = calculate_quick_hash(large)

    # Change middle bytes - hash should remain unchanged (since middle is skipped)
    modified_middle = b"X" * (512 * 1024)
    large.write_bytes(chunk_1mb + modified_middle + last_1mb)
    assert calculate_quick_hash(large) == initial_hash

    # Change first byte - hash must change
    large.write_bytes(b"B" + chunk_1mb[1:] + modified_middle + last_1mb)
    assert calculate_quick_hash(large) != initial_hash


def test_find_audio_files(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()

    (tmp_path / "t1.mp3").touch()
    (tmp_path / "t2.WAV").touch()
    (sub / "t3.flac").touch()
    (sub / "t4.m4a").touch()
    (sub / "t5.ogg").touch()
    (tmp_path / "readme.txt").touch()
    (sub / "cover.jpg").touch()

    files = find_audio_files(tmp_path)
    names = {f.name.lower() for f in files}
    assert names == {"t1.mp3", "t2.wav", "t3.flac", "t4.m4a", "t5.ogg"}


def test_extract_metadata_fallback(tmp_path: Path):
    dummy = tmp_path / "Eric Prydz - Opus.wav"
    dummy.write_bytes(b"RIFF" + b"\x00" * 500)

    meta = extract_metadata(dummy)
    assert meta.artist == "Eric Prydz"
    assert meta.title == "Opus"
    assert meta.file_hash is not None
    assert str(dummy.resolve()) == meta.file_path


def test_tracks_db_upsert_lifecycle():
    fake_path = f"/music/test_{uuid4()}.mp3"
    meta = TrackMetadata(
        file_path=fake_path,
        file_hash="hash_v1",
        title="Test Track",
        artist="Test Artist",
        album="Test Album",
        duration_sec=180.5,
        sample_rate=44100,
    )

    with get_connection() as conn:
        with conn.transaction():
            # 1. Insert
            track_id, status = upsert_track(conn, meta)
            assert status == UpsertStatus.INSERTED

            track = get_track_by_id(conn, track_id)
            assert track is not None
            assert track.title == "Test Track"
            assert track.artist == "Test Artist"
            assert track.analyzed_at is None

            # Mark as analyzed to test reset on hash change
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE tracks SET analyzed_at = now() WHERE id = %s;",
                    (track_id,),
                )

            # 2. Re-insert identical track -> UNCHANGED
            _, status = upsert_track(conn, meta)
            assert status == UpsertStatus.UNCHANGED
            track_after = get_track_by_id(conn, track_id)
            assert track_after.analyzed_at is not None

            # 3. Update with new hash -> HASH_CHANGED and analyzed_at reset to NULL
            meta.file_hash = "hash_v2"
            _, status = upsert_track(conn, meta)
            assert status == UpsertStatus.HASH_CHANGED

            track_updated = get_track_by_id(conn, track_id)
            assert track_updated.file_hash == "hash_v2"
            assert track_updated.analyzed_at is None

            # 4. Lookup by path
            by_path = get_track_by_path(conn, fake_path)
            assert by_path is not None
            assert by_path.id == track_id

            # Clean up test track
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tracks WHERE id = %s;", (track_id,))
