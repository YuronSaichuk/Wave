import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mutagen
from mutagen.easyid3 import EasyID3

from wave_core.storage.db import get_connection
from wave_core.storage.tracks import TrackMetadata, UpsertStatus, upsert_track

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".ogg"}
HASH_CHUNK_SIZE = 1024 * 1024  # 1 MB


@dataclass
class ScanSummary:
    total_found: int = 0
    inserted: int = 0
    hash_changed: int = 0
    metadata_updated: int = 0
    unchanged: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def calculate_quick_hash(path: Path) -> str:
    """
    Calculate sha256 of first 1MB + last 1MB + file size.
    For small files (<= 2MB), hashes the entire file + file size.
    """
    size = path.stat().st_size
    h = hashlib.sha256()

    with open(path, "rb") as f:
        if size <= HASH_CHUNK_SIZE * 2:
            h.update(f.read())
        else:
            first_chunk = f.read(HASH_CHUNK_SIZE)
            f.seek(size - HASH_CHUNK_SIZE)
            last_chunk = f.read(HASH_CHUNK_SIZE)
            h.update(first_chunk)
            h.update(last_chunk)

    h.update(str(size).encode("ascii"))
    return h.hexdigest()


def parse_filename_fallback(path: Path) -> tuple[str | None, str]:
    """
    Extract (artist, title) from filename when tags are absent or empty.
    e.g. 'Artist - Title.mp3' -> ('Artist', 'Title')
    'Title.wav' -> (None, 'Title')
    """
    stem = path.stem.strip()
    for separator in [" - ", " — ", " – "]:
        if separator in stem:
            parts = stem.split(separator, 1)
            artist = parts[0].strip() or None
            title = parts[1].strip() or stem
            return artist, title

    return None, stem


def extract_metadata(path: Path) -> TrackMetadata:
    """
    Extract tags and audio metadata from an audio file using mutagen.
    Falls back to filename parsing for title/artist if tags are missing.
    """
    file_path = str(path.resolve())
    file_hash = calculate_quick_hash(path)

    title: str | None = None
    artist: str | None = None
    album: str | None = None
    duration_sec: float | None = None
    sample_rate: int | None = None

    # 1. Try reading with easy tags (supports mp3, flac, ogg, etc.)
    try:
        audio = mutagen.File(path, easy=True)
        if audio is not None:
            if audio.tags:
                title = audio.tags.get("title", [None])[0]
                artist = audio.tags.get("artist", [None])[0]
                album = audio.tags.get("album", [None])[0]

            if audio.info:
                duration_sec = getattr(audio.info, "length", None)
                sample_rate = getattr(audio.info, "sample_rate", None)
    except Exception as exc:
        logger.debug("Easy tag read failed for %s: %s", path, exc)

    # 2. Fallback to standard mutagen.File if easy tags returned no info
    if duration_sec is None or sample_rate is None or title is None:
        try:
            audio_raw = mutagen.File(path)
            if audio_raw is not None:
                if duration_sec is None and audio_raw.info:
                    duration_sec = getattr(audio_raw.info, "length", None)
                if sample_rate is None and audio_raw.info:
                    sample_rate = getattr(audio_raw.info, "sample_rate", None)
                if title is None and audio_raw.tags:
                    # Generic tag lookup
                    for t_key in ["title", "TIT2", "\xa9nam"]:
                        if t_key in audio_raw.tags:
                            val = audio_raw.tags[t_key]
                            title = str(val[0] if isinstance(val, list) else val)
                            break
        except Exception as exc:
            logger.debug("Raw tag read failed for %s: %s", path, exc)

    # 3. If title or artist still missing, parse filename
    parsed_artist, parsed_title = parse_filename_fallback(path)
    if not title:
        title = parsed_title
    if not artist and parsed_artist:
        artist = parsed_artist

    # Round duration if present
    if duration_sec is not None:
        duration_sec = round(float(duration_sec), 3)

    return TrackMetadata(
        file_path=file_path,
        file_hash=file_hash,
        title=title,
        artist=artist,
        album=album,
        duration_sec=duration_sec,
        sample_rate=sample_rate,
    )


def find_audio_files(directory: Path) -> list[Path]:
    """Recursively discover all audio files with supported extensions."""
    if not directory.exists():
        return []

    files: list[Path] = []
    for item in sorted(directory.rglob("*")):
        if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(item)
    return files


def scan_directory(
    directory: Path,
    progress_callback: Callable[[Path, int, int], None] | None = None,
) -> ScanSummary:
    """
    Recursively scan directory for audio files and upsert them into the database.
    """
    summary = ScanSummary()
    files = find_audio_files(directory)
    summary.total_found = len(files)

    if not files:
        return summary

    with get_connection() as conn:
        for idx, file_path in enumerate(files, start=1):
            if progress_callback:
                progress_callback(file_path, idx, summary.total_found)

            try:
                meta = extract_metadata(file_path)
                with conn.transaction():
                    _, status = upsert_track(conn, meta)

                if status == UpsertStatus.INSERTED:
                    summary.inserted += 1
                elif status == UpsertStatus.HASH_CHANGED:
                    summary.hash_changed += 1
                elif status == UpsertStatus.METADATA_UPDATED:
                    summary.metadata_updated += 1
                elif status == UpsertStatus.UNCHANGED:
                    summary.unchanged += 1
            except Exception as exc:
                logger.exception("Failed to process %s", file_path)
                summary.errors.append((str(file_path), str(exc)))

    return summary
