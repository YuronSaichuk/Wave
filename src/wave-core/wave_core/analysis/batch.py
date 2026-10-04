"""Batch audio analysis orchestration and multiprocessing execution (T1.10)."""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import psycopg

from wave_core.analysis.normalization import (
    check_normalization_needed,
    normalize_library_features,
)
from wave_core.analysis.pipeline import analyze_track_file
from wave_core.analysis.scanner import extract_metadata, find_audio_files
from wave_core.config import settings
from wave_core.storage.cache import get_cache_path, save_feature_cache
from wave_core.storage.db import get_connection
from wave_core.storage.features import (
    ANALYSIS_VERSION,
    TrackFeaturesRecord,
    upsert_track_features,
)
from wave_core.storage.tracks import get_track_by_id, upsert_track

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchItemResult:
    """Result of analyzing an individual audio track."""

    track_id: UUID
    file_path: str
    success: bool
    error: str | None = None
    record_dict: dict[str, Any] | None = None
    duration_sec: float = 0.0
    compute_time_sec: float = 0.0


@dataclass(slots=True)
class BatchSummary:
    """Consolidated summary of a batch analysis run."""

    total_found: int
    analyzed_count: int
    skipped_count: int
    failed_count: int
    failed_items: list[tuple[str, str]]  # [(file_path, error_message)]
    elapsed_sec: float
    normalized: bool = False
    average_speed_sec: float = 0.0


def _worker_process_track(task_args: tuple[str, str, str]) -> tuple[str, str, bool, str | None, dict[str, Any] | None, float, float]:
    """Worker function executed in parallel processes.

    Computes pure DSP features and writes the compressed .npz cache file.
    Does NOT access PostgreSQL directly to avoid cross-process connection sharing.

    Args:
        task_args: (track_id_str, file_path_str, cache_path_str)

    Returns:
        (track_id_str, file_path_str, success, error_str, record_dict, duration_sec, compute_time_sec)
    """
    track_id_str, file_path_str, cache_path_str = task_args
    file_path = Path(file_path_str)
    cache_path = Path(cache_path_str)
    t0 = time.perf_counter()

    try:
        # 1. Run full DSP analysis
        analysis = analyze_track_file(file_path)

        # 2. Persist compressed .npz cache directly from worker
        save_feature_cache(
            path=cache_path,
            band_energy=analysis.band_energy,
            rms=analysis.rms,
            onset_env=analysis.rhythm.onset_env,
            beats=analysis.rhythm.beats,
            downbeats=analysis.rhythm.downbeats,
            chroma=analysis.harmony.chroma,
            mfcc=analysis.spectral.mfcc,
            downsample_features_if_needed=True,
        )

        # 3. Construct dictionary of features for main process DB write
        record_dict: dict[str, Any] = {
            "track_id": UUID(track_id_str),
            "bpm": float(analysis.rhythm.bpm),
            "bpm_confidence": float(analysis.rhythm.bpm_confidence),
            "beat_grid_path": str(cache_path),
            "first_beat_sec": float(analysis.rhythm.first_beat_sec),
            "key_pitch": analysis.harmony.pitch_class,
            "key_mode": 1 if analysis.harmony.key_mode.lower() == "major" else 0,
            "key_confidence": float(analysis.harmony.key_confidence),
            "camelot": analysis.harmony.camelot,
            "lufs_integrated": float(analysis.loudness.lufs_integrated),
            "lufs_range": float(analysis.loudness.lufs_range),
            "true_peak_db": float(analysis.loudness.true_peak_db),
            "centroid_mean": float(np.mean(analysis.spectral.centroid)),
            "centroid_std": float(np.std(analysis.spectral.centroid)),
            "rolloff_mean": float(np.mean(analysis.spectral.rolloff)),
            "rolloff_std": float(np.std(analysis.spectral.rolloff)),
            "flatness_mean": float(np.mean(analysis.spectral.flatness)),
            "flatness_std": float(np.std(analysis.spectral.flatness)),
            "bandwidth_mean": float(np.mean(analysis.spectral.bandwidth)),
            "bandwidth_std": float(np.std(analysis.spectral.bandwidth)),
            "zcr_mean": float(np.mean(analysis.spectral.zcr)),
            "zcr_std": float(np.std(analysis.spectral.zcr)),
            "timbre_vec": analysis.timbre_vec,
            "chroma_vec": analysis.chroma_vec,
            "band_vec": analysis.band_vec,
            "frames_path": str(cache_path),
            "sections": analysis.structure.sections_json,
        }

        compute_time = time.perf_counter() - t0
        return (track_id_str, file_path_str, True, None, record_dict, analysis.duration_sec, compute_time)
    except Exception as exc:  # noqa: BLE001 - Error isolation boundary for worker processes
        compute_time = time.perf_counter() - t0
        return (track_id_str, file_path_str, False, str(exc), None, 0.0, compute_time)


def analyze_batch(
    target_path: Path | str | None = None,
    force: bool = False,
    workers: int | None = None,
    cache_dir: Path | str | None = None,
    conn: psycopg.Connection | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    auto_normalize: bool = True,
) -> BatchSummary:
    """Execute batch audio analysis across files or directories.

    Args:
        target_path: File or directory to analyze (defaults to WAVE_LIBRARY).
        force: If True, re-analyzes tracks even if already analyzed.
        workers: Number of parallel worker processes (defaults to max(1, os.cpu_count() - 1)).
        cache_dir: Optional custom feature cache directory.
        conn: Optional active database connection.
        on_progress: Callback invoked as each track completes: (completed, total, track_name).
        auto_normalize: If True, runs feature normalization if library grew >= 20%.

    Returns:
        BatchSummary instance with execution statistics.
    """
    start_time = time.perf_counter()
    target = Path(target_path).resolve() if target_path is not None else settings.library.resolve()

    if not target.exists():
        raise FileNotFoundError(f"Target path does not exist: {target}")

    # 1. Discover audio files
    if target.is_file():
        audio_files = [target]
    else:
        audio_files = find_audio_files(target)

    total_found = len(audio_files)
    if total_found == 0:
        return BatchSummary(
            total_found=0,
            analyzed_count=0,
            skipped_count=0,
            failed_count=0,
            failed_items=[],
            elapsed_sec=time.perf_counter() - start_time,
            normalized=False,
            average_speed_sec=0.0,
        )

    # 2. Database preparation and candidate filtering
    def _run_batch(active_conn: psycopg.Connection) -> BatchSummary:
        tasks_to_run: list[tuple[str, str, str]] = []
        skipped_count = 0

        for f_path in audio_files:
            meta = extract_metadata(f_path)
            t_id, _ = upsert_track(active_conn, meta)

            c_path = get_cache_path(t_id, cache_dir)

            if not force:
                track = get_track_by_id(active_conn, t_id)
                if (
                    track is not None
                    and track.analyzed_at is not None
                    and track.analysis_version == ANALYSIS_VERSION
                    and c_path.exists()
                ):
                    skipped_count += 1
                    continue

            tasks_to_run.append((str(t_id), str(f_path), str(c_path)))

        analyzed_count = 0
        failed_count = 0
        failed_items: list[tuple[str, str]] = []
        total_tasks = len(tasks_to_run)

        # 3. Parallel or sequential worker execution
        num_workers = workers if workers is not None and workers > 0 else max(1, (os.cpu_count() or 2) - 1)
        # For single tasks or single worker, run in-process for speed and deterministic testing
        if total_tasks == 0:
            pass
        elif num_workers == 1 or total_tasks == 1:
            for i, task_args in enumerate(tasks_to_run, 1):
                t_id_str, f_str, ok, err, r_dict, _, _ = _worker_process_track(task_args)
                f_name = Path(f_str).name
                if ok and r_dict is not None:
                    rec = TrackFeaturesRecord(**r_dict)
                    upsert_track_features(active_conn, rec)
                    analyzed_count += 1
                else:
                    failed_count += 1
                    failed_items.append((f_str, err or "Unknown error"))
                    logger.warning("Analysis failed for %s: %s", f_name, err)

                if on_progress:
                    on_progress(i, total_tasks, f_name)
        else:
            # Parallel pool execution using spawn context
            ctx = get_context("spawn")
            with ctx.Pool(processes=min(num_workers, total_tasks)) as pool:
                for i, (t_id_str, f_str, ok, err, r_dict, _, _) in enumerate(
                    pool.imap_unordered(_worker_process_track, tasks_to_run), 1
                ):
                    f_name = Path(f_str).name
                    if ok and r_dict is not None:
                        rec = TrackFeaturesRecord(**r_dict)
                        upsert_track_features(active_conn, rec)
                        analyzed_count += 1
                    else:
                        failed_count += 1
                        failed_items.append((f_str, err or "Unknown error"))
                        logger.warning("Analysis failed for %s: %s", f_name, err)

                    if on_progress:
                        on_progress(i, total_tasks, f_name)

        # 4. Check normalization trigger
        normalized = False
        if auto_normalize and analyzed_count > 0:
            is_needed, _, _, _ = check_normalization_needed(active_conn, threshold=0.20)
            if is_needed:
                norm_res = normalize_library_features(active_conn, force=True)
                normalized = norm_res.applied

        elapsed = time.perf_counter() - start_time
        avg_speed = elapsed / analyzed_count if analyzed_count > 0 else 0.0

        return BatchSummary(
            total_found=total_found,
            analyzed_count=analyzed_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            failed_items=failed_items,
            elapsed_sec=elapsed,
            normalized=normalized,
            average_speed_sec=avg_speed,
        )

    if conn is not None:
        return _run_batch(conn)
    else:
        with get_connection() as managed_conn, managed_conn.transaction():
            return _run_batch(managed_conn)
