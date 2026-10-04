"""Full audio analysis pipeline and storage coordination."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import librosa
import numpy as np
import psycopg

from wave_core.analysis.bands import extract_band_energy
from wave_core.analysis.harmony import HarmonyFeatures, extract_harmony_features
from wave_core.analysis.loudness import LoudnessFeatures, measure_loudness
from wave_core.analysis.rhythm import RhythmFeatures, extract_rhythm_features
from wave_core.analysis.scanner import extract_metadata
from wave_core.analysis.spectral import (
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    SpectralFeatures,
    extract_spectral_features,
    load_audio,
)
from wave_core.analysis.structure import StructureFeatures, analyze_structure
from wave_core.storage.cache import get_cache_path, save_feature_cache
from wave_core.storage.db import get_connection
from wave_core.storage.features import TrackFeaturesRecord, upsert_track_features
from wave_core.storage.tracks import get_track_by_path, upsert_track


@dataclass(slots=True)
class FullAnalysisResult:
    """Consolidated audio analysis results containing spectral, rhythm, harmony, loudness, and vectors."""

    spectral: SpectralFeatures
    band_energy: np.ndarray  # [8, N] float32
    rhythm: RhythmFeatures
    harmony: HarmonyFeatures
    loudness: LoudnessFeatures
    structure: StructureFeatures
    rms: np.ndarray  # [N] float32
    timbre_vec: np.ndarray  # (40,) float32
    chroma_vec: np.ndarray  # (12,) float32
    band_vec: np.ndarray  # (8,) float32
    duration_sec: float
    sample_rate: int


def analyze_track_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
    progress_callback: object | None = None,
) -> FullAnalysisResult:
    """Run complete deterministic DSP analysis on mono audio array.

    Args:
        y: Mono float32 audio array.
        sr: Sample rate in Hz.
        hop_length: Hop length in samples.
        n_fft: FFT window size in samples.
        progress_callback: Optional callable(progress: float, message: str) -> None.

    Returns:
        FullAnalysisResult containing all feature models and search vectors.
    """
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Audio signal y must be a non-empty 1D array")

    y = np.ascontiguousarray(y, dtype=np.float32)
    duration_sec = float(len(y) / sr)

    # 1. Spectral features (STFT, Centroid, Rolloff, Bandwidth, Flatness, ZCR, MFCC)
    if callable(progress_callback):
        progress_callback(0.25, "spectral features")
    spectral = extract_spectral_features(y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # 2. 8-Band energy (smoothed dB)
    if callable(progress_callback):
        progress_callback(0.45, "band energy and rhythm")
    band_energy = extract_band_energy(spectral.stft, sr=sr, n_fft=n_fft)

    # 3. Rhythm and beat tracking
    rhythm = extract_rhythm_features(y, sr=sr, hop_length=hop_length, sub_energy=band_energy[0])

    # 4. Harmony and Camelot key
    if callable(progress_callback):
        progress_callback(0.65, "harmony and structure")
    harmony = extract_harmony_features(y, sr=sr, hop_length=hop_length)

    # 5. Loudness & Structure
    loudness = measure_loudness(y=y, sr=sr)
    structure = analyze_structure(y=y, sr=sr, hop_length=hop_length, loudness_info=loudness)

    # 6. RMS envelope over STFT frames
    if callable(progress_callback):
        progress_callback(0.85, "aggregating vectors")
    rms = librosa.feature.rms(y=y, hop_length=hop_length).squeeze(0).astype(np.float32)

    # 7. Search vectors:
    # Timbre vector: MFCC 20 x (mean, std) -> 40 dimensions
    mean_mfcc = np.mean(spectral.mfcc, axis=1)
    std_mfcc = np.std(spectral.mfcc, axis=1)
    timbre_vec = np.concatenate([mean_mfcc, std_mfcc]).astype(np.float32)

    # Chroma vector: Mean chroma -> L2-normalized -> 12 dimensions
    chroma_mean = np.mean(harmony.chroma, axis=1).astype(np.float32)
    norm = float(np.linalg.norm(chroma_mean))
    chroma_vec = (chroma_mean / norm if norm > 1e-8 else chroma_mean).astype(np.float32)

    # Band energy vector: Mean energy across 8 bands -> 8 dimensions
    band_vec = np.mean(band_energy, axis=1).astype(np.float32)

    return FullAnalysisResult(
        spectral=spectral,
        band_energy=band_energy,
        rhythm=rhythm,
        harmony=harmony,
        loudness=loudness,
        structure=structure,
        rms=rms,
        timbre_vec=timbre_vec,
        chroma_vec=chroma_vec,
        band_vec=band_vec,
        duration_sec=duration_sec,
        sample_rate=sr,
    )


def analyze_track_file(
    file_path: Path | str,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
    progress_callback: object | None = None,
) -> FullAnalysisResult:
    """Load an audio file and run complete deterministic DSP analysis."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if callable(progress_callback):
        progress_callback(0.15, "loading audio")
    y, loaded_sr = load_audio(path, sr=sr)
    return analyze_track_audio(
        y=y,
        sr=loaded_sr,
        hop_length=hop_length,
        n_fft=n_fft,
        progress_callback=progress_callback,
    )


def store_track_analysis(
    conn: psycopg.Connection,
    track_id: UUID,
    analysis: FullAnalysisResult,
    cache_dir: Path | str | None = None,
) -> tuple[Path, TrackFeaturesRecord]:
    """Persist frame-level features into .npz cache and aggregate features into PostgreSQL.

    Args:
        conn: Open database connection.
        track_id: Track primary key UUID.
        analysis: FullAnalysisResult containing all DSP output.
        cache_dir: Optional cache directory (defaults to settings.cache).

    Returns:
        Tuple of (cache_path, TrackFeaturesRecord).
    """
    cache_path = get_cache_path(track_id, cache_dir)

    # 1. Save frame-level features to compressed .npz
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

    # 2. Build TrackFeaturesRecord for database upsert
    record = TrackFeaturesRecord(
        track_id=track_id,
        bpm=float(analysis.rhythm.bpm),
        bpm_confidence=float(analysis.rhythm.bpm_confidence),
        beat_grid_path=str(cache_path),
        first_beat_sec=float(analysis.rhythm.first_beat_sec),
        key_pitch=analysis.harmony.pitch_class,
        key_mode=1 if analysis.harmony.key_mode.lower() == "major" else 0,
        key_confidence=float(analysis.harmony.key_confidence),
        camelot=analysis.harmony.camelot,
        lufs_integrated=float(analysis.loudness.lufs_integrated),
        lufs_range=float(analysis.loudness.lufs_range),
        true_peak_db=float(analysis.loudness.true_peak_db),
        centroid_mean=float(np.mean(analysis.spectral.centroid)),
        centroid_std=float(np.std(analysis.spectral.centroid)),
        rolloff_mean=float(np.mean(analysis.spectral.rolloff)),
        rolloff_std=float(np.std(analysis.spectral.rolloff)),
        flatness_mean=float(np.mean(analysis.spectral.flatness)),
        flatness_std=float(np.std(analysis.spectral.flatness)),
        bandwidth_mean=float(np.mean(analysis.spectral.bandwidth)),
        bandwidth_std=float(np.std(analysis.spectral.bandwidth)),
        zcr_mean=float(np.mean(analysis.spectral.zcr)),
        zcr_std=float(np.std(analysis.spectral.zcr)),
        timbre_vec=analysis.timbre_vec,
        chroma_vec=analysis.chroma_vec,
        band_vec=analysis.band_vec,
        frames_path=str(cache_path),
        sections=analysis.structure.sections_json,
    )

    # 3. Upsert into database
    upsert_track_features(conn, record)

    return cache_path, record


def analyze_and_store_track(
    file_path: Path | str,
    track_id: UUID | None = None,
    conn: psycopg.Connection | None = None,
    cache_dir: Path | str | None = None,
    force: bool = False,
    progress_callback: object | None = None,
) -> tuple[Path, TrackFeaturesRecord]:
    """Analyze an audio track and persist both .npz cache and database records.

    Args:
        file_path: Path to the audio file.
        track_id: Optional track UUID. If None, resolves or creates record in `tracks`.
        conn: Optional active database connection. If None, acquires one from pool.
        cache_dir: Optional custom cache directory.
        force: If False and track was already analyzed with valid cache, returns existing.
        progress_callback: Optional callable(progress: float, message: str) -> None.

    Returns:
        Tuple of (cache_path, TrackFeaturesRecord).
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    def _execute(active_conn: psycopg.Connection) -> tuple[Path, TrackFeaturesRecord]:
        nonlocal track_id
        if callable(progress_callback):
            progress_callback(0.05, "verifying metadata")
        if track_id is None:
            existing = get_track_by_path(active_conn, str(path))
            if existing is not None:
                track_id = existing.id
            else:
                meta = extract_metadata(path)
                track_id, _ = upsert_track(active_conn, meta)

        cache_path = get_cache_path(track_id, cache_dir)
        if not force and cache_path.exists():
            from wave_core.storage.features import get_track_features

            existing_features = get_track_features(active_conn, track_id)
            if existing_features is not None:
                if callable(progress_callback):
                    progress_callback(1.0, "already analyzed")
                return cache_path, existing_features

        # Run analysis and store results
        analysis = analyze_track_file(path, progress_callback=progress_callback)
        if callable(progress_callback):
            progress_callback(0.9, "saving features and cache")
        result = store_track_analysis(active_conn, track_id, analysis, cache_dir)
        if callable(progress_callback):
            progress_callback(1.0, "complete")
        return result

    if conn is not None:
        return _execute(conn)
    else:
        with get_connection() as managed_conn, managed_conn.transaction():
            return _execute(managed_conn)
