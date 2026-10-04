"""Per-track frame-level feature caching (.npz) storage and retrieval."""

from pathlib import Path
from uuid import UUID

import numpy as np

from wave_core.config import settings

# Canonical array keys saved into each track's .npz cache
CACHE_KEYS: tuple[str, ...] = (
    "band_energy",
    "rms",
    "onset_env",
    "beats",
    "downbeats",
    "chroma",
    "mfcc",
)

# Maximum acceptable size for a 4-minute compressed feature cache
MAX_CACHE_SIZE_BYTES_4MIN: int = 2 * 1024 * 1024  # 2 MB

DEFAULT_SAMPLE_RATE: int = 22050
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_TARGET_FPS: float = 5.0


def _downsample_features(features: np.ndarray, factor: int = 8) -> np.ndarray:
    """Downsample feature matrix across time via block averaging."""
    if factor <= 1 or features.shape[1] <= factor:
        return features
    n_features, n_frames = features.shape
    n_blocks = n_frames // factor
    trimmed = features[:, : n_blocks * factor]
    reshaped = trimmed.reshape(n_features, n_blocks, factor)
    return np.mean(reshaped, axis=2).astype(np.float32)


def get_cache_path(track_id: UUID | str, cache_dir: Path | str | None = None) -> Path:
    """Resolve destination .npz cache path for a given track identifier."""
    base_dir = Path(cache_dir) if cache_dir is not None else settings.cache
    return base_dir / f"{track_id}.npz"


def save_feature_cache(
    path: Path | str,
    band_energy: np.ndarray,
    rms: np.ndarray,
    onset_env: np.ndarray,
    beats: np.ndarray,
    downbeats: np.ndarray,
    chroma: np.ndarray,
    mfcc: np.ndarray,
    downsample_features_if_needed: bool = True,
    sr: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> Path:
    """Save frame-level audio features into a compressed .npz archive using float32 arrays.

    Archive schema:
        band_energy: [8 x N] float32 - 8 logarithmic frequency bands for lighting/mixing
        rms:         [N] float32     - RMS envelope over STFT frames
        onset_env:   [N] float32     - Spectral flux / onset envelope
        beats:       [B] float32     - Beat onset timestamps in seconds
        downbeats:   [D] float32     - Downbeat timestamps (bar starts) in seconds
        chroma:      [12 x M] float32 - Downsampled chroma profile (~5 Hz)
        mfcc:        [20 x M] float32 - Downsampled MFCC coefficients (~5 Hz)

    Args:
        path: Destination .npz file path.
        band_energy: 2D array [8, N] of logarithmic energy.
        rms: 1D array [N] (or [1, N]) of RMS energy.
        onset_env: 1D array [N] of onset envelope.
        beats: 1D array of beat timestamps in seconds.
        downbeats: 1D array of downbeat timestamps in seconds.
        chroma: 2D array [12, M] or [12, N].
        mfcc: 2D array [20, M] or [20, N].
        downsample_features_if_needed: If True, downsamples full-rate chroma and mfcc to ~5 Hz.
        sr: Audio sample rate in Hz.
        hop_length: STFT hop length in samples.

    Returns:
        Path to the saved compressed .npz file.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate and prepare band_energy [8, N]
    band_energy_arr = np.ascontiguousarray(band_energy, dtype=np.float32)
    if band_energy_arr.ndim != 2 or band_energy_arr.shape[0] != 8:
        raise ValueError(f"band_energy must have shape [8, N], got {band_energy_arr.shape}")
    n_frames = band_energy_arr.shape[1]

    # Validate and prepare rms [N]
    rms_arr = np.ascontiguousarray(rms, dtype=np.float32).squeeze()
    if rms_arr.ndim != 1 or len(rms_arr) != n_frames:
        raise ValueError(f"rms must have length {n_frames}, got {rms_arr.shape}")

    # Validate and prepare onset_env [N]
    onset_arr = np.ascontiguousarray(onset_env, dtype=np.float32).squeeze()
    if onset_arr.ndim != 1 or len(onset_arr) != n_frames:
        raise ValueError(f"onset_env must have length {n_frames}, got {onset_arr.shape}")

    # Validate beats and downbeats
    beats_arr = np.ascontiguousarray(beats, dtype=np.float32).ravel()
    downbeats_arr = np.ascontiguousarray(downbeats, dtype=np.float32).ravel()

    # Prepare chroma [12, M]
    chroma_arr = np.ascontiguousarray(chroma, dtype=np.float32)
    if chroma_arr.ndim != 2 or chroma_arr.shape[0] != 12:
        raise ValueError(f"chroma must have shape [12, M], got {chroma_arr.shape}")

    # Prepare mfcc [20, M]
    mfcc_arr = np.ascontiguousarray(mfcc, dtype=np.float32)
    if mfcc_arr.ndim != 2 or mfcc_arr.shape[0] != 20:
        raise ValueError(f"mfcc must have shape [20, M], got {mfcc_arr.shape}")

    # Downsample chroma and mfcc to ~5 Hz if full frame rate provided
    if downsample_features_if_needed:
        fps = sr / hop_length
        ds_factor = max(1, round(fps / DEFAULT_TARGET_FPS))
        if chroma_arr.shape[1] == n_frames and ds_factor > 1:
            chroma_arr = _downsample_features(chroma_arr, factor=ds_factor)
        if mfcc_arr.shape[1] == n_frames and ds_factor > 1:
            mfcc_arr = _downsample_features(mfcc_arr, factor=ds_factor)

    # Save compressed archive
    np.savez_compressed(
        target_path,
        band_energy=band_energy_arr,
        rms=rms_arr,
        onset_env=onset_arr,
        beats=beats_arr,
        downbeats=downbeats_arr,
        chroma=chroma_arr,
        mfcc=mfcc_arr,
    )

    return target_path


def load_feature_cache(path: Path | str) -> dict[str, np.ndarray]:
    """Load cached features from a .npz file into memory.

    Args:
        path: Path to .npz file.

    Returns:
        Dictionary mapping feature names to float32 numpy arrays.
    """
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Cache file not found: {target_path}")

    features: dict[str, np.ndarray] = {}
    with np.load(target_path) as data:
        for key in CACHE_KEYS:
            if key not in data:
                raise KeyError(f"Corrupted cache file {target_path}: missing key '{key}'")
            features[key] = np.array(data[key], dtype=np.float32)

    return features


def get_cache_size_bytes(path: Path | str) -> int:
    """Return the file size in bytes of a cached .npz file."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Cache file not found: {target_path}")
    return target_path.stat().st_size
