"""Loudness analysis using ITU-R BS.1770 / EBU R128 (pyloudnorm): integrated LUFS, LRA, and true peak."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly

from wave_core.analysis.spectral import SAMPLE_RATE, load_audio

# Floor for silence in LUFS
MIN_LUFS: float = -70.0


@dataclass(slots=True)
class LoudnessFeatures:
    """Loudness measurements adhering to ITU-R BS.1770 standards."""

    lufs_integrated: float  # Integrated loudness across track (standard target: -14 LUFS)
    lufs_range: float  # Loudness range (LRA) in LU, representing dynamic variation
    true_peak_db: float  # Maximum peak level in dBFS with 4x oversampling


def compute_true_peak_db(y: np.ndarray) -> float:
    """Estimate true peak level in dBFS using 4x oversampling.

    Args:
        y: Audio time series.

    Returns:
        True peak in dBFS (<= 0.0 for normal signals, -80.0 for silence).
    """
    if len(y) == 0:
        return -80.0

    try:
        # 4x oversampling to detect inter-sample peaks
        y_4x = resample_poly(y, up=4, down=1)
        peak = float(np.max(np.abs(y_4x)))
    except (ValueError, TypeError, MemoryError):
        peak = float(np.max(np.abs(y)))

    if peak <= 1e-6:
        return -80.0

    return float(20.0 * np.log10(peak))


def measure_loudness(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> LoudnessFeatures:
    """Measure integrated loudness (LUFS), loudness range (LRA), and true peak dB.

    Args:
        y: Audio time series (mono float32).
        sr: Sample rate in Hz (default: 22050).

    Returns:
        LoudnessFeatures dataclass instance.
    """
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Audio signal y must be a non-empty 1D array")

    y = np.ascontiguousarray(y, dtype=np.float32)
    true_peak = compute_true_peak_db(y)

    # Check for silence or near-silence
    rms = float(np.sqrt(np.mean(y**2)))
    if rms <= 1e-6 or len(y) < int(sr * 0.4):
        return LoudnessFeatures(
            lufs_integrated=MIN_LUFS,
            lufs_range=0.0,
            true_peak_db=true_peak,
        )

    meter = pyln.Meter(sr)

    try:
        lufs_int = float(meter.integrated_loudness(y))
        if np.isneginf(lufs_int) or np.isnan(lufs_int) or lufs_int < MIN_LUFS:
            lufs_int = MIN_LUFS
    except (ValueError, TypeError, AttributeError):
        lufs_int = MIN_LUFS

    try:
        lufs_rg = float(meter.loudness_range(y))
        if np.isnan(lufs_rg) or lufs_rg < 0.0:
            lufs_rg = 0.0
    except (ValueError, TypeError, AttributeError):
        lufs_rg = 0.0

    return LoudnessFeatures(
        lufs_integrated=round(lufs_int, 2),
        lufs_range=round(lufs_rg, 2),
        true_peak_db=round(true_peak, 2),
    )


def measure_loudness_from_file(
    path: str | Path,
    sr: int = SAMPLE_RATE,
) -> LoudnessFeatures:
    """Load audio file and measure loudness parameters."""
    y, sr_loaded = load_audio(path, sr=sr)
    return measure_loudness(y=y, sr=sr_loaded)
