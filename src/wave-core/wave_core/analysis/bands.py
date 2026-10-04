"""8-band logarithmic energy analysis for similarity and light show cue engine."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

from wave_core.analysis.spectral import (
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    WINDOW,
    load_audio,
)

# 8 logarithmic frequency bands defined in 02-audio-analysis.md
BANDS_HZ: tuple[tuple[float, float], ...] = (
    (20.0, 60.0),  # sub        - kick drum fundamental
    (60.0, 150.0),  # bass       - basslines
    (150.0, 400.0),  # low_mid    - instrument body, warmth
    (400.0, 1000.0),  # mid        - vocals, snare fundamental
    (1000.0, 2500.0),  # high_mid   - presence, clarity
    (2500.0, 5000.0),  # presence   - attack, edge
    (5000.0, 8000.0),  # brilliance - cymbals, hi-hats
    (8000.0, 11000.0),  # air        - top end openness
)

BAND_NAMES: tuple[str, ...] = (
    "sub",
    "bass",
    "low_mid",
    "mid",
    "high_mid",
    "presence",
    "brilliance",
    "air",
)

NUM_BANDS: int = len(BANDS_HZ)
DEFAULT_SMOOTH_WINDOW: int = 3
DEFAULT_TOP_DB: float = 80.0


@dataclass(slots=True)
class BandEnergies:
    """Container for 8-band energy analysis results."""

    matrix: np.ndarray  # shape [8, N], float32 (dB values)
    band_names: tuple[str, ...] = BAND_NAMES
    bands_hz: tuple[tuple[float, float], ...] = BANDS_HZ
    sample_rate: int = SAMPLE_RATE
    hop_length: int = HOP_LENGTH

    @property
    def n_frames(self) -> int:
        """Total number of frames."""
        return int(self.matrix.shape[1])

    @property
    def n_bands(self) -> int:
        """Total number of frequency bands."""
        return int(self.matrix.shape[0])

    def get_band(self, name: str) -> np.ndarray:
        """Return 1D array of dB values across time for specified band name."""
        if name not in self.band_names:
            raise KeyError(f"Unknown band name: '{name}'. Available: {self.band_names}")
        idx = self.band_names.index(name)
        return self.matrix[idx]

    def mean_vector(self) -> np.ndarray:
        """Calculate mean dB across time for each band (shape: [8], float32)."""
        return np.mean(self.matrix, axis=1).astype(np.float32)


def create_band_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    bands: tuple[tuple[float, float], ...] = BANDS_HZ,
) -> np.ndarray:
    """Create a binary filterbank matrix mapping STFT bins to frequency bands.

    Args:
        sr: Sample rate in Hz.
        n_fft: FFT window length.
        bands: Tuple of (f_min, f_max) frequency ranges in Hz.

    Returns:
        Filterbank matrix of shape [num_bands, 1 + n_fft // 2], float32.
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    num_bins = len(freqs)
    filterbank = np.zeros((len(bands), num_bins), dtype=np.float32)

    for i, (f_min, f_max) in enumerate(bands):
        # Upper edge is inclusive for the highest band to catch nyquist boundary
        if i == len(bands) - 1:
            mask = (freqs >= f_min) & (freqs <= f_max)
        else:
            mask = (freqs >= f_min) & (freqs < f_max)
        filterbank[i, mask] = 1.0

    return filterbank


def extract_band_energy(
    stft_mag: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    bands: tuple[tuple[float, float], ...] = BANDS_HZ,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    top_db: float = DEFAULT_TOP_DB,
) -> np.ndarray:
    """Compute smoothed logarithmic energy across 8 frequency bands from STFT.

    Pipeline:
        1. STFT bin power (|S|^2) summed across each frequency band via filterbank.
        2. Power converted to dB (ref=max, clipped at top_db).
        3. Temporal smoothing via uniform moving average filter.

    Args:
        stft_mag: Magnitude STFT spectrogram of shape [num_bins, num_frames].
        sr: Audio sample rate in Hz.
        n_fft: FFT window length.
        bands: Frequency band boundaries.
        smooth_window: Temporal smoothing window size in frames (<=1 disables smoothing).
        top_db: Maximum dB dynamic range below peak.

    Returns:
        Matrix of shape [8, num_frames], float32.
    """
    if stft_mag.ndim != 2:
        raise ValueError(f"stft_mag must be 2D [num_bins, num_frames], got shape {stft_mag.shape}")

    expected_bins = 1 + n_fft // 2
    if stft_mag.shape[0] != expected_bins:
        raise ValueError(
            f"stft_mag frequency bin count ({stft_mag.shape[0]}) does not match 1 + n_fft // 2 ({expected_bins})"
        )

    # 1. Sum power (|S|^2) per band
    filterbank = create_band_filterbank(sr=sr, n_fft=n_fft, bands=bands)
    power_spec = stft_mag.astype(np.float32) ** 2
    band_energy = filterbank @ power_spec

    # 2. Convert to dB with silence protection
    max_val = float(np.max(band_energy))
    if max_val <= 1e-10:
        band_db = np.full_like(band_energy, -top_db, dtype=np.float32)
    else:
        band_db = librosa.power_to_db(band_energy, ref=max_val, top_db=top_db).astype(np.float32)

    # 3. Temporal smoothing
    if smooth_window > 1 and band_db.shape[1] >= smooth_window:
        band_db = uniform_filter1d(band_db, size=smooth_window, axis=1, mode="nearest")

    return np.ascontiguousarray(band_db, dtype=np.float32)


def extract_band_energy_from_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: str = WINDOW,
    bands: tuple[tuple[float, float], ...] = BANDS_HZ,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    top_db: float = DEFAULT_TOP_DB,
) -> np.ndarray:
    """Compute 8-band energy matrix directly from audio time series."""
    stft_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    stft_mag = np.abs(stft_complex).astype(np.float32)
    return extract_band_energy(
        stft_mag=stft_mag,
        sr=sr,
        n_fft=n_fft,
        bands=bands,
        smooth_window=smooth_window,
        top_db=top_db,
    )


def extract_band_energy_from_file(
    path: str | Path,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: str = WINDOW,
    bands: tuple[tuple[float, float], ...] = BANDS_HZ,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    top_db: float = DEFAULT_TOP_DB,
) -> np.ndarray:
    """Load audio from file and compute 8-band energy matrix."""
    y, sr_loaded = load_audio(path, sr=sr)
    return extract_band_energy_from_audio(
        y=y,
        sr=sr_loaded,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        bands=bands,
        smooth_window=smooth_window,
        top_db=top_db,
    )
