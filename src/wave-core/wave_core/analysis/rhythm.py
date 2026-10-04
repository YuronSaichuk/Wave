"""Rhythm analysis: tempo (BPM), octave correction, beat and downbeat tracking, confidence."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks

from wave_core.analysis.bands import extract_band_energy
from wave_core.analysis.spectral import (
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    load_audio,
)

# DJ tempo boundaries for octave correction
DEFAULT_MIN_BPM: float = 70.0
DEFAULT_MAX_BPM: float = 180.0


@dataclass(slots=True)
class RhythmFeatures:
    """Rhythm analysis results including tempo, beat timestamps, and downbeats."""

    bpm: float  # Primary tempo sit within DJ range [70, 180]
    bpm_raw: float  # Raw tempo as estimated before octave correction
    bpm_confidence: float  # Tempo confidence in [0.0, 1.0] from onset autocorrelation
    first_beat_sec: float  # Timestamp of first detected beat in seconds
    beats: np.ndarray  # 1D array of beat timestamps in seconds (float32)
    downbeats: np.ndarray  # 1D array of downbeat timestamps in seconds (float32)
    downbeat_indices: np.ndarray  # 1D array of indices into beats array (int64)
    onset_env: np.ndarray  # 1D onset strength envelope (float32)
    sample_rate: int = SAMPLE_RATE
    hop_length: int = HOP_LENGTH

    @property
    def n_beats(self) -> int:
        """Total number of detected beats."""
        return len(self.beats)

    @property
    def n_downbeats(self) -> int:
        """Total number of detected downbeats."""
        return len(self.downbeats)

    @property
    def beat_interval_sec(self) -> float:
        """Average interval between beats in seconds (from beats array, not nominal BPM)."""
        if len(self.beats) < 2:
            return 60.0 / self.bpm if self.bpm > 0 else 0.0
        return float(np.mean(np.diff(self.beats)))


def correct_tempo_octave(
    bpm: float,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> float:
    """Correct tempo octave to sit within standard DJ mixing range [min_bpm, max_bpm].

    Examples:
        tempo = 60.0 -> 120.0 (< 70 -> * 2)
        tempo = 190.0 -> 95.0 (> 180 -> / 2)
    """
    if bpm <= 0.0:
        return 0.0

    corrected = float(bpm)
    while corrected < min_bpm:
        corrected *= 2.0
    while corrected > max_bpm:
        corrected /= 2.0
    return float(corrected)


def compute_bpm_confidence(
    onset_env: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    min_bpm: float = 40.0,
    max_bpm: float = 240.0,
) -> float:
    """Compute confidence in [0.0, 1.0] from autocorrelation of the onset strength envelope.

    Args:
        onset_env: 1D onset strength envelope.
        sr: Sample rate in Hz.
        hop_length: STFT hop length.
        min_bpm: Minimum tempo to consider for autocorrelation lag.
        max_bpm: Maximum tempo to consider for autocorrelation lag.

    Returns:
        Confidence score between 0.0 (unreliable/ambient/silence) and 1.0 (clear rhythmic pulse).
    """
    if len(onset_env) == 0 or float(np.max(onset_env)) <= 1e-5:
        return 0.0

    fps = sr / hop_length
    min_lag = max(1, round(60.0 * fps / max_bpm))
    max_lag = min(len(onset_env) - 1, round(60.0 * fps / min_bpm))

    if max_lag <= min_lag + 2:
        return 0.0

    # Autocorrelation of onset envelope
    ac = librosa.autocorrelate(onset_env, max_size=max_lag + 1)
    if len(ac) <= min_lag or ac[0] <= 1e-6:
        return 0.0

    # Normalized autocorrelation window over valid tempo lags
    window = ac[min_lag : max_lag + 1] / ac[0]

    peaks, _ = find_peaks(window, prominence=0.01)
    if len(peaks) == 0:
        return 0.0

    peak_lags = peaks + min_lag
    peak_heights = window[peaks]
    order = np.argsort(peak_heights)[::-1]
    sorted_lags = peak_lags[order]
    sorted_heights = peak_heights[order]

    primary_lag = sorted_lags[0]
    primary_height = float(sorted_heights[0])

    # Find highest competing non-harmonic peak (ignore integer multiples / divisors of primary lag)
    competing_height = 0.0
    for lag, height in zip(sorted_lags[1:], sorted_heights[1:], strict=False):
        ratio = lag / primary_lag
        inv_ratio = primary_lag / lag
        # If lag is approximately 2x or 0.5x, it's an octave harmonic, not a competing meter
        is_harmonic = (abs(ratio - 2.0) < 0.15) or (abs(inv_ratio - 2.0) < 0.15)
        if not is_harmonic:
            competing_height = float(height)
            break

    # Confidence combines peak distinctiveness over competing peak and prominence
    ratio_score = (primary_height - competing_height) / primary_height if primary_height > 1e-6 else 0.0
    ratio_score = float(np.clip(ratio_score, 0.0, 1.0))

    # Scale by peak height relative to autocorrelation energy (e.g. ambient has small peaks)
    prominence_factor = float(np.clip(primary_height * 2.0, 0.0, 1.0))
    confidence = ratio_score * prominence_factor

    return float(np.clip(confidence, 0.0, 1.0))


def detect_downbeats(
    beats: np.ndarray,
    sub_energy: np.ndarray | None = None,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    group_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify downbeats (first beat of each measure) using sub-band energy.

    For each candidate offset k in 0..group_size-1, sums the sub-band energy on beats
    [k, k+group_size, k+2*group_size, ...] and selects the offset that maximizes energy.

    Args:
        beats: 1D array of beat timestamps in seconds.
        sub_energy: 1D array of sub-band energy across frames (shape [n_frames]).
        sr: Sample rate in Hz.
        hop_length: STFT hop length.
        group_size: Beats per measure (default: 4 for 4/4 meter).

    Returns:
        Tuple of (downbeat_timestamps, downbeat_indices_in_beats).
    """
    n_beats = len(beats)
    if n_beats == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    if n_beats < group_size or sub_energy is None or len(sub_energy) == 0:
        indices = np.arange(0, n_beats, group_size, dtype=np.int64)
        return beats[indices].astype(np.float32), indices

    # Convert beat timestamps to frame indices
    fps = sr / hop_length
    beat_frames = np.clip(
        np.round(beats * fps).astype(int),
        0,
        len(sub_energy) - 1,
    )

    # Sample sub-band energy around each beat (+/- 1 frame window to absorb jitter)
    beat_energies = np.zeros(n_beats, dtype=np.float32)
    for i, frame in enumerate(beat_frames):
        start = max(0, frame - 1)
        end = min(len(sub_energy), frame + 2)
        beat_energies[i] = np.max(sub_energy[start:end])

    # Find best offset k in 0..group_size-1
    best_k = 0
    best_score = -float("inf")

    for k in range(min(group_size, n_beats)):
        group = beat_energies[k::group_size]
        if len(group) > 0:
            score = float(np.sum(group))
            if score > best_score:
                best_score = score
                best_k = k

    downbeat_indices = np.arange(best_k, n_beats, group_size, dtype=np.int64)
    downbeats = beats[downbeat_indices].astype(np.float32)
    return downbeats, downbeat_indices


def extract_rhythm_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    sub_energy: np.ndarray | None = None,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> RhythmFeatures:
    """Extract tempo, beat grid, downbeats, and rhythm confidence from audio.

    Args:
        y: Mono float32 audio time series.
        sr: Sample rate in Hz (default: 22050).
        hop_length: Hop length in samples (default: 512).
        sub_energy: Optional 1D sub-band energy array. If None, computed via STFT.
        min_bpm: Minimum DJ BPM threshold for octave correction.
        max_bpm: Maximum DJ BPM threshold for octave correction.

    Returns:
        RhythmFeatures dataclass instance.
    """
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Audio signal y must be a non-empty 1D array")

    y = np.ascontiguousarray(y, dtype=np.float32)

    # 1. Onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
    ).astype(np.float32)

    # 2. Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
    )
    bpm_raw = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).astype(np.float32)

    # Refine nominal BPM from physical beat intervals if sufficient beats exist
    if len(beats) >= 4:
        intervals = np.diff(beats)
        median_interval = float(np.median(intervals))
        if median_interval > 0.0:
            # Filter out occasional skipped or doubled beats
            valid = intervals[np.abs(intervals - median_interval) < 0.25 * median_interval]
            if len(valid) >= 3:
                bpm_grid = 60.0 / float(np.mean(valid))
                bpm = correct_tempo_octave(bpm_grid, min_bpm=min_bpm, max_bpm=max_bpm)
            else:
                bpm = correct_tempo_octave(bpm_raw, min_bpm=min_bpm, max_bpm=max_bpm)
        else:
            bpm = correct_tempo_octave(bpm_raw, min_bpm=min_bpm, max_bpm=max_bpm)
    else:
        bpm = correct_tempo_octave(bpm_raw, min_bpm=min_bpm, max_bpm=max_bpm)

    # 3. Confidence via autocorrelation
    confidence = compute_bpm_confidence(
        onset_env=onset_env,
        sr=sr,
        hop_length=hop_length,
    )

    # 4. Compute sub-band energy if not provided
    if sub_energy is None:
        stft_complex = librosa.stft(y, n_fft=N_FFT, hop_length=hop_length)
        stft_mag = np.abs(stft_complex).astype(np.float32)
        band_matrix = extract_band_energy(stft_mag, sr=sr, n_fft=N_FFT)
        sub_energy = band_matrix[0]

    # 5. Downbeat detection
    downbeats, downbeat_indices = detect_downbeats(
        beats=beats,
        sub_energy=sub_energy,
        sr=sr,
        hop_length=hop_length,
    )

    first_beat_sec = float(beats[0]) if len(beats) > 0 else 0.0

    return RhythmFeatures(
        bpm=bpm,
        bpm_raw=bpm_raw,
        bpm_confidence=confidence,
        first_beat_sec=first_beat_sec,
        beats=beats,
        downbeats=downbeats,
        downbeat_indices=downbeat_indices,
        onset_env=onset_env,
        sample_rate=sr,
        hop_length=hop_length,
    )


def extract_rhythm_features_from_file(
    path: str | Path,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> RhythmFeatures:
    """Load audio from file and extract rhythm features."""
    y, sr_loaded = load_audio(path, sr=sr)
    return extract_rhythm_features(
        y=y,
        sr=sr_loaded,
        hop_length=hop_length,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )
