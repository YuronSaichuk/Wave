"""Spectral feature extraction: STFT, centroid, rolloff, bandwidth, flatness, contrast, ZCR, MFCC."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

# Audio analysis base parameters (fixed across analysis pipeline)
SAMPLE_RATE: int = 22050  # 22.05 kHz mono
N_FFT: int = 2048  # ~93 ms window
HOP_LENGTH: int = 512  # ~23 ms step -> ~43 frames/sec
WINDOW: str = "hann"
ROLLOFF_PERCENT: float = 0.85
N_MFCC: int = 20


@dataclass(slots=True)
class SpectralFeatures:
    """Frame-by-frame spectral descriptors of an audio signal."""

    stft: np.ndarray  # shape [1025, N], magnitude spectrogram (float32)
    stft_db: np.ndarray  # shape [1025, N], dB spectrogram ref=max (float32)
    centroid: np.ndarray  # shape [N], spectral centroid in Hz (float32)
    rolloff: np.ndarray  # shape [N], spectral rolloff at 85% energy in Hz (float32)
    bandwidth: np.ndarray  # shape [N], spectral bandwidth in Hz (float32)
    flatness: np.ndarray  # shape [N], spectral flatness in [0, 1] (float32)
    contrast: np.ndarray  # shape [7, N], spectral contrast in dB (float32)
    zcr: np.ndarray  # shape [N], zero-crossing rate in [0, 1] (float32)
    mfcc: np.ndarray  # shape [20, N], 20 MFCC coefficients (float32)
    sample_rate: int = SAMPLE_RATE
    hop_length: int = HOP_LENGTH
    n_fft: int = N_FFT
    duration_sec: float = 0.0

    @property
    def n_frames(self) -> int:
        """Total number of analysis frames."""
        return int(self.stft.shape[1])


def load_audio(
    path: str | Path,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Load audio file as mono float32 at specified sample rate."""
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), sr_out


def extract_spectral_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: str = WINDOW,
    roll_percent: float = ROLLOFF_PERCENT,
    n_mfcc: int = N_MFCC,
) -> SpectralFeatures:
    """Compute frame-by-frame spectral features from an audio time series.

    Args:
        y: Audio time series (1D numpy array, mono).
        sr: Audio sample rate in Hz (default: 22050).
        n_fft: STFT window length (default: 2048).
        hop_length: STFT hop length (default: 512).
        window: STFT window type (default: 'hann').
        roll_percent: Energy percentage for rolloff (default: 0.85).
        n_mfcc: Number of MFCCs to extract (default: 20).

    Returns:
        SpectralFeatures dataclass instance containing all extracted features.
    """
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Audio signal y must be a non-empty 1D array")

    # Ensure float32 mono
    y = np.ascontiguousarray(y, dtype=np.float32)
    duration_sec = float(len(y) / sr)

    # 1. STFT and magnitude spectrograms
    stft_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    stft_mag = np.abs(stft_complex).astype(np.float32)
    stft_db = librosa.amplitude_to_db(stft_mag, ref=np.max).astype(np.float32)

    # 2. Spectral descriptors from magnitude spectrogram
    centroid = librosa.feature.spectral_centroid(
        S=stft_mag,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    ).squeeze(0).astype(np.float32)

    rolloff = librosa.feature.spectral_rolloff(
        S=stft_mag,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        roll_percent=roll_percent,
    ).squeeze(0).astype(np.float32)

    bandwidth = librosa.feature.spectral_bandwidth(
        S=stft_mag,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    ).squeeze(0).astype(np.float32)

    flatness = librosa.feature.spectral_flatness(
        S=stft_mag,
        n_fft=n_fft,
        hop_length=hop_length,
    ).squeeze(0).astype(np.float32)

    contrast = librosa.feature.spectral_contrast(
        S=stft_mag,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    ).astype(np.float32)

    # 3. Zero-crossing rate (computed from time-domain signal)
    zcr = librosa.feature.zero_crossing_rate(
        y,
        hop_length=hop_length,
    ).squeeze(0).astype(np.float32)

    # Align frame count if zcr differs slightly due to framing
    target_frames = stft_mag.shape[1]
    if len(zcr) != target_frames:
        if len(zcr) > target_frames:
            zcr = zcr[:target_frames]
        else:
            zcr = np.pad(zcr, (0, target_frames - len(zcr)), mode="edge")

    # 4. MFCCs (20 coefficients)
    # Using power spectrogram directly to avoid recomputing STFT
    power_spec = stft_mag**2
    mel_spec = librosa.feature.melspectrogram(
        S=power_spec,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_spec_db = librosa.power_to_db(mel_spec)
    mfcc = librosa.feature.mfcc(
        S=mel_spec_db,
        n_mfcc=n_mfcc,
    ).astype(np.float32)

    if mfcc.shape[1] != target_frames:
        if mfcc.shape[1] > target_frames:
            mfcc = mfcc[:, :target_frames]
        else:
            mfcc = np.pad(
                mfcc,
                ((0, 0), (0, target_frames - mfcc.shape[1])),
                mode="edge",
            )

    return SpectralFeatures(
        stft=stft_mag,
        stft_db=stft_db,
        centroid=centroid,
        rolloff=rolloff,
        bandwidth=bandwidth,
        flatness=flatness,
        contrast=contrast,
        zcr=zcr,
        mfcc=mfcc,
        sample_rate=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        duration_sec=duration_sec,
    )


def extract_spectral_features_from_file(
    path: str | Path,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: str = WINDOW,
    roll_percent: float = ROLLOFF_PERCENT,
    n_mfcc: int = N_MFCC,
) -> SpectralFeatures:
    """Load audio from file and extract all spectral features."""
    y, sr_loaded = load_audio(path, sr=sr)
    return extract_spectral_features(
        y=y,
        sr=sr_loaded,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        roll_percent=roll_percent,
        n_mfcc=n_mfcc,
    )
