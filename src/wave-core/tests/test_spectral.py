from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wave_core.analysis.spectral import (
    HOP_LENGTH,
    N_FFT,
    N_MFCC,
    SAMPLE_RATE,
    SpectralFeatures,
    extract_spectral_features,
    extract_spectral_features_from_file,
    load_audio,
)


def generate_sine_wave(
    freq_hz: float,
    duration_sec: float = 1.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a pure sine wave signal."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (0.7 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def generate_white_noise(
    duration_sec: float = 1.0,
    sr: int = SAMPLE_RATE,
    seed: int = 42,
) -> np.ndarray:
    """Generate normalized white noise signal."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1.0, 1.0, int(sr * duration_sec)).astype(np.float32)
    return noise


def test_spectral_shapes_and_types():
    y = generate_sine_wave(440.0, duration_sec=1.0)
    features = extract_spectral_features(y, sr=SAMPLE_RATE)

    assert isinstance(features, SpectralFeatures)
    assert features.sample_rate == SAMPLE_RATE
    assert features.hop_length == HOP_LENGTH
    assert features.n_fft == N_FFT
    assert features.duration_sec == pytest.approx(1.0, abs=0.01)

    n_frames = features.n_frames
    assert n_frames > 0

    # STFT shape: (1 + n_fft // 2, n_frames) = (1025, n_frames)
    assert features.stft.shape == (1025, n_frames)
    assert features.stft.dtype == np.float32
    assert features.stft_db.shape == (1025, n_frames)
    assert features.stft_db.dtype == np.float32

    # Descriptors shapes and types
    assert features.centroid.shape == (n_frames,)
    assert features.centroid.dtype == np.float32

    assert features.rolloff.shape == (n_frames,)
    assert features.rolloff.dtype == np.float32

    assert features.bandwidth.shape == (n_frames,)
    assert features.bandwidth.dtype == np.float32

    assert features.flatness.shape == (n_frames,)
    assert features.flatness.dtype == np.float32

    # Spectral contrast: 7 sub-bands by default
    assert features.contrast.shape == (7, n_frames)
    assert features.contrast.dtype == np.float32

    assert features.zcr.shape == (n_frames,)
    assert features.zcr.dtype == np.float32

    # MFCC: 20 coefficients
    assert features.mfcc.shape == (N_MFCC, n_frames)
    assert features.mfcc.dtype == np.float32


def test_sine_wave_centroid_and_flatness():
    # 440 Hz sine wave: centroid should be ~440 Hz, flatness close to 0
    y = generate_sine_wave(440.0, duration_sec=1.5)
    features = extract_spectral_features(y, sr=SAMPLE_RATE)

    # Exclude boundary frames where Hann window tapers to 0
    stable_frames = slice(5, -5)
    mean_centroid = float(np.mean(features.centroid[stable_frames]))
    mean_flatness = float(np.mean(features.flatness[stable_frames]))

    # Centroid should be within 15 Hz of 440 Hz
    assert 425.0 <= mean_centroid <= 455.0
    # Flatness of pure sine is near 0 (< 0.01)
    assert mean_flatness < 0.01


def test_white_noise_flatness():
    # White noise: flatness is high (> 0.5 with Hann window) compared to pure sine (< 0.01)
    y = generate_white_noise(duration_sec=1.5)
    features = extract_spectral_features(y, sr=SAMPLE_RATE)

    stable_frames = slice(5, -5)
    mean_flatness = float(np.mean(features.flatness[stable_frames]))

    # For windowed STFT of finite white noise, flatness is ~0.55-0.65
    assert mean_flatness > 0.50


def test_extract_spectral_features_from_file(tmp_path: Path):
    audio_path = tmp_path / "test_tone.wav"
    y_orig = generate_sine_wave(880.0, duration_sec=1.0)
    sf.write(str(audio_path), y_orig, SAMPLE_RATE)

    features = extract_spectral_features_from_file(audio_path)
    assert features.duration_sec == pytest.approx(1.0, abs=0.02)

    stable_frames = slice(5, -5)
    mean_centroid = float(np.mean(features.centroid[stable_frames]))
    assert 860.0 <= mean_centroid <= 900.0


def test_load_audio_mono_conversion(tmp_path: Path):
    audio_path = tmp_path / "stereo.wav"
    # Create 2-channel stereo audio
    ch1 = generate_sine_wave(440.0, duration_sec=0.5)
    ch2 = generate_sine_wave(880.0, duration_sec=0.5)
    stereo = np.column_stack([ch1, ch2])
    sf.write(str(audio_path), stereo, SAMPLE_RATE)

    y_loaded, sr = load_audio(audio_path)
    assert sr == SAMPLE_RATE
    assert y_loaded.ndim == 1
    assert y_loaded.dtype == np.float32


def test_invalid_input():
    with pytest.raises(ValueError):
        extract_spectral_features(np.array([], dtype=np.float32))

    with pytest.raises(ValueError):
        extract_spectral_features(np.zeros((2, 100), dtype=np.float32))
