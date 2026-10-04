from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wave_core.analysis.bands import (
    BAND_NAMES,
    BANDS_HZ,
    NUM_BANDS,
    BandEnergies,
    create_band_filterbank,
    extract_band_energy,
    extract_band_energy_from_audio,
    extract_band_energy_from_file,
)
from wave_core.analysis.spectral import (
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    extract_spectral_features,
)


def generate_sine(
    freq_hz: float,
    duration_sec: float = 1.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a mono float32 sine wave."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (0.8 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_filterbank_dimensions_and_properties():
    fb = create_band_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, bands=BANDS_HZ)
    expected_bins = 1 + N_FFT // 2  # 1025

    assert fb.shape == (NUM_BANDS, expected_bins)
    assert fb.dtype == np.float32

    # Every band must cover at least one frequency bin
    bins_per_band = np.sum(fb > 0, axis=1)
    assert np.all(bins_per_band >= 4)

    # Sub-band 0 (20-60 Hz) should cover low bins
    sub_bins = np.where(fb[0] > 0)[0]
    assert np.min(sub_bins) >= 1  # 0 Hz (DC) is excluded (< 20 Hz)


def test_sub_band_max_for_50hz_sine():
    # 50 Hz sine wave must have maximum energy in 'sub' band (index 0)
    y = generate_sine(50.0, duration_sec=1.0)
    features = extract_spectral_features(y)

    bands = extract_band_energy(features.stft, sr=SAMPLE_RATE, n_fft=N_FFT)

    assert bands.shape == (8, features.n_frames)
    assert bands.dtype == np.float32

    mean_per_band = np.mean(bands, axis=1)
    assert int(np.argmax(mean_per_band)) == 0  # sub band is index 0
    assert mean_per_band[0] > mean_per_band[1]  # sub > bass
    assert mean_per_band[0] > -5.0  # close to 0 dB peak


@pytest.mark.parametrize(
    ("freq_hz", "expected_idx", "expected_name"),
    [
        (50.0, 0, "sub"),
        (100.0, 1, "bass"),
        (250.0, 2, "low_mid"),
        (600.0, 3, "mid"),
        (1600.0, 4, "high_mid"),
        (3500.0, 5, "presence"),
        (6500.0, 6, "brilliance"),
        (9500.0, 7, "air"),
    ],
)
def test_all_individual_bands_frequency_detection(
    freq_hz: float,
    expected_idx: int,
    expected_name: str,
):
    y = generate_sine(freq_hz, duration_sec=1.0)
    matrix = extract_band_energy_from_audio(y, sr=SAMPLE_RATE)

    assert matrix.shape[0] == 8
    mean_energies = np.mean(matrix, axis=1)
    best_idx = int(np.argmax(mean_energies))

    assert best_idx == expected_idx
    assert BAND_NAMES[best_idx] == expected_name


def test_silence_handling():
    zeros = np.zeros((1025, 50), dtype=np.float32)
    bands = extract_band_energy(zeros, sr=SAMPLE_RATE, n_fft=N_FFT, top_db=80.0)

    assert bands.shape == (8, 50)
    assert bands.dtype == np.float32
    assert not np.isnan(bands).any()
    assert not np.isinf(bands).any()
    # All values should be clipped at -top_db
    assert np.allclose(bands, -80.0)


def test_smoothing_effect():
    y = generate_sine(100.0, duration_sec=1.0)
    features = extract_spectral_features(y)

    unsmoothed = extract_band_energy(features.stft, smooth_window=1)
    smoothed = extract_band_energy(features.stft, smooth_window=5)

    assert unsmoothed.shape == smoothed.shape
    # Both are float32
    assert unsmoothed.dtype == np.float32
    assert smoothed.dtype == np.float32


def test_band_energies_container():
    matrix = np.full((8, 100), -20.0, dtype=np.float32)
    matrix[0, :] = -5.0

    container = BandEnergies(
        matrix=matrix,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    assert container.n_bands == 8
    assert container.n_frames == 100
    assert np.allclose(container.get_band("sub"), -5.0)

    vec = container.mean_vector()
    assert vec.shape == (8,)
    assert vec.dtype == np.float32
    assert vec[0] == pytest.approx(-5.0)

    with pytest.raises(KeyError):
        container.get_band("ultrasound")


def test_extract_band_energy_from_file(tmp_path: Path):
    audio_path = tmp_path / "bass_kick.wav"
    y = generate_sine(55.0, duration_sec=1.0)
    sf.write(str(audio_path), y, SAMPLE_RATE)

    bands = extract_band_energy_from_file(audio_path)
    assert bands.shape[0] == 8
    assert bands.shape[1] > 0
    assert bands.dtype == np.float32
    assert int(np.argmax(np.mean(bands, axis=1))) == 0
