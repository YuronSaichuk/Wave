from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wave_core.analysis.rhythm import (
    RhythmFeatures,
    correct_tempo_octave,
    detect_downbeats,
    extract_rhythm_features,
    extract_rhythm_features_from_file,
)
from wave_core.analysis.spectral import SAMPLE_RATE


def generate_metronome(
    bpm: float = 120.0,
    duration_sec: float = 8.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a clean synthetic metronome with clicks."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = np.zeros_like(t, dtype=np.float32)

    beat_interval = 60.0 / bpm
    click_len = int(sr * 0.02)  # 20 ms click
    click_t = np.linspace(0, 0.02, click_len, endpoint=False)
    # Click with high transient energy across spectrum
    click = (
        0.5 * np.sin(2 * np.pi * 150.0 * click_t)  # low punch
        + 0.5 * np.sin(2 * np.pi * 1200.0 * click_t)  # mid/high click
    ) * np.hanning(click_len)
    click = click.astype(np.float32)

    current_t = 0.0
    while current_t + 0.02 < duration_sec:
        idx = int(current_t * sr)
        end_idx = min(len(y), idx + click_len)
        y[idx:end_idx] += click[: end_idx - idx]
        current_t += beat_interval

    return y


def test_correct_tempo_octave():
    # In-range tempos should not change
    assert correct_tempo_octave(120.0) == 120.0
    assert correct_tempo_octave(128.0) == 128.0
    assert correct_tempo_octave(70.0) == 70.0
    assert correct_tempo_octave(180.0) == 180.0

    # Low tempos (< 70) should be doubled
    assert correct_tempo_octave(60.0) == 120.0
    assert correct_tempo_octave(64.0) == 128.0
    assert correct_tempo_octave(32.5) == 130.0

    # High tempos (> 180) should be halved
    assert correct_tempo_octave(190.0) == 95.0
    assert correct_tempo_octave(240.0) == 120.0
    assert correct_tempo_octave(350.0) == 175.0
    assert correct_tempo_octave(400.0) == 100.0

    # Edge cases
    assert correct_tempo_octave(0.0) == 0.0
    assert correct_tempo_octave(-10.0) == 0.0


def test_metronome_120_bpm():
    y = generate_metronome(bpm=120.0, duration_sec=8.0)
    features = extract_rhythm_features(y)

    assert isinstance(features, RhythmFeatures)
    # Expected: 120 BPM ± 0.5 BPM
    assert features.bpm == pytest.approx(120.0, abs=0.5)
    assert features.bpm_confidence > 0.5

    # Number of beats in 8 seconds at 120 BPM (~16 beats)
    assert 14 <= features.n_beats <= 18
    assert features.beats.dtype == np.float32

    # Beat intervals should average 0.5s ± 0.02s
    assert features.beat_interval_sec == pytest.approx(0.5, abs=0.03)

    # Downbeats (grouped by 4 beats -> every ~2.0s)
    assert features.n_downbeats >= 3
    assert features.downbeats.dtype == np.float32
    assert features.downbeat_indices.dtype in (np.int64, np.int32)
    assert len(features.downbeats) == len(features.downbeat_indices)


def test_downbeat_selection_by_sub_energy():
    # 8 beats at t = 0, 1, 2, 3, 4, 5, 6, 7
    beats = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)

    # Create sub_energy frames: 8 seconds at fps ~ 43.066
    sr = 22050
    hop = 512
    fps = sr / hop
    n_frames = int(8.0 * fps)
    sub_energy = np.zeros(n_frames, dtype=np.float32)

    # Put heavy sub kick on beat 1 and beat 5 (k = 1)
    beat_1_frame = round(1.0 * fps)
    beat_5_frame = round(5.0 * fps)
    sub_energy[beat_1_frame] = 10.0
    sub_energy[beat_5_frame] = 10.0

    # For beats 0, 2, 3, 4, 6, 7 put low energy
    for b in [0.0, 2.0, 3.0, 4.0, 6.0, 7.0]:
        sub_energy[round(b * fps)] = 1.0

    downbeats, indices = detect_downbeats(
        beats=beats,
        sub_energy=sub_energy,
        sr=sr,
        hop_length=hop,
        group_size=4,
    )

    # Offset k=1 should be chosen because beats 1 and 5 have maximum sub energy
    assert list(indices) == [1, 5]
    assert np.allclose(downbeats, [1.0, 5.0])


def test_confidence_low_on_ambient_tone():
    # Constant pure tone has no rhythmic pulse -> low confidence
    t = np.linspace(0, 4, 4 * SAMPLE_RATE, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    features = extract_rhythm_features(tone)
    # Ambience / drones should have confidence < 0.5
    assert features.bpm_confidence < 0.5


def test_extract_rhythm_features_from_file(tmp_path: Path):
    audio_path = tmp_path / "metronome_120.wav"
    y = generate_metronome(bpm=120.0, duration_sec=8.0)
    sf.write(str(audio_path), y, SAMPLE_RATE)

    features = extract_rhythm_features_from_file(audio_path)
    assert features.bpm == pytest.approx(120.0, abs=0.5)
    assert features.bpm_confidence > 0.5
    assert features.n_beats > 0


def test_invalid_input():
    with pytest.raises(ValueError):
        extract_rhythm_features(np.array([], dtype=np.float32))

    with pytest.raises(ValueError):
        extract_rhythm_features(np.zeros((2, 100), dtype=np.float32))
