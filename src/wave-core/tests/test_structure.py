from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wave_core.analysis.loudness import (
    LoudnessFeatures,
    compute_true_peak_db,
    measure_loudness,
)
from wave_core.analysis.spectral import SAMPLE_RATE
from wave_core.analysis.structure import (
    StructureFeatures,
    analyze_structure,
    analyze_structure_from_file,
    downsample_features,
)


def generate_track_with_clear_drop(
    intro_sec: float = 8.0,
    drop_sec: float = 10.0,
    outro_sec: float = 8.0,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, float, float]:
    """Generate a synthetic electronic track with low intro, explosive drop, and quiet outro."""
    total_sec = intro_sec + drop_sec + outro_sec
    t = np.linspace(0, total_sec, int(sr * total_sec), endpoint=False)
    y = np.zeros_like(t, dtype=np.float32)

    n_intro = int(intro_sec * sr)
    n_drop = int(drop_sec * sr)

    # 1. Intro (soft gentle pad)
    t_intro = t[:n_intro]
    y[:n_intro] = 0.15 * np.sin(2 * np.pi * 220.0 * t_intro)

    # 2. Drop (heavy loud 60 Hz sub bass + rich harmonic synths)
    idx_drop_start = n_intro
    idx_drop_end = n_intro + n_drop
    t_drop = t[idx_drop_start:idx_drop_end]
    y[idx_drop_start:idx_drop_end] = (
        0.70 * np.sin(2 * np.pi * 55.0 * t_drop)
        + 0.20 * np.sin(2 * np.pi * 110.0 * t_drop)
        + 0.10 * np.sin(2 * np.pi * 440.0 * t_drop)
    )

    # 3. Outro (fade-out low synth)
    t_outro = t[idx_drop_end:]
    y[idx_drop_end:] = 0.10 * np.sin(2 * np.pi * 330.0 * t_outro)

    drop_start = intro_sec
    drop_end = intro_sec + drop_sec
    return y.astype(np.float32), drop_start, drop_end


def test_loudness_measurement():
    sr = 22050
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    # Full scale sine wave
    y = (0.99 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    loud = measure_loudness(y, sr=sr)
    assert isinstance(loud, LoudnessFeatures)
    # True peak of 0.99 should be close to 0 dBFS (-0.1 dB)
    assert loud.true_peak_db == pytest.approx(-0.1, abs=0.5)
    # Standard sine wave integrated loudness is around -3 to -4 LUFS
    assert -6.0 <= loud.lufs_integrated <= -2.0
    assert loud.lufs_range >= 0.0


def test_loudness_silence():
    zeros = np.zeros(22050 * 2, dtype=np.float32)
    loud = measure_loudness(zeros)
    assert loud.lufs_integrated <= -60.0
    assert loud.true_peak_db <= -70.0


def test_compute_true_peak_db():
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    sine_half = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    # 20 * log10(0.5) = -6.02 dBFS
    assert compute_true_peak_db(sine_half) == pytest.approx(-6.02, abs=0.2)
    assert compute_true_peak_db(np.array([], dtype=np.float32)) == -80.0


def test_downsample_features():
    # 32 features x 80 frames downsampled by factor 8 -> 32 x 10
    features = np.ones((32, 80), dtype=np.float32)
    features[:, :8] = 2.0
    ds = downsample_features(features, factor=8)

    assert ds.shape == (32, 10)
    assert ds.dtype == np.float32
    assert np.allclose(ds[:, 0], 2.0)
    assert np.allclose(ds[:, 1:], 1.0)


def test_track_with_explicit_drop_structure():
    y, _drop_start, _drop_end = generate_track_with_clear_drop(
        intro_sec=8.0,
        drop_sec=10.0,
        outro_sec=8.0,
    )

    structure = analyze_structure(y, sr=SAMPLE_RATE, k=4)
    assert isinstance(structure, StructureFeatures)
    assert structure.duration_sec == pytest.approx(26.0, abs=0.01)
    assert structure.n_sections >= 3

    sections = structure.sections

    # Coverage: starts at 0.0, ends at duration_sec, strictly contiguous
    assert sections[0].start == 0.0
    assert sections[-1].end == pytest.approx(26.0, abs=0.05)
    for i in range(len(sections) - 1):
        assert sections[i].end == pytest.approx(sections[i + 1].start, abs=0.01)

    # Find the section with maximum energy
    energies = [sec.energy for sec in sections]
    max_idx = int(np.argmax(energies))
    max_section = sections[max_idx]

    # TEST REQUIREMENT: Segment with maximum energy MUST be in the middle, NOT at the beginning!
    assert max_idx > 0, "Max energy segment should not be the first section"
    assert max_idx < len(sections) - 1, "Max energy segment should not be the last section"
    assert max_section.label == "drop"
    assert max_section.energy == pytest.approx(1.0, abs=0.01)

    # Max section should fall inside the synthetic drop region (8s to 18s)
    assert max_section.start >= 6.0
    assert max_section.end <= 20.0

    # First section should be labeled intro
    assert sections[0].label == "intro"
    assert sections[0].energy < 0.5

    # Last section should be labeled outro
    assert sections[-1].label == "outro"
    assert sections[-1].energy < 0.5

    # JSONB format check: [{start, end, label, energy}]
    for sec_dict in structure.sections_json:
        assert "start" in sec_dict
        assert "end" in sec_dict
        assert "label" in sec_dict
        assert "energy" in sec_dict
        assert sec_dict["start"] < sec_dict["end"]
        assert sec_dict["label"] in ("intro", "build", "drop", "breakdown", "outro")


def test_short_track_single_section():
    # Very short track (2 seconds)
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 2, SAMPLE_RATE * 2)).astype(np.float32)
    structure = analyze_structure(y, min_section_sec=2.0)

    assert structure.n_sections == 1
    assert structure.sections[0].start == 0.0
    assert structure.sections[0].end == pytest.approx(2.0, abs=0.01)


def test_analyze_structure_from_file(tmp_path: Path):
    audio_path = tmp_path / "drop_track.wav"
    y, _, _ = generate_track_with_clear_drop(intro_sec=6.0, drop_sec=8.0, outro_sec=6.0)
    sf.write(str(audio_path), y, SAMPLE_RATE)

    features = analyze_structure_from_file(audio_path, k=4)
    assert features.n_sections >= 3
    assert features.lufs_integrated > -40.0
    assert features.true_peak_db > -10.0


def test_invalid_input():
    with pytest.raises(ValueError):
        analyze_structure(np.array([], dtype=np.float32))

    with pytest.raises(ValueError):
        measure_loudness(np.array([], dtype=np.float32))
