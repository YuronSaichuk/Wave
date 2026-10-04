from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wave_core.analysis.harmony import (
    CAMELOT_MAP,
    PITCH_NAMES,
    HarmonyFeatures,
    extract_harmony_features,
    extract_harmony_features_from_file,
    pitch_class_to_name,
)
from wave_core.analysis.spectral import SAMPLE_RATE


def generate_chord(
    frequencies: list[float],
    duration_sec: float = 3.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a clean synthetic polyphonic chord signal."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = np.zeros_like(t, dtype=np.float32)
    for freq in frequencies:
        y += np.sin(2 * np.pi * freq * t, dtype=np.float32)
    # Normalize with headroom
    y = 0.7 * (y / len(frequencies))
    return y.astype(np.float32)


def test_camelot_map_completeness():
    # Exactly 24 keys (12 major 'B', 12 minor 'A')
    assert len(CAMELOT_MAP) == 24

    major_camelots = {CAMELOT_MAP[(i, "major")] for i in range(12)}
    minor_camelots = {CAMELOT_MAP[(i, "minor")] for i in range(12)}

    expected_majors = {f"{n}B" for n in range(1, 13)}
    expected_minors = {f"{n}A" for n in range(1, 13)}

    assert major_camelots == expected_majors
    assert minor_camelots == expected_minors

    # Specific anchor pairs
    assert CAMELOT_MAP[(0, "major")] == "8B"  # C major
    assert CAMELOT_MAP[(9, "minor")] == "8A"  # A minor
    assert CAMELOT_MAP[(7, "major")] == "9B"  # G major
    assert CAMELOT_MAP[(4, "minor")] == "9A"  # E minor
    assert CAMELOT_MAP[(6, "major")] == "2B"  # F# major
    assert CAMELOT_MAP[(3, "minor")] == "2A"  # D# minor


def test_c_major_chord_detection():
    # C major triad: C4 (261.63 Hz), E4 (329.63 Hz), G4 (392.00 Hz)
    y = generate_chord([261.63, 329.63, 392.00], duration_sec=3.0)
    features = extract_harmony_features(y)

    assert isinstance(features, HarmonyFeatures)
    assert features.key_pitch == "C"
    assert features.key_mode == "major"
    assert features.camelot == "8B"
    assert features.pitch_class == 0
    assert features.key_confidence > 0.1
    assert features.key_name == "C major"
    assert features.chroma.shape[0] == 12
    assert features.chroma.dtype == np.float32


def test_a_minor_chord_detection():
    # A minor triad: A3 (220.00 Hz), C4 (261.63 Hz), E4 (329.63 Hz)
    y = generate_chord([220.00, 261.63, 329.63], duration_sec=3.0)
    features = extract_harmony_features(y)

    assert features.key_pitch == "A"
    assert features.key_mode == "minor"
    assert features.camelot == "8A"
    assert features.pitch_class == 9
    assert features.key_confidence > 0.1
    assert features.key_name == "A minor"


def test_f_sharp_major_chord_detection():
    # F# major triad: F#4 (369.99 Hz), A#4 (466.16 Hz), C#5 (554.37 Hz)
    y = generate_chord([369.99, 466.16, 554.37], duration_sec=3.0)
    features = extract_harmony_features(y)

    assert features.key_pitch == "F#"
    assert features.key_mode == "major"
    assert features.camelot == "2B"
    assert features.pitch_class == 6


def test_pitch_class_to_name_helper():
    assert pitch_class_to_name(0) == "C"
    assert pitch_class_to_name(9) == "A"
    assert pitch_class_to_name(6) == "F#"
    assert pitch_class_to_name(12) == "C"
    assert len(PITCH_NAMES) == 12


def test_silence_handling():
    zeros = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    features = extract_harmony_features(zeros)

    assert features.key_confidence == 0.0
    assert features.camelot in ("8B", "8A")


def test_extract_harmony_features_from_file(tmp_path: Path):
    audio_path = tmp_path / "chord_c_maj.wav"
    y = generate_chord([261.63, 329.63, 392.00], duration_sec=3.0)
    sf.write(str(audio_path), y, SAMPLE_RATE)

    features = extract_harmony_features_from_file(audio_path)
    assert features.key_pitch == "C"
    assert features.key_mode == "major"
    assert features.camelot == "8B"


def test_invalid_input():
    with pytest.raises(ValueError):
        extract_harmony_features(np.array([], dtype=np.float32))

    with pytest.raises(ValueError):
        extract_harmony_features(np.zeros((2, 100), dtype=np.float32))
