"""Harmony analysis: chroma CQT, Krumhansl-Schmuckler key detection, Camelot wheel mapping."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from wave_core.analysis.spectral import (
    HOP_LENGTH,
    SAMPLE_RATE,
    load_audio,
)

# Canonical 12 pitch class names starting from C
PITCH_NAMES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

# Krumhansl-Schmuckler key profiles (Krumhansl & Kessler, 1982)
# Relative pitch weights from tonic (0: tonic, 1: m2, 2: M2, ..., 11: M7)
KRUMHANSL_MAJOR: tuple[float, ...] = (
    6.35,
    2.23,
    3.48,
    2.33,
    4.38,
    4.09,
    2.52,
    5.19,
    2.39,
    3.66,
    2.29,
    2.88,
)
KRUMHANSL_MINOR: tuple[float, ...] = (
    6.33,
    2.68,
    3.52,
    5.38,
    2.60,
    3.53,
    2.54,
    4.75,
    3.98,
    2.69,
    3.34,
    3.17,
)

# Camelot Wheel mapping: (pitch_class, mode) -> Camelot notation
# Standard Circle of Fifths numbering: 1-12, 'B' = major, 'A' = minor
CAMELOT_MAP: dict[tuple[int, str], str] = {
    # Major keys (Camelot B)
    (0, "major"): "8B",  # C
    (1, "major"): "3B",  # C# / Db
    (2, "major"): "10B",  # D
    (3, "major"): "5B",  # D# / Eb
    (4, "major"): "12B",  # E
    (5, "major"): "7B",  # F
    (6, "major"): "2B",  # F# / Gb
    (7, "major"): "9B",  # G
    (8, "major"): "4B",  # G# / Ab
    (9, "major"): "11B",  # A
    (10, "major"): "6B",  # A# / Bb
    (11, "major"): "1B",  # B
    # Minor keys (Camelot A)
    (0, "minor"): "5A",  # C minor
    (1, "minor"): "12A",  # C# minor
    (2, "minor"): "7A",  # D minor
    (3, "minor"): "2A",  # D# minor
    (4, "minor"): "9A",  # E minor
    (5, "minor"): "4A",  # F minor
    (6, "minor"): "11A",  # F# minor
    (7, "minor"): "6A",  # G minor
    (8, "minor"): "1A",  # G# minor
    (9, "minor"): "8A",  # A minor
    (10, "minor"): "3A",  # A# minor
    (11, "minor"): "10A",  # B minor
}


@dataclass(slots=True)
class HarmonyFeatures:
    """Harmony and tonal analysis results."""

    key_pitch: str  # e.g. "C", "A", "F#"
    key_mode: str  # "major" or "minor"
    camelot: str  # e.g. "8B", "8A", "2B"
    key_confidence: float  # (top1 - top2) / top1 in [0.0, 1.0]
    pitch_class: int  # 0..11
    chroma: np.ndarray  # shape [12, N], float32 (CQT chromagram)
    chroma_mean: np.ndarray  # shape [12], float32 (mean chroma profile)
    sample_rate: int = SAMPLE_RATE
    hop_length: int = HOP_LENGTH

    @property
    def key_name(self) -> str:
        """Full key name, e.g. 'C major', 'A minor'."""
        return f"{self.key_pitch} {self.key_mode}"


def pitch_class_to_name(pitch_class: int) -> str:
    """Return pitch name string for a pitch class index (0..11)."""
    return PITCH_NAMES[pitch_class % 12]


def estimate_key_from_chroma(
    chroma: np.ndarray,
) -> tuple[int, str, str, str, float, dict[str, float]]:
    """Determine musical key using Krumhansl-Schmuckler correlation.

    Args:
        chroma: Chromagram of shape [12, n_frames].

    Returns:
        Tuple of (pitch_class, pitch_name, mode, camelot, confidence, correlation_scores).
    """
    if chroma.shape[0] != 12:
        raise ValueError(f"Chromagram must have 12 pitch classes, got {chroma.shape[0]}")

    chroma_mean = np.mean(chroma, axis=1).astype(np.float64)
    total_energy = float(np.sum(chroma_mean))

    # Silence or near-silence fallback
    if total_energy <= 1e-6 or np.std(chroma_mean) <= 1e-6:
        return 0, "C", "major", "8B", 0.0, {}

    major_template = np.array(KRUMHANSL_MAJOR, dtype=np.float64)
    minor_template = np.array(KRUMHANSL_MINOR, dtype=np.float64)

    scores: dict[tuple[int, str], float] = {}
    named_scores: dict[str, float] = {}

    for i in range(12):
        # Shift profile so that index i corresponds to the tonic
        r_maj = float(np.corrcoef(chroma_mean, np.roll(major_template, i))[0, 1])
        r_min = float(np.corrcoef(chroma_mean, np.roll(minor_template, i))[0, 1])

        scores[(i, "major")] = r_maj
        scores[(i, "minor")] = r_min

        named_scores[f"{PITCH_NAMES[i]} major"] = r_maj
        named_scores[f"{PITCH_NAMES[i]} minor"] = r_min

    # Sort candidates by correlation descending
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_candidate, best_score = ranked[0]
    _, second_score = ranked[1]

    pitch_class, mode = best_candidate
    pitch_name = PITCH_NAMES[pitch_class]
    camelot = CAMELOT_MAP.get((pitch_class, mode), "8B")

    # key_confidence = (first - second) / first
    if best_score > 1e-6:
        confidence = (best_score - second_score) / best_score
        confidence = float(np.clip(confidence, 0.0, 1.0))
    else:
        confidence = 0.0

    return pitch_class, pitch_name, mode, camelot, confidence, named_scores


def extract_harmony_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> HarmonyFeatures:
    """Extract chromagram, key, mode, Camelot code, and confidence from audio.

    Args:
        y: Mono float32 audio signal.
        sr: Sample rate in Hz (default: 22050).
        hop_length: STFT hop length (default: 512).

    Returns:
        HarmonyFeatures dataclass instance.
    """
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Audio signal y must be a non-empty 1D array")

    y = np.ascontiguousarray(y, dtype=np.float32)

    # Compute CQT chromagram (constant-Q logarithmic frequency scale)
    # If audio is very short, fallback safely to chroma_stft
    try:
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=hop_length,
        ).astype(np.float32)
    except (librosa.util.exceptions.ParameterError, ValueError):
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            hop_length=hop_length,
        ).astype(np.float32)

    pitch_class, pitch_name, mode, camelot, confidence, _ = estimate_key_from_chroma(chroma)
    chroma_mean = np.mean(chroma, axis=1).astype(np.float32)

    return HarmonyFeatures(
        key_pitch=pitch_name,
        key_mode=mode,
        camelot=camelot,
        key_confidence=confidence,
        pitch_class=pitch_class,
        chroma=chroma,
        chroma_mean=chroma_mean,
        sample_rate=sr,
        hop_length=hop_length,
    )


def extract_harmony_features_from_file(
    path: str | Path,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> HarmonyFeatures:
    """Load audio from file and extract harmony and key features."""
    y, sr_loaded = load_audio(path, sr=sr)
    return extract_harmony_features(
        y=y,
        sr=sr_loaded,
        hop_length=hop_length,
    )
