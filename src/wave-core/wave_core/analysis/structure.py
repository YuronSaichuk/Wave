"""Structural segmentation and song section analysis (intro, build, drop, breakdown, outro)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from wave_core.analysis.loudness import LoudnessFeatures, measure_loudness
from wave_core.analysis.spectral import (
    HOP_LENGTH,
    SAMPLE_RATE,
    load_audio,
)

# Target frame rate for structural analysis (downsampled to prevent O(N^2) explosion)
TARGET_STRUCTURE_FPS: float = 5.0
DEFAULT_K_SEGMENTS: int = 8
MIN_SECTION_DURATION_SEC: float = 2.0


@dataclass(slots=True)
class Section:
    """A detected structural section of a track."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    label: str  # Heuristic role: intro | build | drop | breakdown | outro
    energy: float  # Normalized section energy in [0.0, 1.0]

    @property
    def duration(self) -> float:
        """Section duration in seconds."""
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary for JSONB storage."""
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "label": self.label,
            "energy": round(self.energy, 3),
        }


@dataclass(slots=True)
class StructureFeatures:
    """Complete structural and loudness analysis features."""

    sections: list[Section]
    duration_sec: float
    lufs_integrated: float
    lufs_range: float
    true_peak_db: float

    @property
    def sections_json(self) -> list[dict[str, Any]]:
        """List of dictionary representations of sections ready for database JSONB."""
        return [sec.to_dict() for sec in self.sections]

    @property
    def n_sections(self) -> int:
        """Total number of structural sections."""
        return len(self.sections)


def downsample_features(features: np.ndarray, factor: int = 8) -> np.ndarray:
    """Downsample feature matrix across time via block averaging.

    Args:
        features: 2D array of shape [n_features, n_frames].
        factor: Downsampling factor (default: 8, reducing ~43 Hz to ~5 Hz).

    Returns:
        Downsampled array of shape [n_features, n_frames // factor].
    """
    if factor <= 1 or features.shape[1] <= factor:
        return features

    n_features, n_frames = features.shape
    n_blocks = n_frames // factor
    trimmed = features[:, : n_blocks * factor]
    reshaped = trimmed.reshape(n_features, n_blocks, factor)
    return np.mean(reshaped, axis=2).astype(np.float32)


def assign_section_labels(
    durations: list[tuple[float, float]],
    raw_energies: list[float],
) -> list[Section]:
    """Assign heuristic musical labels (intro/build/drop/breakdown/outro) based on energy profile.

    Heuristics:
        - Segment with peak energy in middle -> 'drop'
        - First segment with lower energy -> 'intro'
        - Last segment with lower energy -> 'outro'
        - Middle segments below median energy -> 'breakdown'
        - Segments with rising energy directly preceding a drop -> 'build'
    """
    n = len(durations)
    if n == 0:
        return []

    # Normalize energies to [0.0, 1.0]
    max_energy = max(raw_energies) if raw_energies else 1.0
    if max_energy > 1e-6:
        norm_energies = [e / max_energy for e in raw_energies]
    else:
        norm_energies = [0.0] * n

    median_energy = float(np.median(norm_energies)) if norm_energies else 0.5
    max_idx = int(np.argmax(norm_energies))

    labels: list[str] = [""] * n

    # 1. Mark peak energy segment as 'drop'
    labels[max_idx] = "drop"

    # Also mark any section with energy >= 0.88 as drop/chorus
    for i in range(n):
        if norm_energies[i] >= 0.88 and i not in (0, n - 1):
            labels[i] = "drop"

    # 2. First segment: intro if not the absolute peak
    if labels[0] == "":
        if norm_energies[0] <= 0.75 or max_idx != 0:
            labels[0] = "intro"
        else:
            labels[0] = "drop"

    # 3. Last segment: outro if not the absolute peak
    if labels[n - 1] == "":
        if norm_energies[n - 1] <= 0.75 or max_idx != n - 1:
            labels[n - 1] = "outro"
        else:
            labels[n - 1] = "drop"

    # 4. Fill middle segments
    for i in range(1, n - 1):
        if labels[i] != "":
            continue

        # Check if preceding a drop with rising energy -> 'build'
        if i + 1 < n and labels[i + 1] == "drop" and norm_energies[i] > norm_energies[i - 1]:
            labels[i] = "build"
        # If energy is below median -> 'breakdown'
        elif norm_energies[i] <= median_energy:
            labels[i] = "breakdown"
        # If energy is rising -> 'build'
        elif i > 0 and norm_energies[i] > norm_energies[i - 1]:
            labels[i] = "build"
        else:
            labels[i] = "breakdown"

    # Build Section instances
    sections: list[Section] = []
    for (start, end), energy, label in zip(durations, norm_energies, labels, strict=False):
        sections.append(
            Section(
                start=round(float(start), 2),
                end=round(float(end), 2),
                label=label if label else "breakdown",
                energy=round(float(energy), 3),
            )
        )

    return sections


def analyze_structure(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    k: int = DEFAULT_K_SEGMENTS,
    min_section_sec: float = MIN_SECTION_DURATION_SEC,
    loudness_info: LoudnessFeatures | None = None,
) -> StructureFeatures:
    """Analyze track structure: segment boundaries, section labeling, and loudness.

    Args:
        y: Mono float32 audio signal.
        sr: Sample rate in Hz.
        hop_length: Hop length in samples.
        k: Desired target number of structural segments (default: 8).
        min_section_sec: Minimum duration in seconds for a section (avoids tiny slivers).
        loudness_info: Optional precomputed LoudnessFeatures. If None, measured via pyloudnorm.

    Returns:
        StructureFeatures dataclass instance.
    """
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Audio signal y must be a non-empty 1D array")

    y = np.ascontiguousarray(y, dtype=np.float32)
    duration_sec = float(len(y) / sr)

    # 1. Loudness measurements
    if loudness_info is None:
        loudness_info = measure_loudness(y=y, sr=sr)

    # If track is extremely short, treat as a single section
    if duration_sec <= min_section_sec * 2:
        sec = Section(start=0.0, end=duration_sec, label="intro", energy=1.0)
        return StructureFeatures(
            sections=[sec],
            duration_sec=duration_sec,
            lufs_integrated=loudness_info.lufs_integrated,
            lufs_range=loudness_info.lufs_range,
            true_peak_db=loudness_info.true_peak_db,
        )

    # 2. Extract MFCC and Chroma for structural recurrence
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=20)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    except (librosa.util.exceptions.ParameterError, ValueError):
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    # Stack MFCC[1:20] (timbre without volume) and Chroma (harmony)
    stacked = np.vstack([mfcc[1:20], chroma])

    # 3. Downsample to ~5 Hz BEFORE computing agglomerative clustering
    fps = sr / hop_length
    ds_factor = max(1, round(fps / TARGET_STRUCTURE_FPS))
    stacked_ds = downsample_features(stacked, factor=ds_factor)

    # 4. Agglomerative clustering on downsampled features
    n_ds_frames = stacked_ds.shape[1]
    effective_k = min(k, max(2, n_ds_frames // 4))

    # Self-similarity recurrence matrix (affinity mode)
    _ = librosa.segment.recurrence_matrix(stacked_ds, mode="affinity", sym=True)
    raw_boundaries = librosa.segment.agglomerative(stacked_ds, k=effective_k)

    # Convert downsampled frame boundaries to timestamps in seconds
    ds_frame_sec = (ds_factor * hop_length) / sr
    raw_times = sorted({float(b * ds_frame_sec) for b in raw_boundaries})

    # Ensure 0.0 and duration_sec are outer boundaries
    all_times: list[float] = [0.0]
    for t in raw_times:
        if t > 0.0 and t < duration_sec:
            all_times.append(t)
    all_times.append(duration_sec)
    all_times.sort()

    # Filter out boundaries that are too close (< min_section_sec)
    filtered_times: list[float] = [all_times[0]]
    for t in all_times[1:-1]:
        if t - filtered_times[-1] >= min_section_sec:
            filtered_times.append(t)
    if duration_sec - filtered_times[-1] < min_section_sec and len(filtered_times) > 1:
        filtered_times[-1] = duration_sec
    else:
        filtered_times.append(duration_sec)

    # Construct spans (start, end)
    spans = [(filtered_times[i], filtered_times[i + 1]) for i in range(len(filtered_times) - 1)]

    # 5. Compute mean RMS energy per segment
    energies: list[float] = []
    for start, end in spans:
        sample_start = int(start * sr)
        sample_end = min(len(y), int(end * sr))
        chunk = y[sample_start:sample_end]
        if len(chunk) > 0:
            rms = float(np.sqrt(np.mean(chunk**2)))
        else:
            rms = 0.0
        energies.append(rms)

    # 6. Assign heuristic musical labels
    sections = assign_section_labels(spans, energies)

    return StructureFeatures(
        sections=sections,
        duration_sec=duration_sec,
        lufs_integrated=loudness_info.lufs_integrated,
        lufs_range=loudness_info.lufs_range,
        true_peak_db=loudness_info.true_peak_db,
    )


def analyze_structure_from_file(
    path: str | Path,
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    k: int = DEFAULT_K_SEGMENTS,
) -> StructureFeatures:
    """Load audio from file and analyze structural sections and loudness."""
    y, sr_loaded = load_audio(path, sr=sr)
    return analyze_structure(y=y, sr=sr_loaded, hop_length=hop_length, k=k)
