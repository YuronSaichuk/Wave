"""Visual and console audio analysis inspection tool ('wave inspect')."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from wave_core.analysis.bands import BAND_NAMES, extract_band_energy
from wave_core.analysis.harmony import extract_harmony_features
from wave_core.analysis.loudness import measure_loudness
from wave_core.analysis.rhythm import extract_rhythm_features
from wave_core.analysis.scanner import parse_filename_fallback
from wave_core.analysis.spectral import (
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    extract_spectral_features,
    load_audio,
)
from wave_core.analysis.structure import Section, analyze_structure


@dataclass(slots=True)
class InspectResult:
    """Comprehensive inspection results of an audio track."""

    file_path: Path
    title: str
    artist: str
    duration_sec: float
    sample_rate: int
    bpm: float
    bpm_confidence: float
    first_beat_sec: float
    key_pitch: str
    key_mode: str
    camelot: str
    key_confidence: float
    lufs_integrated: float
    lufs_range: float
    true_peak_db: float
    sections: list[Section]
    beats: np.ndarray
    downbeats: np.ndarray
    stft_db: np.ndarray
    band_energy: np.ndarray
    rms: np.ndarray
    time_frames: np.ndarray


# Curated colors for the 8 frequency bands
BAND_COLORS: tuple[str, ...] = (
    "#ff3b30",  # sub - red
    "#ff9500",  # bass - orange
    "#ffcc00",  # low_mid - amber
    "#34c759",  # mid - green
    "#5ac8fa",  # high_mid - sky blue
    "#007aff",  # presence - royal blue
    "#af52de",  # brilliance - purple
    "#ffffff",  # air - white
)

# Curated colors for musical structural sections
SECTION_COLORS: dict[str, str] = {
    "intro": "#38bdf8",  # light blue
    "build": "#fbbf24",  # amber
    "drop": "#ef4444",  # red
    "breakdown": "#c084fc",  # purple
    "outro": "#34d399",  # emerald
}


def inspect_audio(path: Path) -> InspectResult:
    """Run full DSP pipeline on audio file and collect all inspection parameters."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Fallback artist/title from filename or tags
    artist_name, title_name = parse_filename_fallback(path)
    title = title_name if title_name else path.stem
    artist = artist_name if artist_name else "Unknown Artist"

    # 1. Load audio
    y, sr = load_audio(path, sr=SAMPLE_RATE)
    duration_sec = float(len(y) / sr)

    # 2. Spectral features (STFT)
    spectral = extract_spectral_features(y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # 3. 8-Band energy
    band_energy = extract_band_energy(spectral.stft, sr=sr, n_fft=N_FFT)

    # 4. Rhythm and beat tracking
    rhythm = extract_rhythm_features(y, sr=sr, hop_length=HOP_LENGTH, sub_energy=band_energy[0])

    # 5. Harmony and Camelot key
    harmony = extract_harmony_features(y, sr=sr, hop_length=HOP_LENGTH)

    # 6. Loudness & Structure
    loudness = measure_loudness(y=y, sr=sr)
    structure = analyze_structure(y=y, sr=sr, hop_length=HOP_LENGTH, loudness_info=loudness)

    # 7. RMS curve over STFT frames
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH).squeeze(0)
    time_frames = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)

    return InspectResult(
        file_path=path,
        title=title,
        artist=artist,
        duration_sec=duration_sec,
        sample_rate=sr,
        bpm=rhythm.bpm,
        bpm_confidence=rhythm.bpm_confidence,
        first_beat_sec=rhythm.first_beat_sec,
        key_pitch=harmony.key_pitch,
        key_mode=harmony.key_mode,
        camelot=harmony.camelot,
        key_confidence=harmony.key_confidence,
        lufs_integrated=structure.lufs_integrated,
        lufs_range=structure.lufs_range,
        true_peak_db=structure.true_peak_db,
        sections=structure.sections,
        beats=rhythm.beats,
        downbeats=rhythm.downbeats,
        stft_db=spectral.stft_db,
        band_energy=band_energy,
        rms=rms,
        time_frames=time_frames,
    )


def print_inspect_console(result: InspectResult, console: Console) -> None:
    """Print formatted inspection report in the terminal with rich tables and panels."""
    mins = int(result.duration_sec // 60)
    secs = int(result.duration_sec % 60)
    time_str = f"{mins}:{secs:02d} ({result.duration_sec:.1f}s)"

    # Metadata Panel
    header_text = (
        f"[bold white]{result.title}[/bold white] — [cyan]{result.artist}[/cyan]\n"
        f"[dim]{result.file_path}[/dim]\n"
        f"Duration: [green]{time_str}[/green] | Sample Rate: [green]{result.sample_rate} Hz[/green]"
    )
    console.print(Panel(header_text, title="Track Inspection", border_style="cyan"))

    # Summary Grid
    summary_table = Table(title="Audio Analysis Descriptors", border_style="blue")
    summary_table.add_column("Property", style="bold cyan")
    summary_table.add_column("Value", style="bold yellow")
    summary_table.add_column("Confidence / Detail", style="dim")

    # BPM row
    bpm_conf_pct = int(result.bpm_confidence * 100)
    conf_style = "green" if result.bpm_confidence >= 0.5 else "red"
    summary_table.add_row(
        "Tempo (BPM)",
        f"{result.bpm:.1f} BPM",
        f"[{conf_style}]Confidence: {bpm_conf_pct}%[/] (First beat: {result.first_beat_sec:.2f}s)",
    )

    # Key row
    key_conf_pct = int(result.key_confidence * 100)
    summary_table.add_row(
        "Key & Camelot",
        f"{result.key_pitch} {result.key_mode} ([magenta]{result.camelot}[/])",
        f"Confidence: {key_conf_pct}%",
    )

    # Loudness row
    summary_table.add_row(
        "Loudness (LUFS)",
        f"{result.lufs_integrated:.1f} LUFS",
        f"LRA: {result.lufs_range:.1f} LU | True Peak: {result.true_peak_db:.1f} dBFS",
    )

    console.print(summary_table)

    # Sections Table
    sec_table = Table(title=f"Structural Sections ({len(result.sections)} sections)", border_style="magenta")
    sec_table.add_column("#", justify="right", style="dim")
    sec_table.add_column("Label", style="bold")
    sec_table.add_column("Start", justify="right")
    sec_table.add_column("End", justify="right")
    sec_table.add_column("Duration", justify="right")
    sec_table.add_column("Energy", justify="right")
    sec_table.add_column("Visual Energy Bar", justify="left")

    for i, sec in enumerate(result.sections, start=1):
        color = SECTION_COLORS.get(sec.label.lower(), "white")
        bar_len = round(sec.energy * 20)
        bar_str = f"[{color}]{'█' * bar_len}[/{color}]{'░' * (20 - bar_len)}"
        sec_table.add_row(
            str(i),
            f"[{color}]{sec.label.upper()}[/{color}]",
            f"{sec.start:.1f}s",
            f"{sec.end:.1f}s",
            f"{sec.duration:.1f}s",
            f"{sec.energy:.2f}",
            bar_str,
        )

    console.print(sec_table)


def plot_inspect_figure(result: InspectResult, out_path: Path) -> None:
    """Generate 3-subplot visual analysis plot and save to PNG image file.

    Subplots:
        1. Spectrogram (dB) with thin beat markers and bold downbeat lines.
        2. 8-band logarithmic energy over time (dB).
        3. RMS energy envelope with highlighted structural section boundaries.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    time_max = result.duration_sec

    # -------------------------------------------------------------
    # 1. Spectrogram (dB) with Beats & Downbeats
    # -------------------------------------------------------------
    freq_max = result.sample_rate // 2
    im = ax1.imshow(
        result.stft_db,
        origin="lower",
        aspect="auto",
        extent=[0, time_max, 0, freq_max],
        cmap="magma",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax1, format="%+2.0f dB", pad=0.015, aspect=15)

    # Beats (thin cyan dashed)
    if len(result.beats) > 0:
        ax1.vlines(
            result.beats,
            ymin=0,
            ymax=freq_max,
            colors="#38bdf8",
            alpha=0.35,
            linewidth=0.8,
            linestyle="--",
            label="Beats",
        )

    # Downbeats (bold yellow solid)
    if len(result.downbeats) > 0:
        ax1.vlines(
            result.downbeats,
            ymin=0,
            ymax=freq_max,
            colors="#facc15",
            alpha=0.85,
            linewidth=1.8,
            label="Downbeats",
        )

    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(
        f"Spectrogram (dB) | Tempo: {result.bpm:.1f} BPM ({int(result.bpm_confidence * 100)}% conf) | "
        f"Key: {result.key_pitch} {result.key_mode} ({result.camelot})",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylim(0, 11025)
    ax1.legend(loc="upper right", framealpha=0.4, fontsize=9)

    # -------------------------------------------------------------
    # 2. 8 Energy Bands in Time
    # -------------------------------------------------------------
    n_frames = result.band_energy.shape[1]
    time_bands = np.linspace(0, time_max, n_frames, endpoint=False)
    for i, (name, color) in enumerate(zip(BAND_NAMES, BAND_COLORS, strict=False)):
        ax2.plot(
            time_bands,
            result.band_energy[i],
            label=name,
            color=color,
            linewidth=1.2,
            alpha=0.85,
        )

    ax2.set_ylabel("Energy (dB)")
    ax2.set_title("8-Band Logarithmic Energy [8 × N]", fontsize=11, fontweight="bold")
    ax2.set_ylim(-80, 5)
    ax2.grid(True, linestyle=":", alpha=0.3)
    ax2.legend(loc="lower right", ncol=4, framealpha=0.4, fontsize=8)

    # -------------------------------------------------------------
    # 3. RMS Envelope with Sections
    # -------------------------------------------------------------
    ax3.plot(
        result.time_frames,
        result.rms,
        color="#f8fafc",
        linewidth=1.2,
        label="RMS Energy",
        zorder=3,
    )

    max_rms = float(np.max(result.rms)) if len(result.rms) > 0 and np.max(result.rms) > 0 else 1.0

    # Highlight each section with background color & text
    for sec in result.sections:
        color = SECTION_COLORS.get(sec.label.lower(), "#64748b")
        ax3.axvspan(sec.start, sec.end, alpha=0.25, color=color, zorder=1)
        ax3.axvline(sec.start, color=color, linestyle=":", alpha=0.6, zorder=2)
        ax3.axvline(sec.end, color=color, linestyle=":", alpha=0.6, zorder=2)

        # Centered section label
        center_x = 0.5 * (sec.start + sec.end)
        ax3.text(
            center_x,
            max_rms * 0.88,
            sec.label.upper(),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=color,
            zorder=4,
        )

    ax3.set_ylabel("RMS Amplitude")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title(
        f"RMS Envelope & Structural Sections ({len(result.sections)} sections) | "
        f"Integrated: {result.lufs_integrated:.1f} LUFS | Peak: {result.true_peak_db:.1f} dBFS",
        fontsize=11,
        fontweight="bold",
    )
    ax3.set_xlim(0, time_max)
    ax3.grid(True, linestyle=":", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
