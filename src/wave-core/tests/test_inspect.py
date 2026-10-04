from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from rich.console import Console
from typer.testing import CliRunner

from wave_core.analysis.inspect import (
    InspectResult,
    inspect_audio,
    plot_inspect_figure,
    print_inspect_console,
)
from wave_core.analysis.spectral import SAMPLE_RATE
from wave_core.cli import app

runner = CliRunner()


def generate_sample_audio(
    duration_sec: float = 6.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a simple audio track with tone and rhythm clicks for test inspection."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Sine chord tone
    y = 0.4 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 554.37 * t)
    # Add regular clicks every 0.5s (120 BPM)
    beat_step = int(0.5 * sr)
    for idx in range(0, len(y), beat_step):
        click_end = min(len(y), idx + 200)
        y[idx:click_end] += 0.5
    return y.astype(np.float32)


def test_inspect_audio_pipeline(tmp_path: Path):
    audio_path = tmp_path / "Artist Name - Cool Song.wav"
    y = generate_sample_audio(duration_sec=6.0)
    sf.write(str(audio_path), y, SAMPLE_RATE)

    result = inspect_audio(audio_path)
    assert isinstance(result, InspectResult)
    assert result.file_path == audio_path
    assert result.artist == "Artist Name"
    assert result.title == "Cool Song"
    assert result.duration_sec == pytest.approx(6.0, abs=0.05)
    assert result.sample_rate == SAMPLE_RATE
    assert result.bpm > 0
    assert result.key_pitch in ("A", "C#", "C", "E")
    assert result.camelot is not None
    assert len(result.sections) >= 1
    assert result.stft_db.shape[0] == 1025
    assert result.band_energy.shape[0] == 8
    assert len(result.rms) > 0


def test_print_inspect_console(tmp_path: Path):
    audio_path = tmp_path / "test.wav"
    y = generate_sample_audio(duration_sec=4.0)
    sf.write(str(audio_path), y, SAMPLE_RATE)

    result = inspect_audio(audio_path)
    console = Console(record=True, width=120)
    print_inspect_console(result, console=console)
    output = console.export_text()

    assert "Track Inspection" in output
    assert "Audio Analysis Descriptors" in output
    assert "Tempo (BPM)" in output
    assert "Key & Camelot" in output
    assert "Loudness (LUFS)" in output
    assert "Structural Sections" in output


def test_plot_inspect_figure(tmp_path: Path):
    audio_path = tmp_path / "test_plot_audio.wav"
    out_plot = tmp_path / "output_plot.png"
    y = generate_sample_audio(duration_sec=5.0)
    sf.write(str(audio_path), y, SAMPLE_RATE)

    result = inspect_audio(audio_path)
    plot_inspect_figure(result, out_path=out_plot)

    assert out_plot.exists()
    assert out_plot.stat().st_size > 10000  # Non-empty PNG
    # Check PNG magic bytes: \x89PNG\r\n\x1a\n
    header = out_plot.read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n"


def test_cli_inspect_direct_file(tmp_path: Path):
    audio_path = tmp_path / "Direct Track.wav"
    plot_path = tmp_path / "cli_plot.png"
    y = generate_sample_audio(duration_sec=5.0)
    sf.write(str(audio_path), y, SAMPLE_RATE)

    # 1. Run inspect command without plot
    res = runner.invoke(app, ["inspect", str(audio_path)])
    assert res.exit_code == 0
    assert "Track Inspection" in res.stdout
    assert "Direct Track" in res.stdout

    # 2. Run inspect command with --plot
    res_plot = runner.invoke(app, ["inspect", str(audio_path), "--plot", str(plot_path)])
    assert res_plot.exit_code == 0
    assert "Plot successfully saved to" in res_plot.stdout
    assert plot_path.exists()
    assert plot_path.stat().st_size > 10000


def test_cli_inspect_invalid_target():
    res = runner.invoke(app, ["inspect", "non_existent_file_xyz_123.mp3"])
    assert res.exit_code != 0
    assert "Error" in res.stdout
