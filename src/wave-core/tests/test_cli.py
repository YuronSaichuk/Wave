from typer.testing import CliRunner
from wave_core.cli import app
import wave_core

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert wave_core.__version__ in result.stdout
