from pathlib import Path
import sys
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import wave_core
from wave_core.analysis.scanner import scan_directory
from wave_core.config import settings
from wave_core.storage.db import check_db
from wave_core.storage.migrations import run_migrations

app = typer.Typer(
    name="wave",
    help="Wave audio engine CLI",
    add_completion=False,
)
console = Console()


@app.command()
def version():
    """Print the version of wave-core."""
    console.print(f"[bold cyan]Wave Core[/bold cyan] version [green]{wave_core.__version__}[/green]")


@app.command()
def migrate():
    """Run database migrations from migrations/ folder."""
    try:
        console.print(f"[yellow]Applying migrations to:[/yellow] {settings.db}")
        applied = run_migrations()
        if applied:
            for item in applied:
                console.print(f"  [green]✓[/green] Applied: {item}")
        else:
            console.print("  [blue]Database schema is up to date.[/blue]")
    except Exception as e:
        console.print(f"[red]Migration failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("db-check")
def db_check():
    """Verify database connection, pgvector extension, and public tables."""
    try:
        console.print(f"[cyan]Checking connection to:[/cyan] {settings.db}")
        result = check_db()
        console.print("[green]✓[/green] Connected successfully!")
        console.print(f"  pgvector extension: [cyan]{result.get('pgvector_version', 'Not installed')}[/cyan]")

        tables = result.get("tables", [])
        if tables:
            table = Table(title="Existing Tables")
            table.add_column("Table Name", style="magenta")
            for t in tables:
                table.add_row(t)
            console.print(table)
        else:
            console.print("[yellow]No public tables found. Run `wave migrate` to initialize.[/yellow]")
    except Exception as e:
        console.print(f"[red]Database check failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("scan")
def scan(
    path: Path | None = typer.Argument(
        None,
        help="Directory to scan for audio files (defaults to WAVE_LIBRARY).",
    ),
):
    """Scan directory recursively for audio files and upsert them into the database."""
    target_dir = path if path is not None else settings.library
    if not target_dir.exists() or not target_dir.is_dir():
        console.print(f"[red]Error:[/red] Directory does not exist: {target_dir}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Scanning audio files at:[/cyan] {target_dir.resolve()}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Scanning tracks...", total=None)

        def on_progress(file_path: Path, current: int, total: int):
            progress.update(
                task,
                total=total,
                completed=current,
                description=f"[green]Processing ({current}/{total}):[/green] {file_path.name}",
            )

        summary = scan_directory(target_dir, progress_callback=on_progress)

    table = Table(title="Scan Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Total Files Found", str(summary.total_found))
    table.add_row("New Tracks Inserted", str(summary.inserted))
    table.add_row("Hash Changed (Pending Re-analysis)", str(summary.hash_changed))
    table.add_row("Metadata Updated", str(summary.metadata_updated))
    table.add_row("Unchanged", str(summary.unchanged))
    table.add_row(
        "Errors",
        str(len(summary.errors)),
        style="red" if summary.errors else "green",
    )

    console.print(table)

    if summary.errors:
        console.print("[yellow]Warnings/Errors encountered during scan:[/yellow]")
        for err_path, err_msg in summary.errors[:10]:
            console.print(f"  [red]×[/red] {err_path}: {err_msg}")
        if len(summary.errors) > 10:
            console.print(f"  ...and {len(summary.errors) - 10} more errors.")


if __name__ == "__main__":
    app()
