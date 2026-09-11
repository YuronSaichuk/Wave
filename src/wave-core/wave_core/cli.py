import sys
import typer
from rich.console import Console
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


if __name__ == "__main__":
    app()
