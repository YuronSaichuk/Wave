import logging
from pathlib import Path

import psycopg

from wave_core.config import settings

logger = logging.getLogger(__name__)


def get_migrations_dir() -> Path:
    """Find the migrations directory relative to the package or repo."""
    # Try relative to src/wave-core/migrations
    current = Path(__file__).resolve().parent
    for parent in [current, current.parent, current.parent.parent, current.parent.parent.parent]:
        candidate = parent / "migrations"
        if candidate.is_dir():
            return candidate
        candidate_src = parent / "src" / "wave-core" / "migrations"
        if candidate_src.is_dir():
            return candidate_src
    raise FileNotFoundError("Could not locate migrations directory.")


def run_migrations() -> list[str]:
    """Execute all unapplied .sql migrations in alphabetical order."""
    migrations_dir = get_migrations_dir()
    sql_files = sorted(migrations_dir.glob("*.sql"))
    applied: list[str] = []

    with psycopg.connect(settings.db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version text PRIMARY KEY,
                    applied_at timestamptz NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute("SELECT version FROM schema_migrations;")
            already_applied = {row[0] for row in cur.fetchall()}

        for sql_file in sql_files:
            version = sql_file.name
            if version in already_applied:
                continue

            logger.info("Applying migration: %s", version)
            content = sql_file.read_text(encoding="utf-8")

            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(content)
                    cur.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s);",
                        (version,),
                    )
            applied.append(version)

    return applied
