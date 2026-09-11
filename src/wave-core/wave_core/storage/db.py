import logging
from contextlib import contextmanager
from typing import Generator

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from wave_core.config import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def configure_connection(conn: psycopg.Connection) -> None:
    """Register pgvector types on newly acquired connection."""
    register_vector(conn)


def get_pool() -> ConnectionPool:
    """Get or initialize the psycopg connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.db,
            min_size=1,
            max_size=10,
            open=True,
            configure=configure_connection,
        )
    return _pool


def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Context manager for acquiring a connection from the pool."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def check_db() -> dict:
    """Check database connection, pgvector extension, and list tables."""
    with psycopg.connect(settings.db) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()

            # Check pgvector
            cur.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
            )
            vec_row = cur.fetchone()
            vector_version = vec_row[0] if vec_row else None

            # Get tables
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
                """
            )
            tables = [row[0] for row in cur.fetchall()]

    return {
        "status": "ok",
        "pgvector_version": vector_version,
        "tables": tables,
    }
