"""The application's database: one SQLite file, versioned migrations, durable writes, backups.

SQLite is enough for one person's records and keeps everything in one file
that can be copied. The connection runs in WAL mode with full synchronous
writes, so a committed transaction survives a crash or power loss, and with
foreign keys on. Every write goes through ``transaction()``, which commits all
or nothing.

Migrations are numbered and applied in order, each in its own transaction, and
recorded in ``schema_migrations``. A migration is never edited after release; a
change is a new migration. Before any migration runs on an existing database, a
backup is taken.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .schema import MIGRATIONS

DEFAULT_NAME = "portfolio.sqlite3"


def now_utc() -> str:
    """A UTC timestamp to the microsecond, the one format every table uses."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def connect(path: str | Path, migrate: bool = True, backup_dir: str | Path | None = None) -> sqlite3.Connection:
    """Open the database, creating it if needed, and bring its schema up to date."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists() and path.stat().st_size > 0
    con = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    if migrate:
        pending = pending_migrations(con)
        if pending and existed:
            backup(con, Path(backup_dir) if backup_dir else path.parent / "backups", label=f"pre-migration-{pending[0]}")
        for version in pending:
            apply_migration(con, version)
    return con


def schema_version(con: sqlite3.Connection) -> int:
    con.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    row = con.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return row[0] or 0


def pending_migrations(con: sqlite3.Connection) -> list[int]:
    current = schema_version(con)
    return [v for v in sorted(MIGRATIONS) if v > current]


def apply_migration(con: sqlite3.Connection, version: int):
    with transaction(con):
        for statement in _statements(MIGRATIONS[version]):
            con.execute(statement)
        con.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)", (version, now_utc()))


def _statements(script: str) -> list[str]:
    """Split a migration into statements; ``executescript`` would commit mid-transaction."""
    return [s.strip() for s in script.split(";\n") if s.strip()]


@contextmanager
def transaction(con: sqlite3.Connection, immediate: bool = True) -> Iterator[sqlite3.Connection]:
    """All or nothing. ``immediate`` takes the write lock at the start, so two writers
    queue instead of one failing halfway; nested use joins the outer transaction."""
    if con.in_transaction:
        yield con
        return
    con.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield con
    except BaseException:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")


def backup(con: sqlite3.Connection, folder: Path, label: str = "manual") -> Path:
    """A consistent copy of the whole database, taken while it stays in use."""
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = folder / f"portfolio_{stamp}_{label}.sqlite3"
    n = 1
    while target.exists():
        n += 1
        target = folder / f"portfolio_{stamp}_{label}_{n}.sqlite3"
    dest = sqlite3.connect(target)
    try:
        con.backup(dest)
    finally:
        dest.close()
    return target
