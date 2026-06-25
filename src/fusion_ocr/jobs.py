"""SQLite-backed job table. Idempotent by content hash: dropping the same PDF twice
is a no-op (the existing job/artifacts are reused). Tiny on purpose — a dozen docs a
day needs nothing more; the same contract scales to a real queue later."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    sha256       TEXT PRIMARY KEY,
    source_path  TEXT NOT NULL,
    status       TEXT NOT NULL,         -- queued | running | done | error
    error        TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
"""


class JobStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, sha256: str) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute("SELECT * FROM jobs WHERE sha256=?", (sha256,)).fetchone()

    def upsert_queued(self, sha256: str, source_path: str) -> bool:
        """Return True if newly queued, False if it already existed."""
        now = time.time()
        with self._conn() as c:
            existing = c.execute(
                "SELECT 1 FROM jobs WHERE sha256=?", (sha256,)
            ).fetchone()
            if existing:
                return False
            c.execute(
                "INSERT INTO jobs(sha256, source_path, status, created_at, updated_at)"
                " VALUES(?,?,?,?,?)",
                (sha256, source_path, "queued", now, now),
            )
            return True

    def set_status(self, sha256: str, status: str, error: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE sha256=?",
                (status, error, time.time(), sha256),
            )

    def all(self) -> list[sqlite3.Row]:
        with self._conn() as c:
            return c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
