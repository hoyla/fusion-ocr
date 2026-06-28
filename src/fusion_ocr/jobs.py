"""SQLite-backed job table — and the QUEUE BOUNDARY of the system. Producers enqueue
(`upsert_queued`), a worker claims atomically (`claim`: queued -> running) and completes
(`set_status`), consumers read (`get` / `list`). Idempotent by content hash: dropping the
same PDF twice is a no-op (the existing job/artifacts are reused).

Tiny on purpose — a dozen docs a day needs nothing more. But this method surface IS the
contract a future distributed queue would implement: an SQS / ElasticMQ adapter (on-estate,
airgap-compatible) is a drop-in here, not a rewrite. Keep all queue access going through
these methods so that swap stays cheap."""

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
            c.execute("PRAGMA journal_mode=WAL")   # concurrent reads alongside a writer
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, sha256: str) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute("SELECT * FROM jobs WHERE sha256=?", (sha256,)).fetchone()

    def upsert_queued(self, sha256: str, source_path: str) -> bool:
        """Return True if newly queued, False if it already existed. Atomic: a single
        INSERT .. ON CONFLICT DO NOTHING leans on the sha256 PK, so two concurrent submits
        of the same content can't both 'win' (the old SELECT-then-INSERT could race into a
        duplicate-processing or IntegrityError). rowcount is 1 on insert, 0 on conflict."""
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO jobs(sha256, source_path, status, created_at, updated_at)"
                " VALUES(?,?,?,?,?) ON CONFLICT(sha256) DO NOTHING",
                (sha256, source_path, "queued", now, now),
            )
            return cur.rowcount == 1

    def claim(self, sha256: str, reprocess: bool = False) -> bool:
        """Atomically take a job for processing: queued -> running, in one statement, so
        concurrent workers can't both claim it (rowcount is 1 for the winner, 0 otherwise).
        Returns True if THIS caller claimed it. With reprocess=True, also re-claims a done /
        error job (for --force / --rerun-from), but never steals one already running."""
        cond = "status != 'running'" if reprocess else "status = 'queued'"
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE jobs SET status='running', updated_at=? WHERE sha256=? AND {cond}",
                (time.time(), sha256),
            )
            return cur.rowcount == 1

    def set_status(self, sha256: str, status: str, error: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE sha256=?",
                (status, error, time.time(), sha256),
            )

    def list(self, status: str | None = None) -> list[sqlite3.Row]:
        """All jobs, newest first — optionally filtered by status. Backs the 'out' feed
        (e.g. GET /jobs?status=done) so a consumer can pull completed work."""
        with self._conn() as c:
            if status:
                return c.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC", (status,)
                ).fetchall()
            return c.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
