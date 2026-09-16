"""SQLite-backed job table — and the QUEUE BOUNDARY of the system. Producers enqueue
(`upsert_queued`), a worker claims atomically (`claim`: queued -> running), heartbeats while
it works (`touch`) and completes (`set_status`), consumers read (`get` / `list`). Idempotent
by content hash: dropping the same PDF twice is a no-op (the existing job/artifacts are
reused).

The claim is a LEASE, not a permanent hand-off: a worker killed between `claim` and
`set_status` (OOM, power, `kill -9`) used to leave its job `running` forever — unclaimable,
invisible to `--force`, never retried. Now a live worker refreshes `updated_at` on a timer,
and `requeue_stale` hands back any running job whose heartbeat is older than the lease
(review 03 (b)). The lease targets a DEAD worker, not a hung one: a worker stuck inside a
call still heartbeats, and that failure belongs to the call's own timeout.

The store also carries the RUNTIME CONFIG OVERRIDES (`settings` table): `PATCH /config`
used to mutate only the API process's Config, so in the two-process deployment the worker —
the process that actually runs the pipeline — never saw the change (review 03). Now the API
records each validated override here, and both processes apply the current set on top of
their file config (the worker at every scan, the API at startup), so a runtime tuning reaches
the jobs and re-keys their recipe fingerprint. `POST /config/save` promotes the overrides to
config.toml and clears them (they are then the file).

Tiny on purpose — a dozen docs a day needs nothing more. But this method surface IS the
contract a future distributed queue would implement: an SQS / ElasticMQ adapter (on-estate,
airgap-compatible) is a drop-in here, not a rewrite (visibility timeout = the lease). Keep all
queue access going through these methods so that swap stays cheap."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    sha256        TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL,
    status        TEXT NOT NULL,         -- queued | running | done | error
    error         TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,         -- last state change OR heartbeat (see touch)
    original_name TEXT                   -- the client's / dropped filename (provenance)
);
CREATE TABLE IF NOT EXISTS settings (
    path        TEXT PRIMARY KEY,        -- dotted Config path, e.g. fuse_min_sim / vlm.model
    value       TEXT NOT NULL,           -- JSON-encoded, already validated (settings.validate)
    updated_at  REAL NOT NULL
);
"""

# Columns added after the first release, applied to an existing store on open (SQLite has
# no ADD COLUMN IF NOT EXISTS). name -> column DDL.
_MIGRATIONS: dict[str, str] = {
    "original_name": "TEXT",
}


class JobStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")   # concurrent reads alongside a writer
            c.executescript(_SCHEMA)
            have = {r["name"] for r in c.execute("PRAGMA table_info(jobs)")}
            for col, ddl in _MIGRATIONS.items():
                if col not in have:
                    c.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ddl}")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, sha256: str) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute("SELECT * FROM jobs WHERE sha256=?", (sha256,)).fetchone()

    def upsert_queued(self, sha256: str, source_path: str,
                      original_name: str | None = None) -> bool:
        """Return True if newly queued, False if it already existed. Atomic: a single
        INSERT .. ON CONFLICT DO NOTHING leans on the sha256 PK, so two concurrent submits
        of the same content can't both 'win' (the old SELECT-then-INSERT could race into a
        duplicate-processing or IntegrityError). rowcount is 1 on insert, 0 on conflict.
        `original_name` is the human name the content arrived under (the API's client
        filename, or the dropped file's name) — the first registration's name is kept."""
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO jobs(sha256, source_path, status, created_at, updated_at,"
                " original_name) VALUES(?,?,?,?,?,?) ON CONFLICT(sha256) DO NOTHING",
                (sha256, source_path, "queued", now, now, original_name),
            )
            return cur.rowcount == 1

    def claim(self, sha256: str, reprocess: bool = False) -> bool:
        """Atomically take a job for processing: queued -> running, in one statement, so
        concurrent workers can't both claim it (rowcount is 1 for the winner, 0 otherwise).
        Returns True if THIS caller claimed it. With reprocess=True, also re-claims a done /
        error job (for --force / --rerun-from), but never steals one already running — a
        running job whose worker has died comes back via `requeue_stale`, not by force."""
        cond = "status != 'running'" if reprocess else "status = 'queued'"
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE jobs SET status='running', updated_at=? WHERE sha256=? AND {cond}",
                (time.time(), sha256),
            )
            return cur.rowcount == 1

    def touch(self, sha256: str) -> None:
        """Heartbeat: refresh a RUNNING job's updated_at so `requeue_stale` can tell a live
        worker from a dead one. A no-op for any other status (never resurrects a job)."""
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET updated_at=? WHERE sha256=? AND status='running'",
                (time.time(), sha256),
            )

    def requeue_stale(self, lease_seconds: float) -> list[str]:
        """Return to `queued` every RUNNING job whose last heartbeat is older than
        `lease_seconds` — the worker that claimed it died mid-job — and return their
        sha256s. A live worker heartbeats well inside the lease, so a job genuinely still
        being processed is never taken back. The reason is left in `error` until the next
        completion overwrites it, so the requeue is visible in `GET /jobs`."""
        now = time.time()
        cutoff = now - lease_seconds
        note = (f"requeued: no worker heartbeat for {lease_seconds:g}s "
                f"(worker died mid-job?)")
        requeued: list[str] = []
        with self._conn() as c:
            stale = [r["sha256"] for r in c.execute(
                "SELECT sha256 FROM jobs WHERE status='running' AND updated_at < ?",
                (cutoff,))]
            for sha in stale:
                # Re-check the predicate in the UPDATE: a heartbeat racing in between the
                # SELECT and here keeps the job (rowcount 0), so it isn't reported either.
                cur = c.execute(
                    "UPDATE jobs SET status='queued', error=?, updated_at=?"
                    " WHERE sha256=? AND status='running' AND updated_at < ?",
                    (note, now, sha, cutoff),
                )
                if cur.rowcount == 1:
                    requeued.append(sha)
        return requeued

    def set_status(self, sha256: str, status: str, error: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE sha256=?",
                (status, error, time.time(), sha256),
            )

    # -- runtime config overrides (shared by the API and every worker) -----------------

    def set_overrides(self, updates: dict) -> None:
        """Record validated {path: value} runtime overrides (upsert; later wins)."""
        now = time.time()
        with self._conn() as c:
            for path, value in updates.items():
                c.execute(
                    "INSERT INTO settings(path, value, updated_at) VALUES(?,?,?)"
                    " ON CONFLICT(path) DO UPDATE SET value=excluded.value,"
                    " updated_at=excluded.updated_at",
                    (path, json.dumps(value), now),
                )

    def overrides(self) -> tuple[dict, float]:
        """The current override set and its version — the latest updated_at (0.0 when
        empty) — so a worker can apply the set only when it has changed."""
        with self._conn() as c:
            rows = c.execute("SELECT path, value, updated_at FROM settings").fetchall()
        return ({r["path"]: json.loads(r["value"]) for r in rows},
                max((r["updated_at"] for r in rows), default=0.0))

    def clear_overrides(self) -> None:
        """Drop every override — after POST /config/save wrote them into config.toml."""
        with self._conn() as c:
            c.execute("DELETE FROM settings")

    def list(self, status: str | None = None) -> list[sqlite3.Row]:
        """All jobs, newest first — optionally filtered by status. Backs the 'out' feed
        (e.g. GET /jobs?status=done) so a consumer can pull completed work."""
        with self._conn() as c:
            if status:
                return c.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC", (status,)
                ).fetchall()
            return c.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
