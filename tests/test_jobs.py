"""Job store — idempotent-by-content-hash queueing, done atomically so concurrent
submits of the same content can't both 'win' or raise."""

from __future__ import annotations

from fusion_ocr.jobs import JobStore


def test_upsert_queued_is_idempotent_and_atomic(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite")
    sha = "a" * 64
    assert jobs.upsert_queued(sha, "/in/x.pdf") is True     # first insert wins
    assert jobs.upsert_queued(sha, "/in/x.pdf") is False    # conflict -> no-op, no raise
    row = jobs.get(sha)
    assert row["status"] == "queued" and row["source_path"] == "/in/x.pdf"


def test_claim_is_atomic_one_shot_and_reprocessable(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite")
    sha = "b" * 64
    jobs.upsert_queued(sha, "/in/x.pdf")
    assert jobs.claim(sha) is True                  # queued -> running, this caller wins
    assert jobs.get(sha)["status"] == "running"
    assert jobs.claim(sha) is False                 # already running -> can't double-claim
    jobs.set_status(sha, "done")
    assert jobs.claim(sha) is False                 # done isn't claimable normally
    assert jobs.claim(sha, reprocess=True) is True  # ...but a forced reprocess re-claims it


def test_list_filters_by_status(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite")
    jobs.upsert_queued("c" * 64, "/in/a.pdf")
    jobs.upsert_queued("d" * 64, "/in/b.pdf")
    jobs.set_status("d" * 64, "done")
    assert {r["sha256"] for r in jobs.list()} == {"c" * 64, "d" * 64}
    assert [r["sha256"] for r in jobs.list(status="done")] == ["d" * 64]


# ---- lease: a worker that dies mid-job must not strand its job as running --------

def _backdate(jobs, sha, seconds):
    import time
    with jobs._conn() as c:
        c.execute("UPDATE jobs SET updated_at=? WHERE sha256=?", (time.time() - seconds, sha))


def test_requeue_stale_hands_back_only_expired_running_jobs(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite")
    dead, live, done = "e" * 64, "f" * 64, "0" * 64
    for sha in (dead, live, done):
        jobs.upsert_queued(sha, f"/in/{sha[:2]}.pdf")
        assert jobs.claim(sha)
    jobs.set_status(done, "done")
    _backdate(jobs, dead, 3600)                     # its worker stopped heartbeating an hour ago
    _backdate(jobs, done, 3600)                     # old but finished — not a lease holder
    assert jobs.requeue_stale(lease_seconds=600) == [dead]
    assert jobs.get(dead)["status"] == "queued" and "heartbeat" in jobs.get(dead)["error"]
    assert jobs.get(live)["status"] == "running"     # fresh heartbeat -> untouched
    assert jobs.get(done)["status"] == "done"
    assert jobs.claim(dead) is True                  # back in circulation for any worker
    assert jobs.requeue_stale(lease_seconds=600) == []   # nothing left to reap


def test_touch_refreshes_a_running_lease_only(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite")
    sha = "1" * 64
    jobs.upsert_queued(sha, "/in/x.pdf")
    _backdate(jobs, sha, 3600)
    jobs.touch(sha)                                   # queued: a heartbeat can't resurrect it
    assert jobs.requeue_stale(600) == [] and jobs.get(sha)["status"] == "queued"
    assert jobs.claim(sha)
    _backdate(jobs, sha, 3600)
    jobs.touch(sha)                                   # running: the beat extends the lease
    assert jobs.requeue_stale(600) == []
    assert jobs.get(sha)["status"] == "running"


# ---- original_name: provenance for content-keyed jobs, migrated onto old stores ----

def test_original_name_is_recorded_and_first_registration_wins(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite")
    sha = "2" * 64
    jobs.upsert_queued(sha, "/in/abc__report.pdf", original_name="report.pdf")
    jobs.upsert_queued(sha, "/in/abc__copy.pdf", original_name="copy.pdf")   # same content
    assert jobs.get(sha)["original_name"] == "report.pdf"
    assert jobs.list()[0]["original_name"] == "report.pdf"


def test_pre_existing_store_is_migrated_on_open(tmp_path):
    import sqlite3
    db = tmp_path / "jobs.sqlite"
    with sqlite3.connect(db) as c:                    # the first-release schema, no original_name
        c.executescript("""CREATE TABLE jobs (sha256 TEXT PRIMARY KEY, source_path TEXT NOT NULL,
            status TEXT NOT NULL, error TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
            INSERT INTO jobs VALUES ('3333', '/in/old.pdf', 'done', NULL, 1.0, 1.0);""")
    jobs = JobStore(db)                               # opens without error, adds the column
    assert jobs.get("3333")["original_name"] is None  # old rows: unknown, not invented
    assert jobs.upsert_queued("4" * 64, "/in/new.pdf", original_name="new.pdf")
    assert jobs.get("4" * 64)["original_name"] == "new.pdf"


# ---- runtime config overrides: the shared state PATCH /config writes and workers read ----

def test_overrides_round_trip_version_and_clear(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite")
    assert jobs.overrides() == ({}, 0.0)                    # empty table = version 0
    jobs.set_overrides({"fuse_min_sim": 0.5, "prefer_apple_vision": True})
    got, v1 = jobs.overrides()
    assert got == {"fuse_min_sim": 0.5, "prefer_apple_vision": True} and v1 > 0
    import time
    time.sleep(0.01)
    jobs.set_overrides({"fuse_min_sim": 0.6})               # later wins, version advances
    got, v2 = jobs.overrides()
    assert got["fuse_min_sim"] == 0.6 and got["prefer_apple_vision"] is True and v2 > v1
    jobs.clear_overrides()
    assert jobs.overrides() == ({}, 0.0)
