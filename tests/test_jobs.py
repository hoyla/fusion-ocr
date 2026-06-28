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
