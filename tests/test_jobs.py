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
