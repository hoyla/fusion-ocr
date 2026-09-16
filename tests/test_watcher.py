"""Watcher settle gate — a file still being copied (recent mtime) must not be hashed and
processed until it settles, and the digest the watcher computed is passed through to
process() rather than re-hashed."""

from __future__ import annotations

from fusion_ocr import config as config_mod
from fusion_ocr import watcher as watcher_mod
from fusion_ocr.jobs import JobStore
from fusion_ocr.models import Document
from fusion_ocr.pipeline import sha256_of


def _stub_process(ran):
    def _p(*args, **kwargs):
        ran.append(kwargs.get("digest"))
        return Document(source_path="x", sha256="x")
    return _p


def _drop(in_dir):
    in_dir.mkdir(parents=True, exist_ok=True)
    pdf = in_dir / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf


def test_unsettled_file_is_skipped(tmp_path, monkeypatch):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    _drop(tmp_path / "in")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    ran: list = []
    monkeypatch.setattr(watcher_mod, "process", _stub_process(ran))
    # the just-written file looks unsettled under a huge window -> skipped this scan
    assert watcher_mod.scan_once(cfg, jobs, min_settle=10_000) == 0
    assert ran == []


def test_settled_file_processed_with_passed_digest(tmp_path, monkeypatch):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    pdf = _drop(tmp_path / "in")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    ran: list = []
    monkeypatch.setattr(watcher_mod, "process", _stub_process(ran))
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0) == 1   # no settle window
    assert ran == [sha256_of(pdf)]                                 # passed through, not re-hashed


def test_worker_drains_api_enqueued_job(tmp_path, monkeypatch):
    # Simulate POST /jobs: the file is in in/ and already registered 'queued' (not new to
    # the folder). The status-driven worker must CLAIM and process it, not skip it as "seen".
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    pdf = _drop(tmp_path / "in")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    jobs.upsert_queued(sha256_of(pdf), str(pdf))               # pre-registered, like the API
    ran: list = []
    monkeypatch.setattr(watcher_mod, "process", _stub_process(ran))
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0) == 1
    assert ran == [sha256_of(pdf)]
    assert jobs.get(sha256_of(pdf))["status"] == "done"


def test_processed_file_is_moved_and_not_rescanned(tmp_path, monkeypatch):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    pdf = _drop(tmp_path / "in")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    monkeypatch.setattr(watcher_mod, "process", _stub_process([]))
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0, move_processed=True) == 1
    assert not pdf.exists()                                        # moved out of in/
    assert len(list((tmp_path / "in" / "processed").glob("*.pdf"))) == 1
    # the moved file isn't re-globbed (in/processed is a subdir) -> nothing to do next scan
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0, move_processed=True) == 0


def test_failed_file_moved_to_failed(tmp_path, monkeypatch):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    pdf = _drop(tmp_path / "in")
    jobs = JobStore(tmp_path / "jobs.sqlite")

    def _boom(*a, **k):
        raise RuntimeError("nope")
    monkeypatch.setattr(watcher_mod, "process", _boom)
    watcher_mod.scan_once(cfg, jobs, min_settle=0.0, move_processed=True)
    assert not pdf.exists()
    assert len(list((tmp_path / "in" / "failed").glob("*.pdf"))) == 1


def test_no_move_by_default(tmp_path, monkeypatch):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    pdf = _drop(tmp_path / "in")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    monkeypatch.setattr(watcher_mod, "process", _stub_process([]))
    watcher_mod.scan_once(cfg, jobs, min_settle=0.0)              # default: --once semantics
    assert pdf.exists()                                           # left in place


# ---- review 03 (b)/(c): a dead worker's job is requeued; a vanished file can't kill the loop

def _backdate(jobs, sha, seconds):
    import time
    with jobs._conn() as c:
        c.execute("UPDATE jobs SET updated_at=? WHERE sha256=?", (time.time() - seconds, sha))


def test_vanished_file_does_not_kill_the_scan(tmp_path, monkeypatch):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    in_dir = tmp_path / "in"
    in_dir.mkdir(parents=True)
    (in_dir / "a.pdf").write_bytes(b"%PDF-1.4 first")
    b = in_dir / "b.pdf"
    b.write_bytes(b"%PDF-1.4 second")
    real = watcher_mod.sha256_of

    def _vanish(path):                 # a.pdf disappears between the listing and the hash
        if path.name == "a.pdf":
            path.unlink()
            raise FileNotFoundError(path)
        return real(path)
    monkeypatch.setattr(watcher_mod, "sha256_of", _vanish)
    ran: list = []
    monkeypatch.setattr(watcher_mod, "process", _stub_process(ran))
    assert watcher_mod.scan_once(cfg, jobs := JobStore(tmp_path / "jobs.sqlite"),
                                 min_settle=0.0) == 1          # the loop went on to b.pdf
    assert ran == [real(b)]
    assert jobs.get(real(b))["status"] == "done"


def test_stale_running_job_is_requeued_and_processed(tmp_path, monkeypatch):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    pdf = _drop(tmp_path / "in")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    sha = sha256_of(pdf)
    jobs.upsert_queued(sha, str(pdf))
    assert jobs.claim(sha)             # a previous worker took it ...
    _backdate(jobs, sha, 3600)         # ... and died an hour ago (no heartbeats since)
    ran: list = []
    monkeypatch.setattr(watcher_mod, "process", _stub_process(ran))
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0, lease_seconds=600) == 1
    assert ran == [sha] and jobs.get(sha)["status"] == "done"


def test_live_running_job_is_not_stolen(tmp_path, monkeypatch):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    pdf = _drop(tmp_path / "in")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    sha = sha256_of(pdf)
    jobs.upsert_queued(sha, str(pdf))
    assert jobs.claim(sha)             # another worker holds it, heartbeat fresh
    ran: list = []
    monkeypatch.setattr(watcher_mod, "process", _stub_process(ran))
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0, lease_seconds=600) == 0
    assert ran == [] and jobs.get(sha)["status"] == "running"


def test_heartbeat_keeps_the_lease_fresh_while_processing(tmp_path, monkeypatch):
    import time
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    _drop(tmp_path / "in")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    beats: list = []

    def _slow(*args, **kwargs):        # a long job: the lease must advance while it runs
        sha = kwargs["digest"]
        t0 = jobs.get(sha)["updated_at"]
        time.sleep(0.3)
        beats.append(jobs.get(sha)["updated_at"] > t0)
        return Document(source_path="x", sha256=sha)
    monkeypatch.setattr(watcher_mod, "process", _slow)
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0, heartbeat_seconds=0.05) == 1
    assert beats == [True]


def test_redrop_of_done_content_is_moved_out_not_rehashed_forever(tmp_path, monkeypatch):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    pdf = _drop(tmp_path / "in")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    ran: list = []
    monkeypatch.setattr(watcher_mod, "process", _stub_process(ran))
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0, move_processed=True) == 1
    again = tmp_path / "in" / "same-bytes-other-name.pdf"
    again.write_bytes(b"%PDF-1.4 fake")                      # identical bytes to the first drop
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0, move_processed=True) == 0
    assert len(ran) == 1                                     # processed exactly once
    assert not again.exists()                                # handled: moved with the rest
    assert len(list((tmp_path / "in" / "processed").glob("*.pdf"))) == 1


# ---- review 03: PATCH /config must reach the worker (runtime overrides via the job store)

def test_worker_applies_runtime_overrides_from_the_store(tmp_path):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    jobs.set_overrides({"fuse_min_sim": 0.5, "vlm.model": "some/other-reader"})   # what the API recorded
    sync = watcher_mod.OverrideSync()
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0, overrides=sync) == 0
    assert cfg.fuse_min_sim == 0.5 and cfg.vlm.model == "some/other-reader"      # the worker saw it
    v = sync.version
    assert sync.pull(cfg, jobs) is False and sync.version == v                     # unchanged set: no-op
    import time
    time.sleep(0.01)
    jobs.set_overrides({"fuse_min_sim": 0.7})
    assert sync.pull(cfg, jobs) is True and cfg.fuse_min_sim == 0.7                # a change is picked up


def test_worker_survives_an_unknown_override(tmp_path, capsys):
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out")
    jobs = JobStore(tmp_path / "jobs.sqlite")
    jobs.set_overrides({"not_a_setting": 1})            # e.g. written by a newer API build
    assert watcher_mod.OverrideSync().pull(cfg, jobs) is False
    assert "not applied" in capsys.readouterr().err     # logged, not fatal
