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
