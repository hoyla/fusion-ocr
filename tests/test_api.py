"""API upload-path safety (extra: api). Tests the pure sanitizers — importing the
module is side-effect-free now (app is lazy), so no server/config/sqlite is built."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="needs the api extra")

from fusion_ocr.api import _is_sha256, _safe_name  # noqa: E402


def test_safe_name_strips_directory_traversal():
    assert _safe_name("../../etc/passwd") == "passwd"      # the path-traversal payload
    assert _safe_name("/abs/path/x.pdf") == "x.pdf"
    assert _safe_name("sub/dir/report.pdf") == "report.pdf"
    assert _safe_name("normal.pdf") == "normal.pdf"


def test_safe_name_handles_empty_and_dot_segments():
    assert _safe_name(None) == "upload.pdf"
    assert _safe_name("") == "upload.pdf"
    assert _safe_name("..") == "upload.pdf"
    assert _safe_name(".") == "upload.pdf"


def test_is_sha256_rejects_path_components():
    assert _is_sha256("a" * 64)
    assert _is_sha256("0123456789abcdef" * 4)
    assert not _is_sha256("../../etc/passwd")   # the GET /jobs/{sha256} traversal payload
    assert not _is_sha256("a" * 63)             # wrong length
    assert not _is_sha256("g" * 64)             # non-hex
    assert not _is_sha256("")


# ---- upload guards: size cap + PDF sniff (streamed, before hashing) --------

_TOKEN = "test-token"


def _client(tmp_path, **cfg_kw):
    from fastapi.testclient import TestClient

    from fusion_ocr import config as config_mod
    from fusion_ocr.api import create_app
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out",
                            airgap=False, **cfg_kw)
    client = TestClient(create_app(cfg, token=_TOKEN),
                        headers={"Authorization": f"Bearer {_TOKEN}"})
    return client, cfg


def _post_pdf(client, body: bytes):
    return client.post("/jobs", files={"pdf": ("doc.pdf", body, "application/pdf")})


def test_upload_rejects_non_pdf(tmp_path):
    client, _ = _client(tmp_path)
    r = _post_pdf(client, b"this is not a pdf")
    assert r.status_code == 415
    assert not list((tmp_path / "in").glob("*.pdf"))      # nothing left behind


def test_upload_rejects_oversized(tmp_path):
    client, _ = _client(tmp_path, max_upload_mb=0.001)    # ~1 KB cap
    r = _post_pdf(client, b"%PDF-1.4\n" + b"0" * 4000)
    assert r.status_code == 413
    assert not list((tmp_path / "in").glob("*.pdf"))      # partial file cleaned up


def test_upload_enqueues_and_returns_202(tmp_path):
    client, _ = _client(tmp_path)
    r = _post_pdf(client, b"%PDF-1.4\nhello\n%%EOF")
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and len(body["sha256"]) == 64
    # the upload is parked in in/ for the worker; submit did NOT process it inline
    assert list((tmp_path / "in").glob("*.pdf"))


def test_jobs_feed_reflects_queue(tmp_path):
    client, _ = _client(tmp_path)
    sha = _post_pdf(client, b"%PDF-1.4\nhello\n%%EOF").json()["sha256"]
    listed = client.get("/jobs").json()["jobs"]
    assert any(j["sha256"] == sha and j["status"] == "queued" for j in listed)
    assert client.get("/jobs", params={"status": "done"}).json()["jobs"] == []   # none done yet
    one = client.get(f"/jobs/{sha}").json()
    assert one["status"] == "queued" and one["artifacts"] == []


def test_enqueue_then_worker_drains_end_to_end(tmp_path, monkeypatch):
    # The full async contract: API enqueues -> worker (sharing the same in/out) drains ->
    # GET reflects done + artifacts. Proves the API and watcher meet on one queue.
    from pathlib import Path

    from fusion_ocr import config as config_mod
    from fusion_ocr import storage
    from fusion_ocr import watcher as watcher_mod
    from fusion_ocr.jobs import JobStore
    from fusion_ocr.models import Document

    client, cfg = _client(tmp_path)
    sha = _post_pdf(client, b"%PDF-1.4\nhello\n%%EOF").json()["sha256"]

    def _fake_process(pdf, c, **k):           # worker stub: drop an artifact under out/<sha>/
        d = storage.job_dir(c, k["digest"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "document.md").write_text("ok")
        return Document(source_path=str(pdf), sha256=k["digest"])

    monkeypatch.setattr(watcher_mod, "process", _fake_process)
    jobs = JobStore(Path(cfg.out_dir) / "jobs.sqlite")            # same DB the API wrote to
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0) == 1
    done = client.get(f"/jobs/{sha}").json()
    assert done["status"] == "done" and "document.md" in done["artifacts"]


# ---- bearer-token auth (fail closed) --------------------------------------

def test_create_app_fails_closed_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("FUSION_OCR_API_TOKEN", raising=False)
    from fusion_ocr import config as config_mod
    from fusion_ocr.api import create_app
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out", airgap=False)
    with pytest.raises(RuntimeError, match="FUSION_OCR_API_TOKEN"):
        create_app(cfg)                       # no token, no env -> refuses to start


def test_requests_require_the_bearer_token(tmp_path):
    from fastapi.testclient import TestClient

    from fusion_ocr import config as config_mod
    from fusion_ocr.api import create_app
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out", airgap=False)
    client = TestClient(create_app(cfg, token="secret"))   # no default auth header
    assert client.get("/config").status_code == 401                       # missing
    assert client.get("/config", headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = client.get("/config", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


# ---- POST /config/save: explicit, opt-in persistence ----------------------

def test_config_save_persists_runtime_changes(tmp_path):
    pytest.importorskip("tomli_w", reason="needs the api extra")
    from fastapi.testclient import TestClient

    from fusion_ocr import config as config_mod
    from fusion_ocr.api import create_app
    cfg_path = tmp_path / "config.toml"
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out", airgap=False)
    client = TestClient(create_app(cfg, token="t", config_path=cfg_path),
                        headers={"Authorization": "Bearer t"})

    client.patch("/config", json={"fuse_min_sim": 0.5})  # in-process only so far
    assert not cfg_path.exists()                          # PATCH alone never persisted
    r = client.post("/config/save")
    assert r.status_code == 200 and r.json()["saved"] == str(cfg_path)
    assert config_mod.load(cfg_path).fuse_min_sim == 0.5  # now on disk
