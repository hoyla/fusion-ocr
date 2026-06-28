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


def test_upload_accepts_pdf(tmp_path, monkeypatch):
    import fusion_ocr.api as api_mod
    from fusion_ocr.models import Document
    monkeypatch.setattr(api_mod, "process",
                        lambda *a, **k: Document(source_path="x", sha256=k.get("digest", "x")))
    client, _ = _client(tmp_path)
    r = _post_pdf(client, b"%PDF-1.4\nhello\n%%EOF")
    assert r.status_code == 200 and r.json()["status"] == "done"


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


# ---- #2: process() runs off the event loop --------------------------------

def test_submit_offloads_process_off_the_loop(tmp_path, monkeypatch):
    import anyio

    import fusion_ocr.api as api_mod
    from fusion_ocr.models import Document
    used = {}
    real = anyio.to_thread.run_sync

    async def _spy(func, *a, **k):
        used["offloaded"] = True
        return await real(func, *a, **k)

    monkeypatch.setattr(anyio.to_thread, "run_sync", _spy)
    monkeypatch.setattr(api_mod, "process",
                        lambda *a, **k: Document(source_path="x", sha256=k.get("digest", "x")))
    client, _ = _client(tmp_path)
    assert _post_pdf(client, b"%PDF-1.4\nhi\n%%EOF").status_code == 200
    assert used.get("offloaded") is True       # process() went through run_sync, not inline
