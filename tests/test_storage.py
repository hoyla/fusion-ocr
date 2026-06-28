"""Storage seam — content-addressed artifact location (the swap point for a future
object-store adapter)."""

from __future__ import annotations

from fusion_ocr import config as config_mod
from fusion_ocr import storage


def test_job_dir_and_artifacts(tmp_path):
    cfg = config_mod.Config(out_dir=tmp_path / "out")
    d = storage.job_dir(cfg, "abc123")
    assert d == tmp_path / "out" / "abc123"
    assert storage.artifacts(cfg, "abc123") == []          # nothing produced yet
    d.mkdir(parents=True)
    (d / "overlay.pdf").write_bytes(b"x")
    (d / "document.md").write_text("x")
    assert storage.artifacts(cfg, "abc123") == ["document.md", "overlay.pdf"]   # sorted names
