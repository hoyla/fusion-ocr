"""Page-raster cache — render once, serve both ndarray and PNG consumers; LRU under a
byte budget. Uses a tiny real PDF so the rasterisation path is exercised end to end."""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz", reason="needs the ocr extra (PyMuPDF)")
np = pytest.importorskip("numpy", reason="needs the ocr extra (numpy)")

from fusion_ocr import raster  # noqa: E402


def _pdf(tmp_path, pages=1):
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i} hello")
    path = tmp_path / "doc.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


def test_pixmap_is_cached_per_key(tmp_path):
    raster.clear()
    with fitz.open(_pdf(tmp_path)) as pdf:
        a = raster.page_pixmap(pdf, 0, 150)
        b = raster.page_pixmap(pdf, 0, 150)
        assert a is b                       # same key -> same object (cache hit, no re-render)
        assert raster.page_pixmap(pdf, 0, 200) is not a   # different dpi -> different render


def test_ndarray_matches_direct_render(tmp_path):
    # byte-identical to the inline conversion the stages used to do, so OCR output is unchanged
    raster.clear()
    with fitz.open(_pdf(tmp_path)) as pdf:
        got = raster.page_ndarray(pdf, 0, 150)
        pix = pdf[0].get_pixmap(dpi=150)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = arr[:, :, :3]
        elif pix.n == 1:
            arr = np.repeat(arr, 3, axis=2)
    assert got.shape[2] == 3
    assert np.array_equal(got, arr)


def test_png_is_png_and_clip_is_distinct(tmp_path):
    raster.clear()
    with fitz.open(_pdf(tmp_path)) as pdf:
        full = raster.page_png(pdf, 0, 150)
        assert full[:8] == b"\x89PNG\r\n\x1a\n"
        assert raster.page_png(pdf, 0, 150, clip=(0, 0, 100, 100)) != full   # crop ≠ full page


def test_byte_budget_keeps_cache_bounded(tmp_path, monkeypatch):
    raster.clear()
    monkeypatch.setattr(raster, "_BUDGET_BYTES", 1)   # tiny -> hold at most one entry
    with fitz.open(_pdf(tmp_path, pages=2)) as pdf:
        raster.page_pixmap(pdf, 0, 150)
        raster.page_pixmap(pdf, 1, 150)               # evicts page 0 under the budget
    assert len(raster._cache) <= 1
    raster.clear()
