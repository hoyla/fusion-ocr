"""Real-geometry test for the PaddleOCR stage.

Builds an IMAGE-ONLY PDF (text rasterised into an image, no text layer) so triage
flags it for OCR, then asserts PaddleOCR recovers the words with boxes that land
inside the page in PDF coordinates.

Skipped unless the `ocr` extra is installed (paddleocr + paddlepaddle); the rest of
the suite stays green without the heavy stack.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz", reason="needs PyMuPDF")
pytest.importorskip("paddleocr", reason="needs the `ocr` extra")

import fitz  # noqa: E402

from fusion_ocr import config as config_mod  # noqa: E402
from fusion_ocr.pipeline import DEFAULT_PIPELINE, process  # noqa: E402


def _image_only_pdf(path, text="HELLO WORLD invoice 2026"):
    """Render text onto a page, flatten to a pixmap, and rebuild the page from the
    image — leaving no extractable text layer."""
    src = fitz.open()
    pg = src.new_page(width=612, height=792)
    pg.insert_text((72, 200), text, fontsize=32)
    pix = pg.get_pixmap(dpi=200)
    src.close()

    out = fitz.open()
    page = out.new_page(width=612, height=792)
    page.insert_image(page.rect, pixmap=pix)
    out.save(str(path))
    out.close()


def test_paddleocr_produces_boxed_text(tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    _image_only_pdf(pdf_path)

    # Confirm it's genuinely image-only (no text layer to cheat from).
    with fitz.open(pdf_path) as d:
        assert d[0].get_text("text").strip() == ""

    cfg = config_mod.Config(out_dir=tmp_path / "out", airgap=False)
    doc = process(pdf_path, cfg, pipeline=DEFAULT_PIPELINE)

    segs = [s for p in doc.pages for s in p.segments]
    assert segs, "expected PaddleOCR to detect at least one line"

    combined = " ".join(s.det_text or "" for s in segs).lower()
    assert "hello" in combined or "world" in combined

    # Boxes must be real geometry in PDF points, inside the 612x792 page.
    page = doc.pages[0]
    for s in segs:
        assert s.det_conf is not None and 0.0 <= s.det_conf <= 1.0
        x0, y0, x1, y1 = s.box.bbox
        assert 0 <= x0 < x1 <= page.width + 1
        assert 0 <= y0 < y1 <= page.height + 1

    # End-to-end: the overlay PDF should now actually be produced.
    assert "overlay_pdf" in doc.artifacts
