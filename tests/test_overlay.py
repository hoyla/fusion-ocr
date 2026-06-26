"""Overlay — font resolution + searchable invisible text (incl. Thai)."""

from __future__ import annotations

import pytest

from fusion_ocr.models import Box, Document, Page, Segment
from fusion_ocr.overlay.pymupdf_overlay import _resolve_font, build_overlay


def test_resolve_font_prefers_explicit_then_falls_back(tmp_path):
    # explicit valid path wins
    f = tmp_path / "f.ttf"; f.write_bytes(b"x")
    assert _resolve_font(str(f)) == str(f)
    # bogus path -> auto-detect (a system Unicode font) or None (helv fallback)
    assert _resolve_font("/nope/missing.ttf") in (None,) or _resolve_font("/nope/missing.ttf")


def _doc_with_segment(tmp_path, text):
    fitz = pytest.importorskip("fitz", reason="needs PyMuPDF")
    src = tmp_path / "src.pdf"
    d = fitz.open(); d.new_page(width=612, height=792); d.save(str(src)); d.close()
    doc = Document(source_path=str(src), sha256="x")
    page = Page(index=0, width=612, height=792)
    page.segments = [Segment(id="a", page=0,
                             box=Box(points=[(50, 90), (400, 90), (400, 110), (50, 110)]),
                             best_text=text, source="fused")]
    doc.pages = [page]
    return doc


def test_overlay_latin_searchable(tmp_path):
    pytest.importorskip("fitz", reason="needs PyMuPDF")
    import fitz
    doc = _doc_with_segment(tmp_path, "Invoice total due")
    out = tmp_path / "ov.pdf"
    assert build_overlay(doc, out)
    assert fitz.open(out)[0].search_for("Invoice")


def test_overlay_thai_searchable(tmp_path):
    pytest.importorskip("fitz", reason="needs PyMuPDF")
    import fitz
    if _resolve_font(None) is None:
        pytest.skip("no Unicode font available -> Thai overlay search not supported")
    term = "กระทรวงพาณิชย์"
    doc = _doc_with_segment(tmp_path, f"กรมพัฒนาธุรกิจการค้า {term}")
    out = tmp_path / "ov.pdf"
    assert build_overlay(doc, out)
    # the fix: with a Unicode font, search_for finds the Thai term (helv = 0 hits)
    assert fitz.open(out)[0].search_for(term)
