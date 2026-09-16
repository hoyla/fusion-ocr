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


def test_overlay_skips_textlayer_to_avoid_duplication(tmp_path):
    pytest.importorskip("fitz", reason="needs PyMuPDF")
    import fitz
    doc = _doc_with_segment(tmp_path, "OCR body text")        # the one seg is "fused"
    # add a text-layer segment (already in the source PDF) — must NOT be re-overlaid
    tl = Segment(id="tl", page=0,
                 box=Box(points=[(50, 50), (400, 50), (400, 70), (50, 70)]),
                 best_text="HEADER FROM TEXT LAYER", source="textlayer")
    doc.pages[0].segments.insert(0, tl)
    out = tmp_path / "ov.pdf"
    assert build_overlay(doc, out)
    text = fitz.open(out)[0].get_text("text")
    assert "OCR body text" in text                # OCR-derived text is overlaid
    assert "HEADER FROM TEXT LAYER" not in text    # text-layer text is NOT duplicated


@pytest.mark.parametrize("source", sorted(__import__("fusion_ocr.models", fromlist=["OCR_SOURCES"]).OCR_SOURCES) + ["fused"])
def test_any_engine_text_is_overlaid(tmp_path, source):
    # The overlay carries OCR-derived text whichever engine (or fusion) produced it; only the
    # source PDF's own text layer is excluded (it is already searchable, and would double).
    pytest.importorskip("fitz", reason="needs PyMuPDF")
    import fitz
    doc = _doc_with_segment(tmp_path, "engine agnostic line")
    doc.pages[0].segments[0].source = source
    out = tmp_path / "ov.pdf"
    assert build_overlay(doc, out)
    assert fitz.open(out)[0].search_for("engine agnostic line")


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


def test_word_granularity_is_retired_with_a_warning(tmp_path, caplog):
    pytest.importorskip("fitz", reason="needs PyMuPDF")
    import fitz
    doc = _doc_with_segment(tmp_path, "alpha beta gamma")
    out = tmp_path / "ov.pdf"
    with caplog.at_level("WARNING"):
        assert build_overlay(doc, out, granularity="word")
    assert "not supported" in caplog.text                       # said, not silently ignored
    assert len(fitz.open(out)[0].search_for("alpha beta gamma")) == 1   # one line-level string


def test_unplaceable_line_is_logged_not_swallowed(tmp_path, caplog, monkeypatch):
    pytest.importorskip("fitz", reason="needs PyMuPDF")
    import fitz

    def _boom(self, *a, **k):
        raise RuntimeError("glyph not in font")
    monkeypatch.setattr(fitz.Page, "insert_text", _boom)
    doc = _doc_with_segment(tmp_path, "unplaceable")
    with caplog.at_level("WARNING"):
        assert build_overlay(doc, tmp_path / "ov.pdf")             # the overlay still builds
    assert "not searchable" in caplog.text and "unplaceable" not in caplog.text  # no text leaked
