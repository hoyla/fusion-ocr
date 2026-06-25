"""Triage v2 routing tests — synthetic PDFs, no model deps.

Cover the decisions that matter on the real corpus: born-digital rides its text
layer, image-only / partial-layer pages route to OCR, zero-width contamination is
stripped, and page rotation is captured.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz", reason="needs PyMuPDF")
import fitz  # noqa: E402

from fusion_ocr import config as config_mod  # noqa: E402
from fusion_ocr.models import Document  # noqa: E402
from fusion_ocr.stages.triage import Triage  # noqa: E402


def _run(path):
    doc = Document(source_path=str(path), sha256="x")
    return Triage().run(doc, config_mod.Config()).pages[0]


def test_born_digital_rides_text_layer(tmp_path):
    path = tmp_path / "born.pdf"
    d = fitz.open()
    pg = d.new_page()
    y = 72
    for _ in range(20):  # fill the page so coverage is real
        pg.insert_text((72, y), "The quick brown fox jumps over the lazy dog.", fontsize=12)
        y += 24
    d.save(str(path)); d.close()

    page = _run(path)
    assert page.has_text_layer is True
    assert page.needs_ocr is False
    assert [s for s in page.segments if s.source == "textlayer"]


def test_image_only_routes_to_ocr(tmp_path):
    # Flatten text into a full-page image -> no text layer.
    src = fitz.open(); sp = src.new_page()
    sp.insert_text((72, 200), "scanned content here", fontsize=28)
    pix = sp.get_pixmap(dpi=150); src.close()
    path = tmp_path / "scan.pdf"
    out = fitz.open(); pg = out.new_page()
    pg.insert_image(pg.rect, pixmap=pix)
    out.save(str(path)); out.close()

    page = _run(path)
    assert page.needs_ocr is True
    assert not [s for s in page.segments if s.source == "textlayer"]


def test_clean_strips_zero_width_and_pua_scoring():
    # Unit-test the helpers directly: PyMuPDF substitutes a visible glyph for an
    # inserted zero-width char, so a PDF round-trip can't reproduce the real
    # Mandelson contamination (genuine ZWSP in the text layer).
    from fusion_ocr.stages.triage import _clean, _pua_count

    contaminated = "in​voi​ce﻿ total"  # ZWSP + BOM
    assert _clean(contaminated) == "invoice total"

    # PUA codepoints (the Thai tone-mark case) are counted; normal text is not.
    assert _pua_count("abc") == 2
    assert _pua_count("ordinary text") == 0


def test_rotation_is_captured(tmp_path):
    path = tmp_path / "rot.pdf"
    d = fitz.open(); pg = d.new_page()
    pg.insert_text((72, 200), "rotated page", fontsize=20)
    pg.set_rotation(270)
    d.save(str(path)); d.close()

    assert _run(path).rotation == 270
