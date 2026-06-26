"""Build the searchable overlay PDF.

The OCRmyPDF-style technique, done ourselves so the overlay carries the *fused*
best-of text and so the line<->word granularity decision is ours: open the source
PDF, and for each Segment write its `best_text` as INVISIBLE text (render mode 3)
positioned inside the segment's box. Result: a visually identical PDF whose text is
selectable / searchable / highlightable.

FONT: the invisible text needs a font that actually covers the script, or search
breaks. The base-14 "helv" font can't encode Thai/CJK/Arabic — the glyphs round-trip
to *some* Unicode but `search_for` then misses (verified: 0/4 Thai terms). A broad
Unicode font (Arial Unicode, or a configured TTF) fixes it (4/4). Text is NFC-
normalised for good measure. Falls back to helv (Latin only) if no Unicode font is
found, so it still runs anywhere.

`granularity="line"` writes one invisible string per segment box (MVP).
`granularity="word"` is the follow-on: subdivide each box across its words.

Returns True if an overlay was written, False if nothing to do / PyMuPDF absent.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

from ..models import Document

# Broad-coverage Unicode fonts to look for (first hit wins). Arial Unicode (macOS)
# covers Thai/Cyrillic/CJK/Arabic/Latin; the Noto paths are common Linux/VPC spots.
_FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _resolve_font(font_path: str | None) -> str | None:
    """Return a usable Unicode TTF path, or None to fall back to base-14 helv."""
    if font_path and os.path.exists(font_path):
        return font_path
    for cand in _FONT_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return None


def build_overlay(doc: Document, out_path: Path, granularity: str = "line",
                  font_path: str | None = None) -> bool:
    segments = [s for p in doc.pages for s in p.segments
                if s.best_text and not s.superseded]
    if not segments:
        return False
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False

    fontfile = _resolve_font(font_path)
    fontname = "uni" if fontfile else "helv"
    font = fitz.Font(fontfile=fontfile) if fontfile else fitz.Font("helv")

    with fitz.open(doc.source_path) as pdf:
        for page in doc.pages:
            if page.index >= pdf.page_count:
                continue
            pg = pdf[page.index]
            for seg in page.segments:
                if not seg.best_text or seg.superseded:
                    continue
                _write_invisible(fitz, pg, seg, granularity, page.rotation,
                                 font, fontname, fontfile)
        pdf.save(str(out_path), garbage=4, deflate=True)
    return True


def _fit_fontsize(font, text: str, box_w: float, box_h: float) -> float:
    """Size the font so the line fills the box width without overflowing — the trick
    that makes the whole string land (a fixed size silently drops overflow)."""
    unit = font.text_length(text, fontsize=1) or 1.0
    return max(1.0, min(box_h, box_w / unit))


def _write_invisible(fitz, pg, seg, granularity: str, rotation: int,
                     font, fontname: str, fontfile: str | None) -> None:
    text = unicodedata.normalize("NFC", seg.best_text)
    x0, y0, x1, y1 = seg.box.bbox
    box_w, box_h = (x1 - x0), (y1 - y0)
    span = box_w if rotation in (0, 180) else box_h
    thickness_dim = box_h if rotation in (0, 180) else box_w

    def place(t: str, ax0: float, ay1: float, width: float) -> None:
        fs = _fit_fontsize(font, t, max(width, 1), thickness_dim)
        # render_mode=3 -> invisible glyphs; present for search/selection, not drawn.
        try:
            pg.insert_text(fitz.Point(ax0, ay1), t, fontname=fontname,
                           fontfile=fontfile, fontsize=fs, render_mode=3,
                           rotate=rotation)
        except Exception:
            pass  # a glyph the font can't encode -> skip that fragment

    if granularity == "word":
        words = text.split()
        if words:
            step = span / len(words)
            for i, w in enumerate(words):
                place(w, x0 + i * step, y1, step)
            return
    place(text, x0, y1, span)
