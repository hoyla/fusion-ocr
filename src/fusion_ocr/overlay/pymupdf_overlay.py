"""Build the searchable overlay PDF.

The OCRmyPDF-style technique, done ourselves so the overlay carries the *fused*
best-of text and so the line<->word granularity decision is ours: open the source
PDF, and for each Segment write its `best_text` as INVISIBLE text (render mode 3)
positioned inside the segment's box. Result: a visually identical PDF whose text is
selectable / searchable / highlightable.

`granularity="line"` writes one invisible string per segment box (MVP).
`granularity="word"` is the follow-on: subdivide each segment box across its words
before writing — isolated to this module plus a word-split helper.

Returns True if an overlay was written, False if it had nothing to do (no segments)
or PyMuPDF isn't installed — so the walking skeleton degrades cleanly.
"""

from __future__ import annotations

from pathlib import Path

from ..models import Document


def build_overlay(doc: Document, out_path: Path, granularity: str = "line") -> bool:
    segments = [s for p in doc.pages for s in p.segments if s.best_text]
    if not segments:
        return False
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False

    with fitz.open(doc.source_path) as pdf:
        for page in doc.pages:
            if page.index >= pdf.page_count:
                continue
            pg = pdf[page.index]
            for seg in page.segments:
                if not seg.best_text:
                    continue
                _write_invisible(fitz, pg, seg, granularity, page.rotation)
        pdf.save(str(out_path), garbage=4, deflate=True)
    return True


_FONT = "helv"  # base-14 (Latin-1). Full Unicode (Thai/CJK/diacritics) overlay
                # needs an embedded TTF — a known follow-up; markdown is already
                # full-Unicode from the VLM.


def _fit_fontsize(fitz, text: str, box_w: float, box_h: float) -> float:
    """Size the font so the line fills the box width without overflowing — the trick
    that makes insert_text actually place the whole string (insert_textbox silently
    drops text that doesn't fit at a fixed size)."""
    unit = fitz.get_text_length(text, fontname=_FONT, fontsize=1) or 1.0
    return max(1.0, min(box_h, box_w / unit))


def _write_invisible(fitz, pg, seg, granularity: str, rotation: int = 0) -> None:
    x0, y0, x1, y1 = seg.box.bbox
    box_w, box_h = (x1 - x0), (y1 - y0)
    # Reading runs along the box's long axis; on a rotated page width/height swap.
    span = box_w if rotation in (0, 180) else box_h

    def place(text, ax0, ay1, width):
        fs = _fit_fontsize(fitz, text, max(width, 1), box_h if rotation in (0, 180) else box_w)
        # render_mode=3 -> invisible glyphs; present for search/selection, not drawn.
        try:
            pg.insert_text(fitz.Point(ax0, ay1), text, fontname=_FONT, fontsize=fs,
                           render_mode=3, rotate=rotation)
        except Exception:
            pass  # non-Latin1 glyph in base font -> skip until TTF embedding lands

    if granularity == "word":
        words = seg.best_text.split()
        if words:
            step = span / len(words)
            for i, w in enumerate(words):
                place(w, x0 + i * step, y1, step)
            return
    place(seg.best_text, x0, y1, span)
