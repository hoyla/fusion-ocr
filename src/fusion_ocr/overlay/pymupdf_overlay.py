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


def _write_invisible(fitz, pg, seg, granularity: str, rotation: int = 0) -> None:
    x0, y0, x1, y1 = seg.box.bbox
    # The glyph height runs perpendicular to the reading direction, which swaps on a
    # rotated page. `rotate=` makes the invisible text read along the line.
    thickness = (y1 - y0) if rotation in (0, 180) else (x1 - x0)
    fontsize = max(thickness, 1)
    # render_mode=3 -> invisible glyphs; present for search/selection, not drawn.
    if granularity == "word":
        # TODO: subdivide across words for word-level boxes (selection fidelity).
        words = seg.best_text.split()
        if words:
            step = (x1 - x0) / len(words)
            for i, w in enumerate(words):
                rect = fitz.Rect(x0 + i * step, y0, x0 + (i + 1) * step, y1)
                pg.insert_textbox(rect, w, render_mode=3, rotate=rotation,
                                  fontsize=fontsize)
            return
    rect = fitz.Rect(x0, y0, x1, y1)
    pg.insert_textbox(rect, seg.best_text, render_mode=3, rotate=rotation,
                      fontsize=fontsize)
