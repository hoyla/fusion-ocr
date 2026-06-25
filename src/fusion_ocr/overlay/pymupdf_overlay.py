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
                _write_invisible(fitz, pg, seg, granularity)
        pdf.save(str(out_path), garbage=4, deflate=True)
    return True


def _write_invisible(fitz, pg, seg, granularity: str) -> None:
    x0, y0, x1, y1 = seg.box.bbox
    # render_mode=3 -> invisible glyphs; the text is present for search/selection
    # but not drawn over the original image.
    if granularity == "word":
        # TODO: subdivide [x0,x1] across seg.best_text.split() for word-level boxes.
        words = seg.best_text.split()
        if words:
            step = (x1 - x0) / len(words)
            for i, w in enumerate(words):
                rect = fitz.Rect(x0 + i * step, y0, x0 + (i + 1) * step, y1)
                pg.insert_textbox(rect, w, render_mode=3, fontsize=max(y1 - y0, 1))
            return
    rect = fitz.Rect(x0, y0, x1, y1)
    pg.insert_textbox(rect, seg.best_text, render_mode=3, fontsize=max(y1 - y0, 1))
