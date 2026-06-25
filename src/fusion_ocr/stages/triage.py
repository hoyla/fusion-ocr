"""Stage 1 — Triage.

REAL IMPL (extra: ocr / pymupdf): open the PDF, and for each page detect whether a
usable embedded text layer exists. Born-digital pages get their char-level text +
boxes lifted straight from PyMuPDF (perfect geometry, no OCR) and become `textlayer`
segments; image-only pages are rasterised and routed to the OCR + VLM tracks.

WALKING SKELETON: create one Page per PDF page if PyMuPDF is available, else a
single placeholder page, so the rest of the plumbing has something to carry.
"""

from __future__ import annotations

from ..config import Config
from ..models import Document, Page


class Triage:
    name = "triage"

    def run(self, doc: Document, cfg: Config) -> Document:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            if not doc.pages:
                doc.pages = [Page(index=0)]
            return doc

        with fitz.open(doc.source_path) as pdf:
            doc.pages = []
            for i, page in enumerate(pdf):
                rect = page.rect
                has_text = bool(page.get_text("text").strip())
                doc.pages.append(
                    Page(
                        index=i,
                        width=rect.width,
                        height=rect.height,
                        has_text_layer=has_text,
                    )
                )
        return doc
