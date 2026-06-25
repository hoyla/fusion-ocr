"""Stage 1 — Triage (per-region, quality-aware).

For each page we pull the embedded text layer line-by-line and decide, per page,
whether OCR is still needed. Real corpus findings this handles:

  * born-digital pages  -> clean text lifted straight from the layer (no OCR)
  * Mandelson-style docs -> zero-width-space contamination stripped
  * Thai cert (well-made but ~4% private-use-area glyphs) -> flagged for OCR repair
  * Thai scan (machine-readable header/footer over a scanned body) -> header/footer
    kept as exact text, page flagged for OCR so the body gets read

So a single page can emit exact `textlayer` segments AND still be marked
`needs_ocr` — fusion later arbitrates any spatial overlap.

The OCR decision uses three signals: no text at all, private-use-area contamination
above threshold, or a large raster image with low text coverage (a scan with only a
partial text layer). Pure char-count is deliberately NOT the gate — a header/footer
can be "dense" yet leave the whole body unread.
"""

from __future__ import annotations

from ..config import Config
from ..models import Box, Document, Page, Segment

# Zero-width / BOM chars that contaminate otherwise-clean born-digital text.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)

_PUA_MAX = 0.01          # >1% private-use chars -> layer unreliable, re-OCR
_LARGE_IMAGE_FRAC = 0.40 # an image covering >40% of the page = likely a scan
_BODY_COVERAGE_MIN = 0.50  # text-box area below this over a big image -> OCR


def _clean(s: str) -> str:
    return s.translate(_ZERO_WIDTH)


def _pua_count(s: str) -> int:
    return sum(1 for c in s if 0xE000 <= ord(c) <= 0xF8FF and not c.isspace())


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
                doc.pages.append(self._triage_page(fitz, page, i))
        return doc

    def _triage_page(self, fitz, page, index: int) -> Page:
        rect = page.rect
        page_area = max(rect.width * rect.height, 1.0)
        P = Page(index=index, width=rect.width, height=rect.height,
                 rotation=page.rotation)

        non_space = 0
        pua = 0
        text_area = 0.0
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                text = _clean("".join(sp.get("text", "") for sp in line.get("spans", [])))
                if not text.strip():
                    continue
                x0, y0, x1, y1 = line["bbox"]
                stripped = [c for c in text if not c.isspace()]
                line_pua = _pua_count(text)
                non_space += len(stripped)
                pua += line_pua
                text_area += abs((x1 - x0) * (y1 - y0))

                clean = (line_pua / len(stripped)) <= _PUA_MAX if stripped else True
                seg = Segment(
                    id=f"p{index}-t{len(P.segments)}",
                    page=index,
                    box=Box(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
                    det_text=text,
                    det_conf=1.0 if clean else 0.3,
                    source="textlayer",
                )
                if clean:
                    seg.best_text = text  # exact glyphs — trust the layer
                P.segments.append(seg)

        # Page-level OCR decision.
        page_pua = (pua / non_space) if non_space else 0.0
        coverage = text_area / page_area
        large_image = self._max_image_frac(page, page_area) > _LARGE_IMAGE_FRAC

        P.has_text_layer = non_space > 0
        P.needs_ocr = (
            non_space == 0                                   # pure image
            or page_pua > _PUA_MAX                           # contaminated layer
            or (large_image and coverage < _BODY_COVERAGE_MIN)  # scan w/ partial layer
        )
        return P

    @staticmethod
    def _max_image_frac(page, page_area: float) -> float:
        biggest = 0.0
        try:
            for img in page.get_images(full=True):
                for r in page.get_image_rects(img[0]):
                    biggest = max(biggest, abs(r.width * r.height))
        except Exception:
            return 0.0
        return biggest / page_area
