"""Stage 2 — Layout / structure (PP-DocLayout via PaddleOCR LayoutDetection).

Detects layout regions per OCR-bound page — paragraph / table / figure / header /
footer — and stores them on `page.regions` with a reading order. Two downstream uses:

  * fusion clusters OCR boxes WITHIN a region, so multi-column / side-by-side text
    doesn't get merged into one line (the columns nightmare);
  * `table` regions are flagged for table-aware handling.

Geometry is mapped back through the page derotation matrix (like ocr_det), so region
boxes are in the PDF's native space. Reading order here is a simple geometric pass
(row-major); a true multi-column reading-order model (PP-StructureV3) is a follow-up —
the region-aware clustering is the main win and doesn't depend on perfect ordering.

Clean passthrough if PaddleOCR / PyMuPDF aren't installed.
"""

from __future__ import annotations

from ..config import Config
from ..models import Box, Document, Region

_DPI = 150

# PP-DocLayout labels -> our RegionKind.
_KIND_MAP = {
    "table": "table",
    "figure": "figure", "image": "figure", "chart": "figure", "seal": "figure",
    "header": "header", "doc_title": "header", "title": "header",
    "paragraph_title": "header", "figure_title": "header", "table_title": "header",
    "abstract_title": "header", "content_title": "header",
    "footer": "footer", "number": "footer", "footnote": "footer",
    "text": "paragraph", "abstract": "paragraph", "content": "paragraph",
    "reference": "paragraph", "formula": "paragraph",
}


class Layout:
    name = "layout"

    def __init__(self, dpi: int = _DPI) -> None:
        self.dpi = dpi
        self._model = None

    def _layout_model(self):
        if self._model is None:
            from paddleocr import LayoutDetection
            self._model = LayoutDetection()
        return self._model

    def run(self, doc: Document, cfg: Config) -> Document:
        try:
            import fitz  # PyMuPDF
            import numpy as np
        except ImportError:
            return doc
        targets = [p for p in doc.pages if p.needs_ocr]
        if not targets:
            return doc
        try:
            model = self._layout_model()
        except ImportError:
            return doc

        scale = self.dpi / 72.0
        with fitz.open(doc.source_path) as pdf:
            for page in targets:
                if page.index >= pdf.page_count:
                    continue
                pg = pdf[page.index]
                deroter = pg.derotation_matrix
                pix = pg.get_pixmap(dpi=self.dpi)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                if pix.n == 4:
                    arr = arr[:, :, :3]
                elif pix.n == 1:
                    arr = np.repeat(arr, 3, axis=2)
                img = np.ascontiguousarray(arr)

                regions = []
                for b in self._detect(model, img):
                    x0, y0, x1, y1 = b["coordinate"]
                    p0 = fitz.Point(x0 / scale, y0 / scale) * deroter
                    p1 = fitz.Point(x1 / scale, y1 / scale) * deroter
                    bx0, bx1 = sorted((p0.x, p1.x))
                    by0, by1 = sorted((p0.y, p1.y))
                    regions.append(Region(
                        box=Box(points=[(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]),
                        kind=_KIND_MAP.get(str(b.get("label", "")).lower(), "other"),
                    ))
                page.regions = _order_regions(regions)
        return doc

    @staticmethod
    def _detect(model, img):
        res = model.predict(img)
        if not res:
            return []
        r = res[0]
        return r.get("boxes", []) if hasattr(r, "get") else []


def _order_regions(regions: list[Region]) -> list[Region]:
    """Row-major reading order: top-to-bottom, then left-to-right within a band.
    Approximate for true multi-column body text (PP-StructureV3 reading-order model is
    the proper fix); good enough to order regions for the markdown view."""
    ordered = sorted(regions, key=lambda r: (round(r.box.bbox[1] / 20), r.box.bbox[0]))
    for i, r in enumerate(ordered):
        r.reading_order = i
    return ordered
