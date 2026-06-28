"""Stage 2 — Layout / structure (PP-DocLayout via PaddleOCR LayoutDetection).

Detects layout regions per OCR-bound page — paragraph / table / figure / header /
footer — and stores them on `page.regions` with a reading order. Two downstream uses:

  * fusion clusters OCR boxes WITHIN a region, so multi-column / side-by-side text
    doesn't get merged into one line (the columns nightmare);
  * `table` regions are flagged for table-aware handling.

Geometry is mapped back through the page derotation matrix (like ocr_det), so region
boxes are in the PDF's native space. Reading order comes from the model itself —
PP-DocLayoutV2 predicts a per-region reading `order` (a learned head, what PP-StructureV3
uses), so we no longer hand-roll an XY-cut. The model runs on the upright (displayed)
raster, so its order is already correct on rotated pages; furniture it leaves unordered
(running headers, page numbers) is placed by position.

Clean passthrough if PaddleOCR / PyMuPDF aren't installed.
"""

from __future__ import annotations

from .. import raster
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
            # PP-DocLayoutV2 (vs the default plus-L): same detection, PLUS a per-region
            # reading-order head — so ordering comes from a maintained model, not our XY-cut.
            self._model = LayoutDetection(model_name="PP-DocLayoutV2")
        return self._model

    def run(self, doc: Document, cfg: Config) -> Document:
        try:
            import fitz  # PyMuPDF
            import numpy as np  # noqa: F401 — gate the ocr extra before raster.page_ndarray
        except ImportError:
            return doc
        # all content pages, incl. born-digital — so born-digital tables get a grid too
        targets = [p for p in doc.pages if p.needs_ocr or p.has_text_layer]
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
                img = raster.page_ndarray(pdf, page.index, self.dpi)

                ranked = []
                for b in self._detect(model, img):
                    x0, y0, x1, y1 = b["coordinate"]
                    # base (derotated) box — stored, so overlay/search land on the page
                    p0 = fitz.Point(x0 / scale, y0 / scale) * deroter
                    p1 = fitz.Point(x1 / scale, y1 / scale) * deroter
                    bx0, bx1 = sorted((p0.x, p1.x))
                    by0, by1 = sorted((p0.y, p1.y))
                    region = Region(
                        box=Box(points=[(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]),
                        kind=_KIND_MAP.get(str(b.get("label", "")).lower(), "other"),
                    )
                    # order from the model (on the upright raster -> correct when rotated)
                    ranked.append((_rank(b.get("order"), (y0 + y1) / 2, img.shape[0]), region))
                ranked.sort(key=lambda t: t[0])
                page.regions = [r for _, r in ranked]
                for i, r in enumerate(page.regions):
                    r.reading_order = i
        return doc

    @staticmethod
    def _detect(model, img):
        res = model.predict(img)
        if not res:
            return []
        r = res[0]
        return r.get("boxes", []) if hasattr(r, "get") else []


def _rank(order, cy: float, page_h: float):
    """Sort key from PP-DocLayoutV2's predicted reading order. In-flow regions carry an
    integer `order`; furniture (running headers, page numbers) come back as None, so
    place top furniture first and bottom furniture last, by vertical position. If the
    model emits no order at all (an older layout model), everything is None and this
    degrades to a plain top-to-bottom pass."""
    if order is not None:
        return (1, float(order))
    return (0 if cy < 0.15 * page_h else 2, cy)
