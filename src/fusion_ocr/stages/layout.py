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
                pix = pg.get_pixmap(dpi=self.dpi)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                if pix.n == 4:
                    arr = arr[:, :, :3]
                elif pix.n == 1:
                    arr = np.repeat(arr, 3, axis=2)
                img = np.ascontiguousarray(arr)

                regions, disp_boxes = [], []
                for b in self._detect(model, img):
                    x0, y0, x1, y1 = b["coordinate"]
                    # displayed (upright) box — what the eye reads, used for the XY-cut
                    # reading order so it's correct on rotated pages.
                    disp_boxes.append(Box(points=[
                        (x0 / scale, y0 / scale), (x1 / scale, y0 / scale),
                        (x1 / scale, y1 / scale), (x0 / scale, y1 / scale)]))
                    # base (derotated) box — stored, so overlay/search land on the page
                    p0 = fitz.Point(x0 / scale, y0 / scale) * deroter
                    p1 = fitz.Point(x1 / scale, y1 / scale) * deroter
                    bx0, bx1 = sorted((p0.x, p1.x))
                    by0, by1 = sorted((p0.y, p1.y))
                    regions.append(Region(
                        box=Box(points=[(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]),
                        kind=_KIND_MAP.get(str(b.get("label", "")).lower(), "other"),
                    ))
                page.regions = _order_regions(regions, disp_boxes)
        return doc

    @staticmethod
    def _detect(model, img):
        res = model.predict(img)
        if not res:
            return []
        r = res[0]
        return r.get("boxes", []) if hasattr(r, "get") else []


def _order_regions(regions: list[Region], order_boxes: list[Box] | None = None) -> list[Region]:
    """Reading order via XY-cut (handles multi-column / header+columns / tables), the
    same approach PP-StructureV3 uses — deterministic and explainable. `order_boxes`
    (displayed/upright space) is used for ordering when given, so the order is visually
    correct on rotated pages even though the stored region boxes are in base space."""
    from ..compose import xy_cut_order

    if not regions:
        return regions
    boxes = order_boxes if order_boxes is not None else [r.box for r in regions]
    ordered = [regions[i] for i in xy_cut_order(boxes)]
    for i, r in enumerate(ordered):
        r.reading_order = i
    return ordered
