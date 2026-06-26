"""Stage — Table-cell extraction (PaddleOCR TableStructureRecognition).

For each `table` region the layout stage found, recover the DETERMINISTIC table
structure: an HTML grid (`region.table_html`) and per-cell boxes (`region.cells`, in
page coordinates). This sits alongside the VLM's table reading — the VLM gives cell
*content* as markdown/HTML; this gives auditable cell *geometry* (for per-cell
highlighting and a structured export), independent of the model.

Crops each table region from the rendered page and maps the returned cell polygons
back to page space. Rotated pages are skipped for now (crop-on-rotated is a refinement;
the table reading still comes through the VLM). Clean passthrough without PaddleOCR.
"""

from __future__ import annotations

from ..config import Config
from ..models import Box, Document

_DPI = 150


class Table:
    name = "table"

    def __init__(self, dpi: int = _DPI, model=None) -> None:
        self.dpi = dpi
        self._model = model  # injectable for tests

    def _table_model(self):
        if self._model is None:
            from paddleocr import TableStructureRecognition
            self._model = TableStructureRecognition()
        return self._model

    def run(self, doc: Document, cfg: Config) -> Document:
        targets = [p for p in doc.pages
                   if p.needs_ocr and not p.rotation
                   and any(r.kind == "table" for r in p.regions)]
        if not targets:
            return doc
        try:
            import fitz  # PyMuPDF
            import numpy as np
        except ImportError:
            return doc
        try:
            model = self._table_model()
        except ImportError:
            return doc

        scale = self.dpi / 72.0
        with fitz.open(doc.source_path) as pdf:
            for page in targets:
                if page.index >= pdf.page_count:
                    continue
                pix = pdf[page.index].get_pixmap(dpi=self.dpi)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                if pix.n == 4:
                    arr = arr[:, :, :3]
                elif pix.n == 1:
                    arr = np.repeat(arr, 3, axis=2)
                img = np.ascontiguousarray(arr)
                H, W = img.shape[:2]

                for region in page.regions:
                    if region.kind != "table":
                        continue
                    x0, y0, x1, y1 = region.box.bbox
                    px0, py0 = max(0, int(x0 * scale)), max(0, int(y0 * scale))
                    px1, py1 = min(W, int(x1 * scale)), min(H, int(y1 * scale))
                    if px1 - px0 < 8 or py1 - py0 < 8:
                        continue
                    crop = np.ascontiguousarray(img[py0:py1, px0:px1])
                    self._extract(model, crop, region, px0, py0, scale)
        return doc

    def _extract(self, model, crop, region, ox, oy, scale) -> None:
        res = model.predict(crop)
        if not res:
            return
        r = res[0]
        struct = r.get("structure", []) if hasattr(r, "get") else []
        region.table_html = "".join(struct)
        cells = []
        for cb in (r.get("bbox", []) if hasattr(r, "get") else []):
            xs, ys = cb[0::2], cb[1::2]
            cx0, cy0 = (ox + min(xs)) / scale, (oy + min(ys)) / scale
            cx1, cy1 = (ox + max(xs)) / scale, (oy + max(ys)) / scale
            cells.append(Box(points=[(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)]))
        region.cells = cells
