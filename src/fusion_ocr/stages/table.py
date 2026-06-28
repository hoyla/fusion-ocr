"""Stage — table structure + content for each `table` region, two engines by page class.

  - BORN-DIGITAL (text layer authoritative) -> PyMuPDF ``find_tables``: exact cell text
    AND geometry straight from the layer — no rasterise, no OCR, no model. Decisively
    better than vision on dense financial tables (every digit exact; the vision+line-fill
    path comes out mostly "spanning" and smushes adjacent columns). Gated to layout
    `table` regions, which also filters find_tables' own false positives (a coloured
    diagram is not a layout table).
  - SCANNED -> PaddleOCR vision: classify the table wired/wireless (TableClassification)
    and recover its grid with the matching SLANeXt (the current structure model
    PP-StructureV3 uses); cells filled later by TableFill, content read by the VLM
    (table_read). All public PaddleOCR predictors — no reimplemented layout logic.

Geometry stays deterministic either way; `region.table_engine` records which produced
it (provenance: exact vs OCR'd). Rotated pages skipped for now. Clean passthrough if
neither engine is importable. A find_tables miss falls through to the vision path, so
nothing regresses.
"""

from __future__ import annotations

from .. import raster
from ..compose import grid_to_table_html
from ..config import Config
from ..models import Box, Document, Page

_DPI = 150
_MIN_OVERLAP = 0.30   # a find_tables table must cover this fraction of the layout region


def _overlap_frac(a, b) -> float:
    """Intersection area / smaller box area — how much two bboxes coincide."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1.0)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
    return inter / min(area_a, area_b)


def _box(bb) -> Box | None:
    if not bb:
        return None
    x0, y0, x1, y1 = bb
    return Box(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


class Table:
    name = "table"

    def __init__(self, dpi: int = _DPI, model=None) -> None:
        self.dpi = dpi
        self._model = model        # injected structure model -> used for all crops (tests)
        self._cls = None           # wired/wireless classifier (lazy)
        self._structure: dict = {}  # variant -> SLANeXt model (lazy, per variant)

    def _classifier(self):
        if self._cls is None:
            from paddleocr import TableClassification
            self._cls = TableClassification()
        return self._cls

    def _structure_model(self, variant: str):
        """SLANeXt for the table type — the current structure model (what PP-StructureV3
        uses), same structure+bbox output as the old SLANet default. Loaded per variant
        on first use; an injected model overrides (tests)."""
        if self._model is not None:
            return self._model
        if variant not in self._structure:
            from paddleocr import TableStructureRecognition
            self._structure[variant] = TableStructureRecognition(
                model_name=f"SLANeXt_{variant}")
        return self._structure[variant]

    def _variant_for(self, crop) -> str:
        """wired (ruled) vs wireless (borderless), via PaddleOCR's public
        TableClassification, so the right SLANeXt is used. Defaults to wired on any issue."""
        try:
            d = self._classifier().predict(crop)[0]
            names = d.get("label_names", []) if hasattr(d, "get") else []
            scores = d.get("scores", []) if hasattr(d, "get") else []
            if len(names) and len(scores):
                top = names[max(range(len(scores)), key=lambda i: scores[i])]
                return "wireless" if "wireless" in str(top) else "wired"
        except Exception:
            pass
        return "wired"

    def run(self, doc: Document, cfg: Config) -> Document:
        targets = [p for p in doc.pages
                   if not p.rotation and any(r.kind == "table" for r in p.regions)]
        if not targets:
            return doc
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return doc

        with fitz.open(doc.source_path) as pdf:
            for page in targets:
                if page.index >= pdf.page_count:
                    continue
                if not page.needs_ocr:                       # born-digital -> exact
                    self._extract_find_tables(pdf[page.index], page)
                remaining = [r for r in page.regions
                             if r.kind == "table" and not r.table_html]
                if remaining:                                # scanned / find_tables miss
                    self._extract_vision(pdf[page.index], page, remaining)
        return doc

    # ---- born-digital: PyMuPDF find_tables (exact, text-layer) ----------------

    def _extract_find_tables(self, pg, page: Page) -> None:
        try:
            found = pg.find_tables().tables
        except Exception:
            return
        if not found:
            return
        for region in page.regions:
            if region.kind != "table" or region.table_html:
                continue
            best, best_f = None, _MIN_OVERLAP
            for t in found:
                f = _overlap_frac(region.box.bbox, t.bbox)
                if f > best_f:
                    best, best_f = t, f
            if best is None:
                continue
            html, cells = grid_to_table_html(self._grid(best))
            if "<td" not in html:
                continue
            region.table_html = html
            region.cells = cells
            region.table_engine = "find_tables"

    @staticmethod
    def _grid(t) -> list[list[tuple[str, Box | None]]]:
        """Row-major (text, box) grid from a find_tables Table. An external header (one
        detected above the body) is prepended as a text row so it isn't lost."""
        rows: list[list[tuple[str, Box | None]]] = []
        hdr = getattr(t, "header", None)
        if hdr is not None and getattr(hdr, "external", False):
            names = getattr(hdr, "names", None) or []
            if any((n or "").strip() for n in names):
                rows.append([((n or ""), None) for n in names])
        box_rows = t.rows
        for ri, trow in enumerate(t.extract()):
            brow = box_rows[ri].cells if ri < len(box_rows) else []
            rows.append([((txt or ""), _box(brow[ci] if ci < len(brow) else None))
                         for ci, txt in enumerate(trow)])
        return rows

    # ---- scanned: PaddleOCR TableStructureRecognition (vision) ----------------

    def _extract_vision(self, pg, page: Page, regions) -> None:
        try:
            import numpy as np
        except ImportError:
            return
        if self._model is None:                  # need the real predictors
            try:
                from paddleocr import TableClassification, TableStructureRecognition  # noqa: F401
            except ImportError:
                return
        scale = self.dpi / 72.0
        img = raster.page_ndarray(pg.parent, page.index, self.dpi)
        H, W = img.shape[:2]
        for region in regions:
            x0, y0, x1, y1 = region.box.bbox
            px0, py0 = max(0, int(x0 * scale)), max(0, int(y0 * scale))
            px1, py1 = min(W, int(x1 * scale)), min(H, int(y1 * scale))
            if px1 - px0 < 8 or py1 - py0 < 8:
                continue
            crop = np.ascontiguousarray(img[py0:py1, px0:px1])
            variant = "wired" if self._model is not None else self._variant_for(crop)
            self._extract(self._structure_model(variant), crop, region, px0, py0, scale)

    def _extract(self, model, crop, region, ox, oy, scale) -> None:
        res = model.predict(crop)
        if not res:
            return
        r = res[0]
        struct = r.get("structure", []) if hasattr(r, "get") else []
        region.table_html = "".join(struct)
        region.table_engine = "table_structure"
        cells = []
        for cb in (r.get("bbox", []) if hasattr(r, "get") else []):
            xs, ys = cb[0::2], cb[1::2]
            cx0, cy0 = (ox + min(xs)) / scale, (oy + min(ys)) / scale
            cx1, cy1 = (ox + max(xs)) / scale, (oy + max(ys)) / scale
            cells.append(Box(points=[(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)]))
        region.cells = cells
