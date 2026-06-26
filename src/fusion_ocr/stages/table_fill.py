"""Stage — Table-cell content fill.

The `table` stage gives each table region a structure (empty HTML grid) and per-cell
boxes. This stage — run AFTER fusion, when segments carry their final `best_text` —
populates the grid by assigning each cell the text of the segments inside it. The
result is a structured table whose every cell backs to a box and a source.

Works for scanned tables (OCR/VLM text) and born-digital tables (text-layer text)
alike, since both arrive as segments with best_text.
"""

from __future__ import annotations

from ..compose import populate_table_html
from ..config import Config
from ..models import Document


class TableFill:
    name = "table_fill"

    def run(self, doc: Document, cfg: Config) -> Document:
        for page in doc.pages:
            tables = [r for r in page.regions
                      if r.kind == "table" and r.table_html and r.cells]
            if not tables:
                continue
            segs = [s for s in page.segments if s.best_text and not s.superseded]
            for region in tables:
                region.table_html = populate_table_html(
                    region.table_html, region.cells, segs)
        return doc
