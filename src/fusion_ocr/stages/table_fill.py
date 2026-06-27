"""Stage — Table-cell content fill.

The `table` stage gives each table region a structure (empty HTML grid) and per-cell
boxes. This stage — run AFTER fusion, when segments carry their final `best_text` —
populates the grid by assigning each cell the text of the segments inside it. The
result is a structured table whose every cell backs to a box and a source.

Only the vision grid (PaddleOCR) arrives with empty cells to fill. A find_tables grid
(`table_engine == "find_tables"`) is already exact — cell text came straight from the
layer — so it is skipped here (re-filling would double its attributes and re-introduce
the coarse line-intersection that find_tables avoids).
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
                      if r.kind == "table" and r.table_html and r.cells
                      and r.table_engine != "find_tables"]   # already exact, don't refill
            if not tables:
                continue
            segs = [s for s in page.segments if s.best_text and not s.superseded]
            for region in tables:
                region.table_html = populate_table_html(
                    region.table_html, region.cells, segs)
        return doc
