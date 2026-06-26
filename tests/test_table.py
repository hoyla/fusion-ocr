"""Table stage — structure + cell-box extraction (mock model, no deps)."""

from __future__ import annotations

import pytest

pytest.importorskip("fitz", reason="needs PyMuPDF")
import fitz  # noqa: E402

from fusion_ocr import config as config_mod  # noqa: E402
from fusion_ocr.models import Box, Document, Page, Region  # noqa: E402
from fusion_ocr.stages.table import Table  # noqa: E402


class _FakeTableModel:
    """Returns a fixed 1-cell structure; bbox in crop-pixel coords (quad)."""
    def predict(self, crop):
        return [{
            "structure": ["<html>", "<body>", "<table>", "<tr>", "<td>", "</td>",
                          "</tr>", "</table>", "</body>", "</html>"],
            "bbox": [[10, 10, 50, 10, 50, 30, 10, 30]],
        }]


def test_table_extracts_html_and_maps_cells(tmp_path):
    pdf = tmp_path / "t.pdf"
    d = fitz.open(); d.new_page(width=612, height=792); d.save(str(pdf)); d.close()

    doc = Document(source_path=str(pdf), sha256="x")
    page = Page(index=0, needs_ocr=True, width=612, height=792)
    page.regions = [Region(
        box=Box(points=[(50, 100), (300, 100), (300, 200), (50, 200)]), kind="table")]
    doc.pages = [page]

    # dpi=72 -> scale=1, so page-points == image-pixels (easy to verify mapping)
    Table(dpi=72, model=_FakeTableModel()).run(doc, config_mod.Config())

    r = doc.pages[0].regions[0]
    assert "<table>" in r.table_html
    assert len(r.cells) == 1
    # region offset (50,100) + cell crop coords (10,10)-(50,30) -> page (60,110)-(100,130)
    assert tuple(round(v) for v in r.cells[0].bbox) == (60, 110, 100, 130)


def test_table_skips_non_table_regions(tmp_path):
    pdf = tmp_path / "t.pdf"
    d = fitz.open(); d.new_page(); d.save(str(pdf)); d.close()
    doc = Document(source_path=str(pdf), sha256="x")
    page = Page(index=0, needs_ocr=True)
    page.regions = [Region(box=Box(points=[(0, 0), (10, 0), (10, 10), (0, 10)]),
                           kind="paragraph")]
    doc.pages = [page]
    Table(dpi=72, model=_FakeTableModel()).run(doc, config_mod.Config())
    assert doc.pages[0].regions[0].table_html == ""  # untouched
