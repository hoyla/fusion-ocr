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


def test_vision_path_tags_engine(tmp_path):
    pdf = tmp_path / "t.pdf"
    d = fitz.open(); d.new_page(width=612, height=792); d.save(str(pdf)); d.close()
    doc = Document(source_path=str(pdf), sha256="x")
    page = Page(index=0, needs_ocr=True, width=612, height=792)   # scanned -> vision
    page.regions = [Region(
        box=Box(points=[(50, 100), (300, 100), (300, 200), (50, 200)]), kind="table")]
    doc.pages = [page]
    Table(dpi=72, model=_FakeTableModel()).run(doc, config_mod.Config())
    assert doc.pages[0].regions[0].table_engine == "table_structure"


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


def test_overlap_frac():
    from fusion_ocr.stages.table import _overlap_frac
    a = (0, 0, 100, 100)
    assert _overlap_frac(a, (0, 0, 100, 100)) == 1.0      # identical
    assert _overlap_frac(a, (200, 200, 300, 300)) == 0.0  # disjoint
    assert _overlap_frac(a, (50, 0, 150, 100)) == 0.5     # half, equal areas
    assert _overlap_frac(a, (0, 0, 50, 50)) == 1.0        # smaller fully inside


def _synthetic_table_pdf(path):
    """A born-digital 2x2 bordered table find_tables can detect."""
    d = fitz.open(); pg = d.new_page(width=300, height=200)
    xs, ys = [40, 160, 280], [40, 90, 140]
    for x in xs:
        pg.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        pg.draw_line((xs[0], y), (xs[-1], y))
    for r, row in enumerate([("Name", "Qty"), ("Apple", "3")]):
        for c, txt in enumerate(row):
            pg.insert_text((xs[c] + 5, ys[r] + 30), txt, fontsize=11)
    d.save(str(path)); d.close()


def test_born_digital_uses_find_tables(tmp_path):
    # born-digital -> exact text-layer extraction, NO vision model needed/injected
    pdf = tmp_path / "born.pdf"; _synthetic_table_pdf(pdf)
    doc = Document(source_path=str(pdf), sha256="x")
    page = Page(index=0, needs_ocr=False, width=300, height=200)
    page.regions = [Region(
        box=Box(points=[(35, 35), (285, 35), (285, 145), (35, 145)]), kind="table")]
    doc.pages = [page]

    Table().run(doc, config_mod.Config())   # no model -> would fail if it hit vision

    r = doc.pages[0].regions[0]
    assert r.table_engine == "find_tables"               # exact path chosen
    assert "Apple" in r.table_html and "Qty" in r.table_html
    assert 'data-confidence="clean"' in r.table_html
    assert len(r.cells) >= 4                              # per-cell geometry from layer


def test_find_tables_gated_to_layout_region(tmp_path):
    # find_tables only claims a region that overlaps an actual detected table; one that
    # doesn't is left untouched by the find_tables path (tested in isolation, before the
    # vision fallback that the full run would apply to an unpopulated table region).
    pdf = tmp_path / "born.pdf"; _synthetic_table_pdf(pdf)
    page = Page(index=0, needs_ocr=False, width=300, height=200)
    page.regions = [Region(  # far from the table at (40,40)-(280,140)
        box=Box(points=[(10, 160), (60, 160), (60, 190), (10, 190)]), kind="table")]
    pg = fitz.open(str(pdf))[0]
    Table()._extract_find_tables(pg, page)
    assert page.regions[0].table_html == ""              # no overlap -> not claimed
    assert page.regions[0].table_engine == ""


def test_born_digital_find_tables_miss_falls_back_to_vision(tmp_path):
    # born-digital page but find_tables finds nothing (blank page) -> the table region
    # falls through to the vision engine, so nothing regresses.
    pdf = tmp_path / "blank.pdf"
    d = fitz.open(); d.new_page(width=300, height=200); d.save(str(pdf)); d.close()
    doc = Document(source_path=str(pdf), sha256="x")
    page = Page(index=0, needs_ocr=False, width=300, height=200)
    page.regions = [Region(
        box=Box(points=[(50, 50), (250, 50), (250, 150), (50, 150)]), kind="table")]
    doc.pages = [page]
    Table(dpi=72, model=_FakeTableModel()).run(doc, config_mod.Config())
    assert doc.pages[0].regions[0].table_engine == "table_structure"   # vision fallback
