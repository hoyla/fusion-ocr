"""Table-cell content fill — assign segment text to cells, populate the grid, render.

No deps: pure helpers + the TableFill stage + the render markdown helper.
"""

from __future__ import annotations

from fusion_ocr import config as config_mod
from fusion_ocr.compose import cell_confidence, cell_text, populate_table_html
from fusion_ocr.models import Box, Document, Page, Region, Segment
from fusion_ocr.stages.render import _page_markdown
from fusion_ocr.stages.table_fill import TableFill


def _box(x0, y0, x1, y1):
    return Box(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _seg(id, x0, y0, x1, y1, best="", source="fused"):
    s = Segment(id=id, page=0, box=_box(x0, y0, x1, y1), source=source)
    s.best_text = best
    return s


# ---- cell content assignment ---------------------------------------------

def test_cell_text_picks_only_inside_segments():
    cell = _box(0, 0, 50, 20)
    inside = _seg("a", 5, 5, 45, 15, best="hello")
    outside = _seg("b", 60, 5, 95, 15, best="world")
    assert cell_text(cell, [inside, outside]) == "hello"


def test_cell_text_joins_in_reading_order():
    cell = _box(0, 0, 100, 60)
    # two lines in the same cell, given out of order
    lower = _seg("lo", 5, 40, 95, 55, best="second")
    upper = _seg("up", 5, 5, 95, 20, best="first")
    assert cell_text(cell, [lower, upper]) == "first second"


# ---- grid population ------------------------------------------------------

def test_populate_table_html_fills_cells_in_order_and_escapes():
    html = "<table><tbody><tr><td></td><td></td></tr></tbody></table>"
    cells = [_box(0, 0, 50, 20), _box(50, 0, 100, 20)]
    segs = [_seg("a", 5, 5, 45, 15, best="A & B"), _seg("b", 55, 5, 95, 15, best="C")]
    out = populate_table_html(html, cells, segs)
    assert ">A &amp; B</td>" in out         # escaped
    assert ">C</td>" in out
    assert 'data-confidence="clean"' in out  # confidence surfaced
    assert out.index("A &amp; B") < out.index("C")   # order preserved


def test_populate_tolerates_cell_count_mismatch():
    html = "<table><tr><td></td><td></td><td></td></tr></table>"
    cells = [_box(0, 0, 10, 10)]            # fewer cells than <td>s
    out = populate_table_html(html, cells, [_seg("a", 1, 1, 9, 9, best="X")])
    assert out.count("</td>") == 3 and ">X</td>" in out   # filled what it could


# ---- calibration: cell confidence ----------------------------------------

def test_cell_confidence_clean_spanning_empty():
    cell = _box(0, 0, 100, 40)
    inside = _seg("in", 5, 5, 95, 35, best="contained")        # ~mostly inside
    straddle = _seg("sp", 50, 5, 250, 35, best="label value")  # extends well beyond
    assert cell_confidence(cell, [inside]) == "clean"
    assert cell_confidence(cell, [straddle]) == "spanning"
    assert cell_confidence(cell, []) == "empty"
    # a neighbouring cell the straddling segment crosses is ALSO flagged spanning
    neighbour = _box(100, 0, 200, 40)
    assert cell_confidence(neighbour, [straddle]) == "spanning"


def test_populate_flags_spanning_cells():
    # two cells, one segment spanning both -> both cells flagged spanning
    html = "<table><tr><td></td><td></td></tr></table>"
    cells = [_box(0, 0, 50, 20), _box(50, 0, 100, 20)]
    spanning = _seg("s", 10, 5, 90, 15, best="label value")    # crosses the divide
    out = populate_table_html(html, cells, [spanning])
    assert out.count('data-confidence="spanning"') == 2


# ---- the stage ------------------------------------------------------------

def _table_doc():
    page = Page(index=0, width=200, height=100)
    region = Region(box=_box(0, 0, 100, 40), kind="table")
    region.reading_order = 0
    region.table_html = "<html><body><table><tbody><tr><td></td><td></td></tr>" \
                        "<tr><td></td><td></td></tr></tbody></table></body></html>"
    region.cells = [_box(0, 0, 50, 20), _box(50, 0, 100, 20),
                    _box(0, 20, 50, 40), _box(50, 20, 100, 40)]
    page.regions = [region]
    page.segments = [
        _seg("c0", 5, 5, 45, 15, best="r1c1"), _seg("c1", 55, 5, 95, 15, best="r1c2"),
        _seg("c2", 5, 25, 45, 35, best="r2c1"), _seg("c3", 55, 25, 95, 35, best="r2c2"),
    ]
    return Document(source_path="x", sha256="x", pages=[page])


def test_table_fill_stage_populates_grid():
    doc = _table_doc()
    TableFill().run(doc, config_mod.Config())
    html = doc.pages[0].regions[0].table_html
    for cell in ("r1c1", "r1c2", "r2c1", "r2c2"):
        assert f">{cell}</td>" in html
    assert 'data-confidence="clean"' in html   # cleanly-contained cells flagged clean


def test_render_emits_table_and_suppresses_its_loose_lines():
    doc = _table_doc()
    TableFill().run(doc, config_mod.Config())
    # add a non-table caption line outside the table region
    doc.pages[0].regions.append(_reg_caption := Region(box=_box(0, 60, 100, 80)))
    doc.pages[0].regions[-1].reading_order = 1
    doc.pages[0].segments.append(_seg("cap", 5, 65, 95, 75, best="Table caption"))

    md = _page_markdown(doc.pages[0])
    assert "<table>" in md and "r1c1" in md          # table rendered with content
    assert "Table caption" in md                      # non-table text kept
    # the table's cell segments are NOT also dumped as loose lines
    assert md.count("r1c1") == 1


# ---- find_tables (born-digital) grid build + TableFill skip ----------------

def test_grid_to_table_html_clean_empty_and_escape():
    from fusion_ocr.compose import grid_to_table_html
    b = _box(0, 0, 10, 10)
    rows = [[("Name", b), ("", b)],            # a blank cell -> empty
            [("A & B", b), ("3", None)]]       # & escaped; None box -> no cell box
    html, cells = grid_to_table_html(rows)
    assert html.count("<tr>") == 2
    assert '<td data-confidence="clean">Name</td>' in html
    assert '<td data-confidence="empty"></td>' in html
    assert ">A &amp; B</td>" in html                  # escaped
    assert len(cells) == 3                            # only the 3 non-None boxes


def test_table_fill_skips_find_tables_grid():
    # a find_tables grid is already exact -> TableFill must not refill it (no doubled
    # attribute, no segment text injected into its empty cells)
    page = Page(index=0, width=200, height=100)
    region = Region(box=_box(0, 0, 100, 40), kind="table", table_engine="find_tables")
    region.table_html = ('<table><tbody><tr><td data-confidence="clean">A</td>'
                         '<td data-confidence="empty"></td></tr></tbody></table>')
    region.cells = [_box(0, 0, 50, 40), _box(50, 0, 100, 40)]
    page.regions = [region]
    page.segments = [_seg("s", 55, 5, 95, 35, best="X")]   # sits in the empty cell
    doc = Document(source_path="x", sha256="x", pages=[page])

    TableFill().run(doc, config_mod.Config())
    html = doc.pages[0].regions[0].table_html
    assert 'data-confidence="empty" data-confidence=' not in html   # not refilled
    assert ">X</td>" not in html                                    # segment not injected
    assert html.count("data-confidence") == 2                        # exactly one per cell
