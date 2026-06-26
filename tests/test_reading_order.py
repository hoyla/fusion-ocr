"""XY-cut reading order — the layouts that matter, deterministic. No deps."""

from __future__ import annotations

from fusion_ocr.compose import xy_cut_order
from fusion_ocr.models import Box, Region
from fusion_ocr.stages.layout import _order_regions


def _box(x0, y0, x1, y1):
    return Box(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def test_single_column_top_to_bottom():
    boxes = [_box(0, 0, 100, 20), _box(0, 30, 100, 50), _box(0, 60, 100, 80)]
    assert xy_cut_order(boxes) == [0, 1, 2]


def test_two_columns_left_then_right():
    # L1, L2 (left), R1, R2 (right); rows don't align so there's no full-width y-gap
    boxes = [_box(0, 0, 40, 20), _box(0, 30, 40, 50),     # left column
             _box(60, 10, 100, 30), _box(60, 40, 100, 60)]  # right column
    assert xy_cut_order(boxes) == [0, 1, 2, 3]            # whole left col, then right


def test_header_then_two_columns():
    boxes = [_box(0, 0, 100, 20),                          # full-width header
             _box(0, 30, 40, 80), _box(60, 30, 100, 80)]   # two columns
    assert xy_cut_order(boxes) == [0, 1, 2]                # header, left, right


def test_header_columns_footer():
    boxes = [_box(0, 0, 100, 20),                          # header
             _box(0, 30, 40, 80), _box(60, 30, 100, 80),   # columns
             _box(0, 90, 100, 110)]                        # footer (must come LAST)
    assert xy_cut_order(boxes) == [0, 1, 2, 3]


def test_table_is_row_major():
    # aligned 2x2 grid -> rows top-to-bottom, cells left-to-right
    boxes = [_box(0, 0, 40, 20), _box(60, 0, 100, 20),     # row 1
             _box(0, 30, 40, 50), _box(60, 30, 100, 50)]   # row 2
    assert xy_cut_order(boxes) == [0, 1, 2, 3]


def test_footer_not_read_first():
    # the Goldfinch bug: body at top, footer at bottom -> body before footer
    boxes = [_box(0, 0, 100, 70), _box(0, 90, 100, 100)]
    assert xy_cut_order(boxes) == [0, 1]


def test_order_regions_assigns_reading_order():
    # given OUT of reading order, _order_regions (XY-cut) re-sequences them
    regions = [Region(box=_box(0, 90, 100, 110)),          # footer (input first)
               Region(box=_box(0, 0, 100, 20)),            # header
               Region(box=_box(0, 30, 100, 80))]           # body
    out = _order_regions(regions)
    assert [round(r.box.bbox[1]) for r in out] == [0, 30, 90]   # header, body, footer
    assert [r.reading_order for r in out] == [0, 1, 2]
