"""Layout reading-order assembly. The order itself comes from PP-DocLayoutV2's learned
head (integration-tested via the eval, not unit-tested); the only local logic is how we
place regions the model leaves unordered (furniture), which _rank covers."""

from __future__ import annotations

from fusion_ocr.stages.layout import _rank


def test_in_flow_regions_sort_by_model_order():
    # model order is authoritative for in-flow regions, regardless of y position
    items = [(3, 100), (1, 700), (2, 400)]   # (model order, cy)
    ranked = sorted(items, key=lambda it: _rank(it[0], it[1], 800))
    assert [order for order, _ in ranked] == [1, 2, 3]


def test_unordered_furniture_placed_by_position():
    page_h = 800.0
    top = _rank(None, 20, page_h)        # running header
    bottom = _rank(None, 790, page_h)    # page number / footer
    inflow = _rank(5, 400, page_h)
    # top furniture before the flow, bottom furniture after it
    assert top < inflow < bottom


def test_degrades_to_top_to_bottom_without_model_order():
    # if every region is unordered (an older model emits no `order`), fall back to y-order
    page_h = 1000.0
    ys = [600, 100, 850, 300]
    ranked = sorted(ys, key=lambda y: _rank(None, y, page_h))
    assert ranked == [100, 300, 600, 850]
