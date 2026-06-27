"""Within-region line order on rotated pages — reading_key sorts in displayed space.

Region order itself is handled by Layout (XY-cut on displayed boxes, see
test_reading_order); this covers the residual: lines *inside* a region are stored
derotated, so their order must be taken in displayed space too."""

from __future__ import annotations

from fusion_ocr.compose import reading_key
from fusion_ocr.models import Box, Region, Segment


def _box(x0, y0, x1, y1):
    return Box(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _seg(name, x0, y0, x1, y1):
    return Segment(id=name, page=0, box=_box(x0, y0, x1, y1), best_text=name)


# Base page 200(w) x 100(h); three lines whose base-y order is the REVERSE of their
# displayed reading order once the page is rotated. A/B/C run left->right in base x
# (top->bottom in a 90deg view) but bottom->top in base y.
_REGION = [Region(box=_box(0, 0, 200, 100), reading_order=0)]
_SEGS = [_seg("A", 10, 70, 30, 90), _seg("B", 90, 30, 110, 50), _seg("C", 170, 0, 190, 20)]


def _order(rotation, disp_w, disp_h):
    return [s.best_text for s in sorted(
        _SEGS, key=lambda s: reading_key(s, _REGION, rotation, disp_w, disp_h))]


def test_unrotated_is_base_space_top_to_bottom():
    # rotation=0 keeps the original key exactly: sort by base y -> C, B, A
    assert _order(0, 200, 100) == ["C", "B", "A"]


def test_rotated_90_orders_in_displayed_space():
    # displayed dims swap to 100x200; displayed top->bottom follows base x -> A, B, C
    assert _order(90, 100, 200) == ["A", "B", "C"]


def test_rotated_270_is_opposite_of_90():
    assert _order(270, 100, 200) == ["C", "B", "A"]


def test_rotated_180_flips_top_to_bottom():
    # 180 keeps dims but inverts y: base-y order C,B,A becomes A,B,C
    assert _order(180, 200, 100) == ["A", "B", "C"]
