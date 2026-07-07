"""Box-placement metric (evidence-plan stream C) — pure geometry/text logic, no models beyond
the dataclasses, no images."""

from __future__ import annotations

from fusion_ocr.eval import placement
from fusion_ocr.models import Box, Page, Segment


def _seg(x0, y0, x1, y1, text):
    return Segment(id="s", page=0, box=Box(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
                   best_text=text)


def test_gt_bbox_funsd_and_sroie_polygon():
    assert placement.gt_bbox({"box": [1, 2, 3, 4]}, "funsd") == [1, 2, 3, 4]
    # SROIE polygon -> enclosing bbox
    assert placement.gt_bbox({"points": [[10, 5], [30, 5], [30, 9], [10, 9]]}, "sroie") == [10, 5, 30, 9]


def test_wellplaced_word_must_be_in_the_right_line_box():
    # image == points (scale 1). Two GT lines far apart; two segments each on their own line.
    page = Page(index=0, width=100, height=100)
    page.segments = [_seg(0, 0, 20, 10, "hello"), _seg(0, 50, 20, 60, "world")]
    lines = [([0, 0, 20, 10], "hello"), ([0, 50, 20, 60], "world")]
    c = placement.placement_counts(page, lines, 100, 100)
    assert (c["placed"], c["plain"], c["total"]) == (2, 2, 2)   # both words on their own line


def test_recognised_but_misplaced_word_counts_plain_not_placed():
    # "world" is recognised, but its segment sits on line 1's box, not line 2's -> plain, not placed.
    page = Page(index=0, width=100, height=100)
    page.segments = [_seg(0, 0, 20, 10, "hello world")]     # both words in the top-line box
    lines = [([0, 0, 20, 10], "hello"), ([0, 50, 20, 60], "world")]
    c = placement.placement_counts(page, lines, 100, 100)
    assert c["total"] == 2
    assert c["plain"] == 2      # both recognised somewhere
    assert c["placed"] == 1     # only "hello" is in its own line's segment; "world" is misplaced
    s = placement.summarize([c])
    assert s["placement_recall"] == 0.5 and s["plain_recall"] == 1.0 and s["placement_gap"] == 0.5


def test_pixel_to_point_scale_is_applied():
    # GT in pixels (image 200 wide), page 100 wide -> 0.5 scale; the segment is in points.
    page = Page(index=0, width=100, height=100)
    page.segments = [_seg(0, 0, 10, 5, "x")]                 # points
    lines = [([0, 0, 20, 10], "x")]                          # pixels; *0.5 -> [0,0,10,5] == segment
    assert placement.placement_counts(page, lines, 200, 200)["placed"] == 1


def test_caseless_matches_uppercase_gt():
    page = Page(index=0, width=100, height=100)
    page.segments = [_seg(0, 0, 20, 10, "Total")]
    lines = [([0, 0, 20, 10], "TOTAL")]                      # SROIE-style uppercase GT
    assert placement.placement_counts(page, lines, 100, 100, caseless=False)["placed"] == 0
    assert placement.placement_counts(page, lines, 100, 100, caseless=True)["placed"] == 1
