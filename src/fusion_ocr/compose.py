"""Composition geometry for mixed-content pages.

Pure, side-effect-light helpers used by fusion to combine a page's MACHINE-READABLE
text layer with the OCR of its IMAGE areas. The unit of decision is the layout region:
a region covered by clean text is `machine_readable` (use the text layer verbatim),
the rest is `ocr` (read it). Kept deliberately free of pipeline objects' behaviour so
each rule is testable in isolation.
"""

from __future__ import annotations

import re

from .models import Box, Region, Segment

_MR_COVERAGE = 0.5  # a region is machine-readable if clean text covers >= this fraction


def _area(b: Box) -> float:
    x0, y0, x1, y1 = b.bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _intersection_area(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return iw * ih


def _contains_centre(region: Box, seg: Box) -> bool:
    rx0, ry0, rx1, ry1 = region.bbox
    sx0, sy0, sx1, sy1 = seg.bbox
    cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def region_text_coverage(region: Region, clean_boxes: list[Box]) -> float:
    """Fraction of the region's area covered by clean text-layer boxes (approximate:
    sums intersections, which is exact when the text boxes don't overlap — the normal
    case for lines — and is capped at 1.0)."""
    ra = _area(region.box)
    if ra <= 0:
        return 0.0
    inter = sum(_intersection_area(region.box, b) for b in clean_boxes)
    return min(1.0, inter / ra)


def classify_regions(regions: list[Region], clean_segments: list[Segment],
                     threshold: float = _MR_COVERAGE) -> None:
    """Tag each region 'textlayer' (machine-readable) or 'ocr' in place."""
    clean_boxes = [s.box for s in clean_segments]
    for r in regions:
        r.source = ("textlayer"
                    if region_text_coverage(r, clean_boxes) >= threshold else "ocr")


def in_machine_readable_region(seg: Segment, regions: list[Region]) -> bool:
    """True if the segment's centre falls in a region tagged machine-readable."""
    return any(r.source == "textlayer" and _contains_centre(r.box, seg.box)
               for r in regions)


def cell_text(cell: Box, segments: list[Segment]) -> str:
    """Join the best_text of segments whose centre falls in the cell, in reading order
    (top-to-bottom, then left-to-right). This is how the table's cell *content* is
    recovered from the page's OCR/text-layer segments."""
    inside = [s for s in segments if s.best_text and _contains_centre(cell, s.box)]
    inside.sort(key=lambda s: (round(s.box.bbox[1] / 5), s.box.bbox[0]))
    return " ".join(s.best_text for s in inside).strip()


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_CONTAINED = 0.7   # a segment >= this fraction inside the cell is "in" it
_TOUCHES = 0.1     # a segment overlapping the cell by >= this fraction touches it


def cell_confidence(cell: Box, segments: list[Segment]) -> str:
    """How trustworthy is this cell's content?

    - ``clean``: every segment touching the cell is (mostly) inside it — its text is
      reliably this cell's.
    - ``spanning``: some segment straddles the cell boundary (a label+value line across
      two cells) — the text may belong partly to a neighbour; do NOT trust it as the
      exact cell value.
    - ``empty``: nothing touches it.

    This is the calibration: we fill what we can and FLAG where we can't justify
    precision, rather than manufacturing a confident-but-wrong cell value.
    """
    touching = [s for s in segments if s.best_text
                and _intersection_area(cell, s.box) >= _TOUCHES * max(_area(s.box), 1.0)]
    if not touching:
        return "empty"
    for s in touching:
        if _intersection_area(cell, s.box) / max(_area(s.box), 1.0) < _CONTAINED:
            return "spanning"
    return "clean"


_EMPTY_CELL = re.compile(r"<td([^>]*)></td>")


def populate_table_html(table_html: str, cells: list[Box],
                        segments: list[Segment]) -> str:
    """Fill each empty `<td>` with its cell's text AND a `data-confidence` attribute
    (clean / spanning / empty). The Nth empty cell corresponds to the Nth cell box
    (TableStructureRecognition's contract). Surfacing confidence lets a human or a
    machine gate on it instead of trusting every cell equally."""
    texts = [cell_text(c, segments) for c in cells]
    confs = [cell_confidence(c, segments) for c in cells]
    counter = {"i": 0}

    def _fill(m):
        i = counter["i"]
        counter["i"] += 1
        if i >= len(cells):
            return m.group(0)  # more <td>s than cell boxes -> leave extra empty
        return (f'<td{m.group(1)} data-confidence="{confs[i]}">'
                f'{_escape(texts[i])}</td>')

    return _EMPTY_CELL.sub(_fill, table_html)


def xy_cut_order(boxes: list[Box]) -> list[int]:
    """Reading order via recursive XY-cut — the same approach PP-StructureV3 uses,
    applied to our layout regions. Returns indices into ``boxes`` in reading order.

    At each step it peels off a full-width horizontal band (top-to-bottom: header, then
    the column block, then footer); where no horizontal gap exists it cuts on a
    full-height vertical band (columns, left-to-right) and recurses. This gives the
    right answer for single-column, multi-column, header+columns, and table-like
    (row-major) layouts — and it's deterministic and explainable, not a black box.
    """
    order: list[int] = []
    _xy_cut(boxes, list(range(len(boxes))), order)
    return order


def _xy_cut(boxes: list[Box], idx: list[int], order: list[int]) -> None:
    if len(idx) <= 1:
        order.extend(idx)
        return
    groups = _split_on_gap(boxes, idx, axis=1)        # horizontal band (rows) first
    if groups is None:
        groups = _split_on_gap(boxes, idx, axis=0)    # else vertical band (columns)
    if groups is None:                                 # no clean cut -> y then x
        order.extend(sorted(idx, key=lambda i: (boxes[i].bbox[1], boxes[i].bbox[0])))
        return
    for g in groups:
        _xy_cut(boxes, g, order)


def _split_on_gap(boxes: list[Box], idx: list[int], axis: int):
    """Split idx into [before, after] across the widest empty band along `axis`
    (axis=1 -> a horizontal gap in y; axis=0 -> a vertical gap in x), or None if no box-
    free band spans the group."""
    lo, hi = axis, axis + 2  # bbox is (x0,y0,x1,y1)
    spans = sorted(((boxes[i].bbox[lo], boxes[i].bbox[hi], i) for i in idx),
                   key=lambda t: t[0])
    max_hi = spans[0][1]
    best_gap, best_k = 0.0, None
    for k in range(1, len(spans)):
        gap = spans[k][0] - max_hi
        if gap > best_gap:
            best_gap, best_k = gap, k
        max_hi = max(max_hi, spans[k][1])
    if best_k is None or best_gap <= 0:
        return None
    return [[t[2] for t in spans[:best_k]], [t[2] for t in spans[best_k:]]]


def reading_key(seg: Segment, regions: list[Region]):
    """Sort key for reading order: the containing region's order, then top-to-bottom,
    then left-to-right. Segments outside every region sort after, by position."""
    order = 1_000_000
    for r in regions:
        if _contains_centre(r.box, seg.box):
            order = r.reading_order
            break
    x0, y0, _, _ = seg.box.bbox
    return (order, round(y0 / 5), x0)
