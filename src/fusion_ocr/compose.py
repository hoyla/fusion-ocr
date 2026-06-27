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


def grid_to_table_html(rows: list[list[tuple[str, Box | None]]]):
    """Build a filled HTML grid + aligned cell boxes from a row-major list of
    ``(text, box|None)`` cells — the find_tables (born-digital) path.

    The text is exact from the PDF's own layer and already correctly celled, so every
    non-empty cell is ``data-confidence="clean"`` (``empty`` where blank): there is no
    segment-to-cell straddling to flag, unlike the vision+fill path. Returns
    ``(table_html, [Box])`` ready for render, _conf_counts, and the segment index."""
    out = ["<table><tbody>"]
    cells: list[Box] = []
    for row in rows:
        out.append("<tr>")
        for text, box in row:
            t = (text or "").strip()
            out.append(f'<td data-confidence="{"clean" if t else "empty"}">'
                       f'{_escape(t)}</td>')
            if box is not None:
                cells.append(box)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out), cells


def _to_display(x: float, y: float, base_w: float, base_h: float, rot: int):
    """Map a base-space (derotated) point to displayed space, matching PyMuPDF's
    page.rotation_matrix. Segment boxes are stored derotated; ordering them by the
    stored y/x sorts by the unrotated layout, which is wrong on a rotated page. We
    sort by what the eye sees instead. Empirically (probe of page.rotation_matrix):
      90 -> (H-y, x)   180 -> (W-x, H-y)   270 -> (y, W-x),   where W,H are base dims."""
    if rot == 90:
        return (base_h - y, x)
    if rot == 180:
        return (base_w - x, base_h - y)
    if rot == 270:
        return (y, base_w - x)
    return (x, y)


def reading_key(seg: Segment, regions: list[Region], rotation: int = 0,
                disp_w: float = 0.0, disp_h: float = 0.0):
    """Sort key for reading order: the containing region's order, then top-to-bottom,
    then left-to-right. Segments outside every region sort after, by position.

    On a rotated page, region order is already correct (Layout orders in displayed
    space), but a region's lines are stored derotated — so within-region order must be
    taken in displayed space too. disp_w/disp_h are the displayed page dims (page.rect);
    base dims are those swapped back for 90/270. rotation==0 keeps the exact original
    key (top-left corner), so unrotated ordering is unchanged."""
    order = 1_000_000
    for r in regions:
        if _contains_centre(r.box, seg.box):
            order = r.reading_order
            break
    x0, y0, x1, y1 = seg.box.bbox
    if rotation in (90, 180, 270):
        base_w, base_h = (disp_h, disp_w) if rotation in (90, 270) else (disp_w, disp_h)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = _to_display(cx, cy, base_w, base_h, rotation)
        return (order, round(dy / 5), dx)
    return (order, round(y0 / 5), x0)
