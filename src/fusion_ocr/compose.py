"""Composition geometry for mixed-content pages.

Pure, side-effect-light helpers used by fusion to combine a page's MACHINE-READABLE
text layer with the OCR of its IMAGE areas. The unit of decision is the layout region:
a region covered by clean text is `machine_readable` (use the text layer verbatim),
the rest is `ocr` (read it). Kept deliberately free of pipeline objects' behaviour so
each rule is testable in isolation.
"""

from __future__ import annotations

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


def populate_table_html(table_html: str, cells: list[Box],
                        segments: list[Segment]) -> str:
    """Insert each cell's text before its `</td>` in the structure HTML. The Nth cell
    box corresponds to the Nth `<td>` (TableStructureRecognition's contract), so we
    walk the `</td>` boundaries in order."""
    texts = [cell_text(c, segments) for c in cells]
    parts = table_html.split("</td>")
    out: list[str] = []
    for i, part in enumerate(parts[:-1]):
        out.append(part)
        out.append(_escape(texts[i]) if i < len(texts) else "")
        out.append("</td>")
    out.append(parts[-1])
    return "".join(out)


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
