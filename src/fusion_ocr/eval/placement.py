"""Box-placement accuracy — evidence-plan stream C (P1: does a searched word highlight the
*right box*?).

The searchability eval (`labels.py::searchable`) only checks a word is findable *somewhere* in
the output PDF. This measures the stronger, product-defining claim: a word recovered for a given
line lands on **that line's box**, so a reporter clicking a claim sees it highlighted in the
right place. It scores the gated product (`page.segments` -> `segment_index.json` / overlay),
not the ungated reading.

METHOD. The 3rd-party gold sets carry per-line (box, text). Each of our segments is assigned to
the GT line box it overlaps most (IoU >= `iou_min`); a GT word is **well-placed** if it appears
in the text of the segment(s) assigned to its own line. Then:

  placement_recall  = well-placed GT words / all GT words        (the headline P1 number)
  plain_recall      = GT words present ANYWHERE on the page      (recognition, ignoring place)
  placement_gap     = plain_recall - placement_recall           (pure mis-placement)

The gap is what P1 adds over recognition: words we read but pinned to the wrong box. On a
deterministic run this tests the *detector's* geometry; on a fused (VLM) run it tests whether
fusion put the VLM's words on the right boxes — the click-a-claim test proper.

COORDINATES. GT boxes are in original-image pixels; our segment boxes are in PDF points. The
image-ingest embeds the page at 96 DPI, so points = pixels * (page.width / image_width) — a
single uniform scale, verified on FUNSD (GT `TO:` [55,108,83,126]*0.75 == segment [42,82,64,95]).
We scale GT -> points with the page/image ratio; no other transform.
"""

from __future__ import annotations

from .metrics import word_tokens

_IOU_MIN = 0.3   # a segment counts as "on" a GT line at or above this overlap


def gt_bbox(entry: dict, source: str) -> list[float] | None:
    """(x0, y0, x1, y1) for a GT line, from either FUNSD's `box` or SROIE's `points` polygon."""
    if source == "funsd":
        b = entry.get("box")
        return list(b) if b and len(b) == 4 else None
    pts = entry.get("points")
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def gt_lines(ann: dict, source: str) -> list[tuple[list[float], str]]:
    """(bbox_px, text) per annotated line, for the sources the eval scores."""
    items = ann.get("form", []) if source == "funsd" else ann.get("ocr_boxes", [])
    out = []
    for it in items:
        text = (it.get("text") or "").strip()
        box = gt_bbox(it, source)
        if text and box:
            out.append((box, text))
    return out


def _iou(a: list[float], b: tuple[float, ...]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def placement_counts(page, lines: list[tuple[list[float], str]], img_w: float, img_h: float,
                     *, caseless: bool = False, iou_min: float = _IOU_MIN) -> dict:
    """Well-placed / plain-recovered / total GT-word counts for one page against its GT lines.

    A GT word is *well-placed* if it appears in the union text of the segments whose best-overlap
    GT line is the one containing that word. *plain* ignores placement (word anywhere on page)."""
    segs = [s for s in page.segments if (s.best_text or "").strip() and not getattr(s, "superseded", False)]
    if not segs or not lines or not img_w or not img_h:
        return {"placed": 0, "plain": 0, "total": 0, "segs": len(segs)}
    sx, sy = page.width / img_w, page.height / img_h
    gt_pt = [([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], t) for b, t in lines]

    fold = str.casefold if caseless else (lambda s: s)
    # assign each segment to its best-overlap GT line (index), if any clears the threshold
    seg_words_by_line: dict[int, set[str]] = {}
    for s in segs:
        best_i, best_iou = -1, iou_min
        for i, (bb, _) in enumerate(gt_pt):
            v = _iou(bb, s.box.bbox)
            if v >= best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            seg_words_by_line.setdefault(best_i, set()).update(
                fold(w) for w in word_tokens(s.best_text or ""))

    all_words = {fold(w) for s in segs for w in word_tokens(s.best_text or "")}
    placed = plain = total = 0
    for i, (_, text) in enumerate(gt_pt):
        gw = [fold(w) for w in word_tokens(text)]
        total += len(gw)
        here = seg_words_by_line.get(i, set())
        placed += sum(1 for w in gw if w in here)
        plain += sum(1 for w in gw if w in all_words)
    return {"placed": placed, "plain": plain, "total": total, "segs": len(segs)}


def summarize(rows: list[dict]) -> dict:
    """Micro-average placement over per-page count dicts."""
    tot = sum(r["total"] for r in rows) or 1
    placed = sum(r["placed"] for r in rows)
    plain = sum(r["plain"] for r in rows)
    return {"pages": len(rows), "gt_words": sum(r["total"] for r in rows),
            "placement_recall": placed / tot, "plain_recall": plain / tot,
            "placement_gap": (plain - placed) / tot}
