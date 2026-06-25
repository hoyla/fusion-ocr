"""Stage 6 — Fusion + alignment.

Full fusion (sequence-aligning VLM text onto deterministic boxes) lands with the VLM
stage. This interim version does the part already needed by the real corpus: resolve
SPATIAL OVERLAP between exact text-layer segments and OCR segments on the same page —

  * a clean text-layer segment (exact glyphs) beats an overlapping OCR box;
  * an OCR box beats an overlapping CONTAMINATED text-layer segment (the Thai
    private-use-area case) — OCR is the repair;

then fill best_text for whatever survives. `det_text` / `vlm_text` are always kept
beside best_text for provenance.
"""

from __future__ import annotations

from ..config import Config
from ..models import Box, Document, Segment

_IOU_OVERLAP = 0.5


def _iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlaps(seg: Segment, others: list[Segment]) -> bool:
    return any(_iou(seg.box, o.box) >= _IOU_OVERLAP for o in others)


class Fusion:
    name = "fusion"

    def run(self, doc: Document, cfg: Config) -> Document:
        for page in doc.pages:
            clean_tl = [s for s in page.segments
                        if s.source == "textlayer" and s.best_text]
            paddle = [s for s in page.segments if s.source == "paddle"]

            kept: list[Segment] = []
            for s in page.segments:
                if s.source == "paddle" and _overlaps(s, clean_tl):
                    continue  # exact text layer already covers this box
                if (s.source == "textlayer" and not s.best_text
                        and _overlaps(s, paddle)):
                    continue  # contaminated layer -> OCR is the repair
                kept.append(s)
            page.segments = kept

            for seg in page.segments:
                if not seg.best_text:
                    seg.best_text = seg.vlm_text or seg.det_text or ""
                    if seg.source == "paddle" and seg.vlm_text:
                        seg.source = "fused"
        return doc
