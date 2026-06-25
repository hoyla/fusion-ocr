"""Stage 6 — Fusion + alignment.

REAL IMPL: for each deterministic box, find the corresponding VLM reading (sequence
alignment over reading-order text, fuzzy-matched). Decide `best_text` per segment:
agree -> high confidence; disagree -> prefer the VLM glyph reading BUT only where
the deterministic layer found ink (the anti-hallucination gate); VLM text with no
underlying box -> dropped/flagged, never silently overlaid. Both `det_text` and
`vlm_text` are retained beside `best_text` for provenance.

WALKING SKELETON: best_text = det_text or textlayer text, source carried through.
"""

from __future__ import annotations

from ..config import Config
from ..models import Document


class Fusion:
    name = "fusion"

    def run(self, doc: Document, cfg: Config) -> Document:
        for page in doc.pages:
            for seg in page.segments:
                if not seg.best_text:
                    seg.best_text = seg.vlm_text or seg.det_text or ""
                    if seg.source == "paddle" and seg.vlm_text:
                        seg.source = "fused"
        return doc
