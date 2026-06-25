"""Stage 4 — Deterministic OCR (the geometry track).

REAL IMPL (extra: ocr): run PaddleOCR det+rec+cls over image-only regions. Each
detected line yields a Segment with `box` (quad polygon), `det_text`, and
`det_conf`. THESE BOXES ARE THE CANONICAL GEOMETRY for the whole pipeline — the
overlay and every highlight are positioned from them, never from the VLM.

WALKING SKELETON: passthrough (no segments produced without the OCR extra).
"""

from __future__ import annotations

from ..config import Config
from ..models import Document


class OcrDet:
    name = "ocr_det"

    def run(self, doc: Document, cfg: Config) -> Document:
        # TODO: PaddleOCR -> Segment(box, det_text, det_conf, source="paddle")
        return doc
