"""Stage 2 — Layout / structure.

REAL IMPL (extra: ocr): run PP-StructureV3 over each image-only page to get layout
regions (paragraph / table / figure / header), table cell structure, and reading
order. These region boxes are deterministic geometry — they anchor where the VLM's
linear transcription gets aligned back to in fusion.

WALKING SKELETON: passthrough.
"""

from __future__ import annotations

from ..config import Config
from ..models import Document


class Layout:
    name = "layout"

    def run(self, doc: Document, cfg: Config) -> Document:
        # TODO: PP-StructureV3 -> Region[] per page.
        return doc
