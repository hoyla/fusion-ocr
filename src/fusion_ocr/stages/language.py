"""Stage 3 — Language / script identification.

REAL IMPL: identify the script(s) present (Latin / Arabic / CJK / Cyrillic ...) from
a quick OCR sample or a VLM probe, to select PaddleOCR language models and the set
of translation targets. Populates Document.languages.

WALKING SKELETON: passthrough (languages left empty -> downstream treats as unknown).
"""

from __future__ import annotations

from ..config import Config
from ..models import Document


class Language:
    name = "language"

    def run(self, doc: Document, cfg: Config) -> Document:
        # TODO: script detection -> doc.languages
        return doc
