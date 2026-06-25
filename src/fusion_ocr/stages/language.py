"""Stage 3 — Language / script identification (drives routing).

Classifies each page's dominant script from the text we already have (the embedded
text layer — even a partial Thai header/footer counts), and stores it on
`page.script`. The router (see fusion_ocr.routing) maps that to a PaddleOCR recogniser
+ VLM reader downstream.

Pure image-only pages with no text layer can't be classified here yet; they keep
script="" and take the default (Latin) route. Image-only script detection — a fast
langid probe — is a documented follow-up (Docs/routing.md).
"""

from __future__ import annotations

from ..config import Config
from ..models import Document
from ..routing import detect_script


class Language:
    name = "language"

    def run(self, doc: Document, cfg: Config) -> Document:
        langs: set[str] = set()
        for page in doc.pages:
            text = " ".join(
                s.det_text or "" for s in page.segments if s.source == "textlayer"
            )
            if text.strip():
                page.script = detect_script(text)
                langs.add(page.script)
        if langs:
            doc.languages = sorted(langs)
        return doc
