"""Stage 5 — VLM read (the semantics track).

For each OCR-bound page we render the image and ask the VLM (via the swappable
OpenAI-compatible client) for a verbatim transcription, stored on the page as
`page.vlm_reading` — the clean reading view. Aligning that reading onto the
deterministic boxes is the fusion stage's job (it has the geometry to do it well).

This is where the value over tesseract appears — handwriting, degraded scans, and
non-Latin script the deterministic recogniser can't read. Geometry still comes from
the deterministic side; the VLM only supplies the reading.

The client is injectable for testing. Pages without `needs_ocr` are skipped, so
born-digital docs never hit the VLM.
"""

from __future__ import annotations

from ..config import Config
from ..models import Document
from ..vlm.client import get_client
from ..vlm.prompts import TRANSCRIBE

_DPI = 150


class VlmRead:
    name = "vlm_read"

    def __init__(self, dpi: int = _DPI, client=None) -> None:
        self.dpi = dpi
        self._client = client

    def run(self, doc: Document, cfg: Config) -> Document:
        targets = [p for p in doc.pages if p.needs_ocr]
        if not targets:
            return doc
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return doc

        client = self._client or get_client(cfg)
        with fitz.open(doc.source_path) as pdf:
            for page in targets:
                if page.index >= pdf.page_count:
                    continue
                png = pdf[page.index].get_pixmap(dpi=self.dpi).tobytes("png")
                try:
                    page.vlm_reading = client.read(png, TRANSCRIBE) or ""
                except Exception:
                    page.vlm_reading = ""  # degrade: fusion falls back to det_text
        return doc
