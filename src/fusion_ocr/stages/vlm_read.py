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
from ..routing import resolve
from ..vlm.openai_compat import OpenAICompatVLM
from ..vlm.prompts import TRANSCRIBE

_DPI = 150


class VlmRead:
    name = "vlm_read"

    def __init__(self, dpi: int = _DPI, client=None) -> None:
        self.dpi = dpi
        self._client = client  # injected -> used for every page (tests)
        self._clients: dict[tuple, OpenAICompatVLM] = {}

    def _client_for(self, base_url: str, model: str, cfg: Config):
        key = (base_url, model)
        if key not in self._clients:
            self._clients[key] = OpenAICompatVLM(
                base_url=base_url, model=model, api_key=cfg.vlm.api_key
            )
        return self._clients[key]

    def run(self, doc: Document, cfg: Config) -> Document:
        targets = [p for p in doc.pages if p.needs_ocr]
        if not targets:
            return doc
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return doc

        with fitz.open(doc.source_path) as pdf:
            for page in targets:
                if page.index >= pdf.page_count:
                    continue
                route = resolve(page.script or "latin", cfg)
                model = route.vlm_model or cfg.vlm.model
                base_url = route.vlm_base_url or cfg.vlm.base_url
                client = self._client or self._client_for(base_url, model, cfg)
                page.read_model = model
                png = pdf[page.index].get_pixmap(dpi=self.dpi).tobytes("png")
                try:
                    page.vlm_reading = client.read(png, TRANSCRIBE) or ""
                except Exception:
                    page.vlm_reading = ""  # degrade: fusion falls back to det_text
        return doc
