"""Stage 5 — VLM read (the semantics track).

For each OCR-bound page we render the image and ask the VLM (via the swappable
OpenAI-compatible client) for a verbatim transcription. The raw reading is kept on
the page (`page.vlm_reading`) as the clean reading view, and its lines are aligned
onto the deterministic PaddleOCR boxes so the overlay carries VLM-quality text on
real geometry.

This is where the value over tesseract appears — handwriting, degraded scans, and
non-Latin script the deterministic recogniser can't read. Geometry still comes from
the deterministic side; the VLM only supplies the reading.

The client is injectable for testing. Pages without `needs_ocr` are skipped, so
born-digital docs never hit the VLM.
"""

from __future__ import annotations

from ..config import Config
from ..models import Document, Page
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
                _align_to_boxes(page)
        return doc


def _align_to_boxes(page: Page) -> None:
    """Distribute the VLM reading's lines onto the page's PaddleOCR boxes in reading
    order. First cut: 1:1 when counts match, else proportional. Full sequence
    alignment (fuzzy-matched against det_text) is the planned refinement."""
    lines = [ln.strip() for ln in page.vlm_reading.splitlines() if ln.strip()]
    boxes = sorted(
        (s for s in page.segments if s.source == "paddle"),
        key=lambda s: (round(s.box.bbox[1] / 5), s.box.bbox[0]),  # top-to-bottom, l-to-r
    )
    if not lines or not boxes:
        return
    if len(lines) == len(boxes):
        for seg, line in zip(boxes, lines):
            seg.vlm_text = line
        return
    ratio = len(lines) / len(boxes)
    for i, seg in enumerate(boxes):
        seg.vlm_text = lines[min(int(i * ratio), len(lines) - 1)]
