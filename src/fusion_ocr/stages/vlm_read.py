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

from ..config import AirgapError, Config
from ..models import Document
from ..routing import resolve
from ..vlm.openai_compat import OpenAICompatVLM
from ..vlm.prompts import select_prompt

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
                # Cheap tier: if Apple Vision already read the page confidently, its
                # det_text IS the reading — skip the VLM entirely (fusion uses det_text).
                if _vision_confident(page, cfg.apple_vision_skip_vlm):
                    page.read_model = "apple_vision"
                    continue
                route = resolve(page.script or "latin", cfg)
                model = route.vlm_model or cfg.vlm.model
                base_url = route.vlm_base_url or cfg.vlm.base_url
                png = pdf[page.index].get_pixmap(dpi=self.dpi).tobytes("png")
                det_chars = sum(len(s.det_text or "") for s in page.segments
                                if s.source == "paddle")

                reading = self._read(png, model, base_url, cfg)

                # Confidence-gated escalation: if the primary read looks like a refusal
                # or the page's deterministic confidence is low, re-read with a stronger
                # model. Keep it only if it's actually better (not a refusal too).
                esc = cfg.vlm.escalation_model
                if esc and esc != model and (
                        _looks_like_refusal(reading, det_chars)
                        or _low_confidence(page, cfg.vlm.escalate_below)):
                    esc_reading = self._read(
                        png, esc, cfg.vlm.escalation_base_url or base_url, cfg)
                    if not _looks_like_refusal(esc_reading, det_chars):
                        reading, model = esc_reading, esc

                if _looks_like_refusal(reading, det_chars):
                    page.vlm_reading = ""   # fusion falls back to routed det_text
                    page.read_model = ""
                else:
                    page.vlm_reading = reading
                    page.read_model = model
        return doc

    def _read(self, png, model, base_url, cfg) -> str:
        client = self._client or self._client_for(base_url, model, cfg)
        try:
            return client.read(png, select_prompt(model)) or ""
        except AirgapError:
            raise  # misconfigured sensitive tier (remote endpoint): fail loud, not det_text
        except Exception:
            return ""  # transient/other: degrade, fusion falls back to det_text


_REFUSAL_MARKERS = (
    "[image content", "[image]", "i cannot", "i can't", "i'm unable", "i am unable",
    "unable to read", "unable to process", "as an ai", "i'm sorry", "i am sorry",
)


def _vision_confident(page, threshold: float) -> bool:
    """True if Apple Vision read this page at high mean confidence — its text is good
    enough to skip the VLM (the cheap printed-text tier)."""
    vis = [s.det_conf for s in page.segments
           if s.source == "vision" and s.det_conf is not None]
    return bool(vis) and (sum(vis) / len(vis) >= threshold)


def _low_confidence(page, threshold: float) -> bool:
    """True if the page's mean PaddleOCR confidence is below the escalation threshold
    (a degraded/hard page worth a stronger reader). 0 disables."""
    if threshold <= 0:
        return False
    confs = [s.det_conf for s in page.segments
             if s.source == "paddle" and s.det_conf is not None]
    return bool(confs) and (sum(confs) / len(confs) < threshold)


def _looks_like_refusal(reading: str, det_chars: int) -> bool:
    """True if the VLM didn't really read the page — empty, a refusal/placeholder, or
    far shorter than what the deterministic engine found (so det_text is better)."""
    r = reading.strip().lower()
    if not r:
        return True
    if len(r) < 200 and any(m in r for m in _REFUSAL_MARKERS):
        return True
    if det_chars >= 80 and len(r) < 0.25 * det_chars:
        return True
    return False
