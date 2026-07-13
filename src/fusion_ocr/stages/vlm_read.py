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

import logging

from .. import raster
from ..config import AirgapError, Config
from ..models import Document
from ..routing import resolve
from ..vlm.openai_compat import OpenAICompatVLM
from ..vlm.prompts import select_prompt

_DPI = 150
_log = logging.getLogger(__name__)


class ReaderError(Exception):
    """The VLM reader failed for a reason OTHER than a legitimate empty/refusal response —
    server down / wedged / timeout / bad model name. Kept distinct from a refusal so the
    pipeline can FAIL LOUD (log it + flag the page) instead of silently degrading to det_text;
    a run where every page raises this is a dead reader, not hard documents."""


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
                base_url=base_url, model=model, api_key=cfg.vlm.api_key,
                max_tokens=cfg.vlm.max_tokens, max_retries=cfg.vlm.max_retries,
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
                # Blank / no-ink short-circuit: if the deterministic engine detected NO text
                # boxes, the page is (near-)blank. Asking the VLM to read an empty image makes
                # it HALLUCINATE (measured on OCR-Quality: blank pages -> invented "$$1/√2$$"),
                # and there'd be no ink to anchor it anyway. Skip: correct output (empty) + a
                # saved VLM call. Keys on DETECTION not recognition, so handwriting — which the
                # detector boxes even when it can't read it — still reaches the VLM.
                if not page.segments:
                    page.read_model = ""
                    continue
                # Cheap tier: if Apple Vision already read the page confidently, its
                # det_text IS the reading — skip the VLM entirely (fusion uses det_text).
                if _vision_confident(page, cfg.apple_vision_skip_vlm):
                    page.read_model = "apple_vision"
                    continue
                route = resolve(page.script or "latin", cfg)
                model = route.vlm_model or cfg.vlm.model
                base_url = route.vlm_base_url or cfg.vlm.base_url
                img = raster.page_jpeg(pdf, page.index, self.dpi, quality=cfg.vlm.jpeg_quality)
                det_chars = sum(len(s.det_text or "") for s in page.segments
                                if s.source == "paddle")

                try:
                    reading = self._read(img, model, base_url, cfg)
                except ReaderError:
                    # Reader unavailable for this page: FLAG it (the warning was logged in
                    # _read) and fall back to routed det_text via fusion — visibly, not silently.
                    page.read_failed = True
                    page.vlm_reading = ""
                    page.read_model = ""
                    continue

                # Confidence-gated escalation: if the primary read looks like a refusal
                # or the page's deterministic confidence is low, re-read with a stronger
                # model. Keep it only if it's actually better (not a refusal too).
                esc = cfg.vlm.escalation_model
                if esc and esc != model and (
                        _looks_like_refusal(reading, det_chars)
                        or _low_confidence(page, cfg.vlm.escalate_below)):
                    try:
                        esc_reading = self._read(
                            img, esc, cfg.vlm.escalation_base_url or base_url, cfg)
                    except ReaderError:
                        esc_reading = ""   # escalation reader failed; keep the primary reading
                    if not _looks_like_refusal(esc_reading, det_chars):
                        reading, model = esc_reading, esc

                if _looks_like_refusal(reading, det_chars):
                    page.vlm_reading = ""   # fusion falls back to routed det_text
                    page.read_model = ""
                else:
                    page.vlm_reading = reading
                    page.read_model = model
        return doc

    def _read(self, img, model, base_url, cfg) -> str:
        client = self._client or self._client_for(base_url, model, cfg)
        try:
            return client.read(img, select_prompt(model)) or ""
        except AirgapError:
            raise  # misconfigured sensitive tier (remote endpoint): fail loud, not det_text
        except Exception as exc:
            # Fail LOUD: a reader failure used to return "" here, silently degrading the whole
            # corpus to det_text with no trace. Log it and raise a distinct error the caller
            # flags on the page (fusion still falls back to det_text — but visibly now).
            _log.warning("VLM read failed (model=%s, endpoint=%s): %s", model, base_url, exc)
            raise ReaderError(f"{type(exc).__name__}: {exc}") from exc


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
    """True if the VLM didn't really read the page — empty, a refusal/placeholder, far
    shorter than the deterministic engine found (so det_text is better), or collapsed into a
    repetition loop. In every case fusion falls back to det_text (and escalation may retry)."""
    r = reading.strip().lower()
    if not r:
        return True
    if len(r) < 200 and any(m in r for m in _REFUSAL_MARKERS):
        return True
    if det_chars >= 80 and len(r) < 0.25 * det_chars:
        return True
    if _is_degenerate_repetition(reading):
        return True
    return False


def _is_degenerate_repetition(reading: str) -> bool:
    """True if the read collapsed into a repetition loop — one token dominating the output, or
    a tiny vocabulary over a long output. Measured failure mode: the VLM emitting '[illegible]
    [illegible] …' to the token cap on a figure-heavy / sparse page. Only long outputs are
    checked, so ordinary prose (high vocabulary variety) is never flagged."""
    from collections import Counter

    words = reading.split()
    if len(words) < 40:
        return False
    _, top = Counter(words).most_common(1)[0]
    return top >= 0.40 * len(words) or len(set(words)) / len(words) < 0.10
