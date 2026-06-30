"""Stage 3 — Language / script identification (drives routing).

Two passes:
  1. From the embedded text layer (cheap, deterministic) — even a partial Thai
     header/footer classifies the page.
  2. For OCR-bound pages with NO text layer (pure scans), a one-word VLM probe of the
     image — so a scanned Thai/Cyrillic page still routes to the right recogniser +
     reader instead of defaulting to Latin. Falls back to Latin if no reader answers.

The probe uses the generalist reader (cfg.vlm); the chosen script is recorded on the
page, so the routing decision stays auditable. Client is injectable for tests.
"""

from __future__ import annotations

from .. import raster
from ..config import AirgapError, Config
from ..models import Document
from ..routing import detect_script

# 72 DPI: the probe only needs to NAME the dominant script, not read it — measured to return
# the same script as 120 DPI (Latin/CJK) while roughly halving the VLM prefill cost (the probe's
# cost is vision-token prefill, scaling with image size, not the one-word generation). Replacing
# the VLM probe with a cheaper detector (ANE Apple Vision / a script classifier) is the
# documented follow-up (routing.md), now that profiling shows the probe is ~14% of runtime.
_PROBE_DPI = 72
_SCRIPT_PROBE = (
    "What is the dominant script of the main printed text in this image? "
    "Answer with EXACTLY ONE word from: Latin, Thai, Cyrillic, Arabic, CJK, Devanagari."
)
_PROBE_MAP = {
    "latin": "latin", "thai": "thai", "cyrillic": "cyrillic", "arabic": "arabic",
    "cjk": "cjk", "chinese": "cjk", "japanese": "cjk", "korean": "cjk", "han": "cjk",
    "devanagari": "devanagari", "hindi": "devanagari",
}


class Language:
    name = "language"

    def __init__(self, client=None, dpi: int = _PROBE_DPI) -> None:
        self._client = client
        self.dpi = dpi

    def run(self, doc: Document, cfg: Config) -> Document:
        # 1. text-layer script detection
        for page in doc.pages:
            text = " ".join(
                s.det_text or "" for s in page.segments if s.source == "textlayer")
            if text.strip():
                page.script = detect_script(text)

        # 2. image-only probe for OCR-bound pages still without a script
        probe = [p for p in doc.pages if p.needs_ocr and not p.script]
        if probe:
            self._probe_pages(doc, cfg, probe)

        langs = sorted({p.script for p in doc.pages if p.script})
        if langs:
            doc.languages = langs
        return doc

    def _probe_pages(self, doc, cfg, pages) -> None:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return
        client = self._client or self._default_client(cfg)
        if client is None:
            return
        with fitz.open(doc.source_path) as pdf:
            for page in pages:
                if page.index >= pdf.page_count:
                    continue
                img = raster.page_jpeg(pdf, page.index, self.dpi, quality=cfg.vlm.jpeg_quality)
                page.script = _probe_script(client, img) or ""

    @staticmethod
    def _default_client(cfg):
        from ..vlm.openai_compat import OpenAICompatVLM
        return OpenAICompatVLM(base_url=cfg.vlm.base_url, model=cfg.vlm.model,
                               api_key=cfg.vlm.api_key, max_tokens=cfg.vlm.max_tokens,
                               max_retries=cfg.vlm.max_retries)


def _probe_script(client, img: bytes) -> str:
    try:
        ans = (client.read(img, _SCRIPT_PROBE) or "").strip().lower()
    except AirgapError:
        raise  # sealed tier pointed at a remote endpoint: fail loud, like vlm_read
    except Exception:
        return ""
    # scan every word so a verbose answer ("the script is Cyrillic") still resolves
    for w in ans.replace(".", " ").replace(",", " ").split():
        if w in _PROBE_MAP:
            return _PROBE_MAP[w]
    return ""
