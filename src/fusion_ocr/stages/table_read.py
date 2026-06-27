"""Stage — table-class routing (focused VLM table read).

Tables are structure, and structure is exactly what line-OCR and Apple Vision handle
poorly (good at characters, weak at grids). So every detected table region on a
scanned page is routed to a focused VLM read: crop the region, ask the reader for the
table with the table prompt, store the clean markdown/HTML on `region.table_vlm`. This
runs whether or not the page got a page-level read — it's the table specialist in the
toolkit — and is routed like any other read (a Thai table goes to Typhoon).

Geometry is NOT taken from this: the deterministic PaddleOCR grid (`table_html` /
`cells`) still owns cell boxes and per-cell confidence for the overlay and structured
export. Content from the VLM, geometry from the grid — both kept (provenance).

Born-digital pages are deliberately skipped: there the exact text and (often) exact
ruling lines are already in the file, so re-reading a raster with a generative model
would only add error. That class is left to the text layer / deterministic grid.

The client is injectable for testing. Rotated pages are skipped (crop-on-rotated is a
refinement, matching the Table stage).
"""

from __future__ import annotations

from ..config import AirgapError, Config
from ..models import Document
from ..routing import resolve
from ..vlm.openai_compat import OpenAICompatVLM
from ..vlm.prompts import select_table_prompt

_DPI = 150


class TableRead:
    name = "table_read"

    def __init__(self, dpi: int = _DPI, client=None) -> None:
        self.dpi = dpi
        self._client = client  # injected -> used for every read (tests)
        self._clients: dict[tuple, OpenAICompatVLM] = {}

    def _client_for(self, base_url: str, model: str, cfg: Config):
        key = (base_url, model)
        if key not in self._clients:
            self._clients[key] = OpenAICompatVLM(
                base_url=base_url, model=model, api_key=cfg.vlm.api_key
            )
        return self._clients[key]

    def run(self, doc: Document, cfg: Config) -> Document:
        if not getattr(cfg, "table_vlm_read", True):
            return doc
        # Scanned (needs_ocr) pages only; born-digital tables stay with the text layer.
        targets = [p for p in doc.pages
                   if p.needs_ocr and not p.rotation
                   and any(r.kind == "table" for r in p.regions)]
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
                for region in page.regions:
                    if region.kind != "table" or region.table_vlm:
                        continue  # idempotent: don't re-read a region already done
                    clip = fitz.Rect(*region.box.bbox)
                    if clip.is_empty or clip.width < 8 or clip.height < 8:
                        continue
                    png = pdf[page.index].get_pixmap(dpi=self.dpi, clip=clip).tobytes("png")
                    text = self._read(png, model, base_url, cfg)
                    if text:
                        region.table_vlm = text
                        region.table_read_by = model
        return doc

    def _read(self, png, model, base_url, cfg) -> str:
        client = self._client or self._client_for(base_url, model, cfg)
        try:
            return (client.read(png, select_table_prompt(model)) or "").strip()
        except AirgapError:
            raise  # misconfigured sensitive tier: fail loud, not silent grid fallback
        except Exception:
            return ""  # degrade: render falls back to the deterministic grid
