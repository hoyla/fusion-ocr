"""Stage 4 — Deterministic OCR (the geometry track).

Runs PaddleOCR over image-only pages and produces one Segment per detected text
line: a quad `box`, the raw `det_text`, and `det_conf`. THESE BOXES ARE THE
CANONICAL GEOMETRY for the whole pipeline — the overlay and every highlight are
positioned from them, never from the VLM.

Pages that already have a usable text layer are skipped (their geometry comes from
PyMuPDF in triage). If PaddleOCR isn't installed the stage is a clean passthrough,
so the walking skeleton still runs without the `ocr` extra.

Coordinate handling: pages are rasterised at `dpi`; PaddleOCR returns pixel
coordinates, which we map back to PDF points (1/72") by dividing by dpi/72 so the
overlay lands on the original page. Page rotation/origin offsets are assumed
standard for the MVP (refine if a rotated-scan case turns up).

Handles both PaddleOCR 2.x (`.ocr()` -> [[ [box,(text,score)], ... ]]) and 3.x
(`.predict()` -> results carrying rec_texts / rec_polys / rec_scores).
"""

from __future__ import annotations

import logging

from ..config import Config
from ..models import Box, Document, Segment

_DEFAULT_DPI = 200
_log = logging.getLogger(__name__)


class OcrDet:
    name = "ocr_det"

    def __init__(self, dpi: int = _DEFAULT_DPI, lang: str = "en") -> None:
        self.dpi = dpi
        self.lang = lang  # fallback when a page has no detected script
        # cache one engine per recogniser language: lang -> (engine, mode)
        self._engines: dict[str, tuple] = {}

    # -- engine lifecycle ---------------------------------------------------

    def _engine_for(self, lang: str):
        """Return (engine, mode) for a recogniser language, building + caching on
        first use. Falls back to English if PaddleOCR doesn't support the language."""
        if lang in self._engines:
            return self._engines[lang]
        from paddleocr import PaddleOCR  # raises ImportError if extra absent

        # CRITICAL for overlay accuracy: disable 3.x document-preprocessing
        # (orientation + UVDoc unwarping + textline orientation) — it warps the image
        # so polygons come back in unwarped space and no longer line up with the page.
        try:
            try:  # 3.x kwargs
                engine = PaddleOCR(
                    lang=lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except TypeError:  # 2.x — no unwarping by default
                engine = PaddleOCR(use_angle_cls=True, lang=lang)
        except Exception:
            # Unsupported recogniser language -> fall back to English (detection,
            # i.e. the boxes, is script-agnostic; the VLM supplies the reading).
            if lang == self.lang:
                raise
            self._engines[lang] = self._engine_for(self.lang)
            return self._engines[lang]

        mode = "predict" if hasattr(engine, "predict") else "ocr"
        self._engines[lang] = (engine, mode)
        return self._engines[lang]

    # -- stage entrypoint ---------------------------------------------------

    def run(self, doc: Document, cfg: Config) -> Document:
        from ..routing import resolve

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return doc

        targets = [p for p in doc.pages if p.needs_ocr]
        if not targets:
            return doc

        scale = self.dpi / 72.0
        try:
            with fitz.open(doc.source_path) as pdf:
                for page in targets:
                    if page.index >= pdf.page_count:
                        continue
                    route = resolve(page.script or "latin", cfg)
                    pg = pdf[page.index]
                    # derotation matrix maps the upright-render coords back into the
                    # PDF's native space, so boxes land on rotated pages.
                    deroter = pg.derotation_matrix
                    img = self._rasterise(fitz, pg)

                    if route.engine == "apple_vision":
                        lines, source = self._run_vision(img, page.script), "vision"
                    else:
                        try:
                            engine, mode = self._engine_for(route.paddle_lang)
                        except ImportError:
                            return doc  # no PaddleOCR -> degrade cleanly
                        lines, source = self._run(engine, mode, img), "paddle"

                    for i, (pts_px, text, conf) in enumerate(lines):
                        if not text:
                            continue
                        pts = []
                        for x, y in pts_px:
                            p = fitz.Point(x / scale, y / scale) * deroter
                            pts.append((p.x, p.y))
                        page.segments.append(
                            Segment(
                                id=f"p{page.index}-l{i}",
                                page=page.index,
                                box=Box(points=pts),
                                det_text=text,
                                det_conf=conf,
                                source=source,
                            )
                        )
        except ImportError:
            return doc
        return doc

    # -- helpers ------------------------------------------------------------

    def _rasterise(self, fitz, pg):
        import numpy as np

        pix = pg.get_pixmap(dpi=self.dpi)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:  # drop alpha
            arr = arr[:, :, :3]
        elif pix.n == 1:  # grey -> 3-channel
            arr = np.repeat(arr, 3, axis=2)
        return np.ascontiguousarray(arr)

    def _run(self, engine, mode, img) -> list[tuple[list, str, float]]:
        """Return [(quad_points_px, text, confidence), ...] across both APIs."""
        if mode == "predict":
            return self._parse_v3(engine.predict(img))
        return self._parse_v2(engine.ocr(img, cls=True))

    def _run_vision(self, img, script) -> list[tuple[list, str, float]]:
        """Apple Vision engine — same (quad_px, text, conf) shape. Degrades to empty
        (so fusion falls back) if Vision/ocrmac/PIL are unavailable."""
        try:
            from PIL import Image

            from ..engines import apple_vision
            langs = apple_vision.VISION_LANGS.get(script or "latin", ["en-US"])
            return apple_vision.recognize(Image.fromarray(img), langs)
        except ImportError:
            return []   # ocrmac/PIL not installed (e.g. non-macOS) — routing handles it
        except Exception:
            # A RUNTIME Vision failure on a page we routed to it: this silently drops the
            # page's geometry (no overlay/search there), so surface it rather than hide it.
            _log.warning("Apple Vision failed on a page (script=%s); no boxes for it",
                         script, exc_info=True)
            return []

    @staticmethod
    def _parse_v2(result) -> list[tuple[list, str, float]]:
        if not result:
            return []
        page = result[0]  # one image in -> one result out
        out: list[tuple[list, str, float]] = []
        for line in page or []:
            box, (text, score) = line[0], line[1]
            out.append(([(float(x), float(y)) for x, y in box], text, float(score)))
        return out

    @staticmethod
    def _parse_v3(result) -> list[tuple[list, str, float]]:
        out: list[tuple[list, str, float]] = []
        for res in result or []:
            data = res.get("res", res) if hasattr(res, "get") else getattr(res, "json", res)
            try:
                polys = data["rec_polys"]
                texts = data["rec_texts"]
                scores = data["rec_scores"]
            except (KeyError, TypeError):
                continue
            for poly, text, score in zip(polys, texts, scores):
                pts = [(float(p[0]), float(p[1])) for p in poly]
                out.append((pts, text, float(score)))
        return out
