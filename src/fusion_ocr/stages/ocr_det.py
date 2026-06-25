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

from ..config import Config
from ..models import Box, Document, Segment

_DEFAULT_DPI = 200


class OcrDet:
    name = "ocr_det"

    def __init__(self, dpi: int = _DEFAULT_DPI, lang: str = "en") -> None:
        self.dpi = dpi
        self.lang = lang
        self._engine = None
        self._mode: str | None = None  # "predict" (3.x) | "ocr" (2.x)

    # -- engine lifecycle ---------------------------------------------------

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        from paddleocr import PaddleOCR  # raises ImportError if extra absent

        try:  # 3.x rejects some 2.x kwargs; keep the constructor minimal.
            self._engine = PaddleOCR(lang=self.lang)
        except TypeError:
            self._engine = PaddleOCR(use_angle_cls=True, lang=self.lang)
        self._mode = "predict" if hasattr(self._engine, "predict") else "ocr"
        return self._engine

    # -- stage entrypoint ---------------------------------------------------

    def run(self, doc: Document, cfg: Config) -> Document:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return doc

        targets = [p for p in doc.pages if not p.has_text_layer]
        if not targets:
            return doc

        try:
            engine = self._ensure_engine()
        except ImportError:
            # No PaddleOCR — leave segments empty; downstream degrades cleanly.
            return doc

        scale = self.dpi / 72.0
        with fitz.open(doc.source_path) as pdf:
            for page in targets:
                if page.index >= pdf.page_count:
                    continue
                img = self._rasterise(fitz, pdf[page.index])
                lines = self._run_engine(engine, img)
                for i, (pts_px, text, conf) in enumerate(lines):
                    if not text:
                        continue
                    pts = [(x / scale, y / scale) for x, y in pts_px]
                    page.segments.append(
                        Segment(
                            id=f"p{page.index}-l{i}",
                            page=page.index,
                            box=Box(points=pts),
                            det_text=text,
                            det_conf=conf,
                            source="paddle",
                        )
                    )
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

    def _run_engine(self, engine, img) -> list[tuple[list, str, float]]:
        """Return [(quad_points_px, text, confidence), ...] across both APIs."""
        if self._mode == "predict":
            return self._parse_v3(engine.predict(img))
        return self._parse_v2(engine.ocr(img, cls=True))

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
