"""Born-digital-as-ground-truth eval runner.

For each born-digital page: the embedded text layer is the reference; render the page to
an image (dropping the text layer), run it through the pipeline as a scan, and score the
recovered text against the reference. No hand-labelling.

Caveat carried in __init__: CER/WER here are END-TO-END (recognition AND reading order),
so dense multi-column / table pages inflate the rate from order differences alone, not
pure recognition error. Eval prose pages for a clean recognition number; tables stress
the layout/reading-order path. The insertion rate is the hallucination proxy.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

from ..config import Config
from .metrics import normalize, score

_MIN_REF_CHARS = 50   # too little text to score (cover/figure page) -> skip


def page_text_layer(pdf_path, page_index: int) -> str:
    # NB: this is PDF content-stream order, which is NOT guaranteed to be visual reading
    # order on multi-column pages (and sort=True is worse — a naive y,x sort interleaves
    # columns). So CER/WER carry reading-order noise on multi-column; the reliable
    # recognition signal is word recall/precision. See __init__ for the caveat.
    import fitz
    with fitz.open(pdf_path) as d:
        return d[page_index].get_text("text")


def make_image_only_pdf(src, page_index: int, dst, dpi: int = 200) -> None:
    """Render one page to a raster and wrap it as a 1-page image-only PDF (no text
    layer), so the pipeline must OCR it."""
    import fitz
    with fitz.open(src) as d:
        pix = d[page_index].get_pixmap(dpi=dpi)
    out = fitz.open()
    w, h = pix.width * 72.0 / dpi, pix.height * 72.0 / dpi
    page = out.new_page(width=w, height=h)
    page.insert_image(page.rect, pixmap=pix)
    out.save(str(dst))
    out.close()


def recovered_text(page) -> str:
    """The text the pipeline recovered for a page: the VLM reading if present, else the
    deterministic segments in reading order (Apple Vision / Paddle)."""
    if page.vlm_reading.strip():
        return page.vlm_reading
    from ..compose import reading_key
    segs = [s for s in page.segments if s.best_text and not s.superseded]
    segs.sort(key=lambda s: reading_key(
        s, page.regions, page.rotation, page.width, page.height))
    return "\n".join(s.best_text for s in segs)


def evaluate_pdf(pdf_path, cfg: Config, pages=None, dpi: int = 200,
                 tmp_root=None) -> list[dict]:
    """Score selected born-digital pages of one PDF. Returns a per-page score() list
    (each annotated with pdf/page)."""
    import fitz
    from ..pipeline import process

    pdf_path = Path(pdf_path)
    with fitz.open(pdf_path) as d:
        n = d.page_count
    sel = list(pages) if pages is not None else list(range(n))
    tmp_root = Path(tmp_root or tempfile.mkdtemp(prefix="fusion_eval_"))
    eval_cfg = dataclasses.replace(cfg, out_dir=tmp_root / "out")

    results = []
    for pi in sel:
        if pi >= n:
            continue
        gt = page_text_layer(pdf_path, pi)
        if len(normalize(gt)) < _MIN_REF_CHARS:
            continue
        img_pdf = tmp_root / f"{pdf_path.stem}_p{pi}.pdf"
        make_image_only_pdf(pdf_path, pi, img_pdf, dpi=dpi)
        doc = process(img_pdf, eval_cfg)
        hyp = recovered_text(doc.pages[0]) if doc.pages else ""
        results.append({"pdf": str(pdf_path), "page": pi, **score(gt, hyp)})
    return results


def evaluate(pdf_paths, cfg: Config, pages=None, dpi: int = 200) -> list[dict]:
    out: list[dict] = []
    for p in pdf_paths:
        out += evaluate_pdf(p, cfg, pages=pages, dpi=dpi)
    return out
