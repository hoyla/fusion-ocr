"""Real-engine geometry tests — the `engines` tier.

Builds an IMAGE-ONLY PDF (text rasterised into an image, no text layer) so triage flags it
for OCR, runs each deterministic engine through the production OcrDet path, and asserts the
SAME contract for all of them: words recovered; boxes in PDF points inside the page; the box
bracketing the known text position (what catches the PaddleOCR 3.x unwarp-offset bug, ~18 pt);
segments carrying the engine's own source label (one fusion recognises); the overlay produced
and searchable — and, second phase, fusion marrying a VLM reading onto those real boxes. That
last part is the 2026-09-16 RapidOCR lesson: an engine whose boxes fusion didn't recognise was
never fused at all, and the A/B never ran with a reading, so nothing noticed for a month.

Each engine skips where it isn't installed / available, so the suite stays green without the
heavy stacks. `pytest -m engines` runs just this tier; `pytest -m "not engines"` skips it for a
fast loop. Hermetic: no layout model (heavy download) and no VLM call (the script probe's
connection refusal is expected and harmless).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz", reason="needs PyMuPDF")
import fitz  # noqa: E402

from fusion_ocr import config as config_mod  # noqa: E402
from fusion_ocr.engines import apple_vision, rapid  # noqa: E402
from fusion_ocr.models import OCR_SOURCES  # noqa: E402
from fusion_ocr.pipeline import process  # noqa: E402
from fusion_ocr.stages.fusion import Fusion  # noqa: E402
from fusion_ocr.stages.language import Language  # noqa: E402
from fusion_ocr.stages.ocr_det import OcrDet  # noqa: E402
from fusion_ocr.stages.render import Render  # noqa: E402
from fusion_ocr.stages.triage import Triage  # noqa: E402

# OCR-only pipeline: no Layout (heavy model download) and no VlmRead (live VLM call).
_OCR_PIPELINE = [Triage(), Language(), OcrDet(), Fusion(), Render()]

# Known insertion geometry — the box must bracket this.
_BASELINE_X = 72.0
_BASELINE_Y = 200.0
_FONTSIZE = 32
_TEXT = "HELLO WORLD invoice 2026"


def _image_only_pdf(path, text=_TEXT):
    """Render text onto a page, flatten to a pixmap, and rebuild the page from the
    image — leaving no extractable text layer."""
    src = fitz.open()
    pg = src.new_page(width=612, height=792)
    pg.insert_text((_BASELINE_X, _BASELINE_Y), text, fontsize=_FONTSIZE)
    pix = pg.get_pixmap(dpi=200)
    src.close()

    out = fitz.open()
    page = out.new_page(width=612, height=792)
    page.insert_image(page.rect, pixmap=pix)
    out.save(str(path))
    out.close()


def _paddle_available() -> bool:
    try:
        import paddleocr  # noqa: F401
        return True
    except ImportError:
        return False


# (engine, config overrides that route to it, the source label its segments must carry)
ENGINES = [
    pytest.param("paddle", {}, "paddle",
                 marks=pytest.mark.skipif(not _paddle_available(), reason="needs the `ocr` extra (paddleocr)")),
    pytest.param("apple_vision", {"prefer_apple_vision": True}, "vision",
                 marks=pytest.mark.skipif(not apple_vision.available(), reason="Apple Vision: macOS + the `vision` extra")),
    pytest.param("rapidocr", {"prefer_rapidocr": True}, "rapid",
                 marks=pytest.mark.skipif(not rapid.available(), reason="needs the `rapid` extra")),
]


@pytest.mark.engines
@pytest.mark.parametrize("engine,overrides,label", ENGINES)
def test_engine_recovers_boxed_text_and_fusion_marries_a_reading_onto_it(tmp_path, engine, overrides, label):
    pdf_path = tmp_path / "scan.pdf"
    _image_only_pdf(pdf_path)
    with fitz.open(pdf_path) as d:                      # genuinely image-only: nothing to cheat from
        assert d[0].get_text("text").strip() == ""

    cfg = config_mod.Config(out_dir=tmp_path / "out", airgap=False, **overrides)
    doc = process(pdf_path, cfg, pipeline=_OCR_PIPELINE)
    page = doc.pages[0]
    segs = [s for s in page.segments if not s.superseded]
    assert segs, f"expected {engine} to detect at least one line"

    # -- the engine's label is its own, and one fusion recognises as an OCR box
    assert {s.source for s in segs} == {label}
    assert label in OCR_SOURCES

    # -- recognition: the words came back
    combined = " ".join(s.det_text or "" for s in segs).lower()
    assert "hello" in combined or "world" in combined, combined

    # -- geometry: real PDF points inside the page, with a confidence
    for s in segs:
        assert s.det_conf is not None and 0.0 <= s.det_conf <= 1.0
        x0, y0, x1, y1 = s.box.bbox
        assert 0 <= x0 < x1 <= page.width + 1
        assert 0 <= y0 < y1 <= page.height + 1

    # -- coordinate ACCURACY, not just bounds: the leftmost box must bracket the known text
    #    start (an engine may split the line into two boxes; the first one holds "HELLO").
    x0, y0, x1, y1 = min(segs, key=lambda s: s.box.bbox[0]).box.bbox
    assert abs(x0 - _BASELINE_X) <= 12, f"{engine}: left edge {x0} off from {_BASELINE_X}"
    assert y0 <= _BASELINE_Y <= y1 + 4, f"{engine}: baseline {_BASELINE_Y} not bracketed by {(x0, y0, x1, y1)}"
    assert (y1 - y0) <= _FONTSIZE * 1.6, f"{engine}: box too tall: {y1 - y0}"

    # -- the overlay is produced and its invisible text is searchable
    assert "overlay_pdf" in doc.artifacts
    with fitz.open(doc.artifacts["overlay_pdf"]) as ov:
        overlaid = ov[0].get_text("text").lower()
    assert "hello" in overlaid or "world" in overlaid, overlaid

    # -- phase two: with a VLM reading present, fusion must marry it onto THESE boxes.
    #    (RapidOCR boxes silently kept their det_text here until #48.)
    page.vlm_reading = _TEXT
    Fusion().run(doc, cfg)
    fused = [s for s in page.segments if not s.superseded]
    assert fused and all(s.source == "fused" for s in fused), [s.source for s in fused]
    assert " ".join(s.best_text for s in fused).split() == _TEXT.split()
    assert all(s.det_text for s in fused)            # provenance: the engine's text kept beside it


def test_engine_for_passes_model_overrides_and_keys_cache(monkeypatch):
    # det_model/rec_model (the engine-A/B knob) must reach PaddleOCR as the 3.x
    # model-name kwargs, and the engine cache must key on them — config can change
    # between runs in one process (PATCH /config), so a lang-only key would serve
    # a stale engine. PaddleOCR itself is stubbed: this is wiring, not inference.
    paddleocr = pytest.importorskip("paddleocr", reason="needs the `ocr` extra")

    calls = []

    class _FakeEngine:
        def __init__(self, **kw):
            calls.append(kw)

        def predict(self, img):
            return []

    monkeypatch.setattr(paddleocr, "PaddleOCR", _FakeEngine)
    stage = OcrDet()
    e_default, _ = stage._engine_for("en")
    e_v5, _ = stage._engine_for("en", "PP-OCRv5_server_det", "PP-OCRv5_server_rec")
    assert e_default is not e_v5                       # distinct cache entries
    assert "text_detection_model_name" not in calls[0]  # default passes no override
    assert calls[1]["text_detection_model_name"] == "PP-OCRv5_server_det"
    assert calls[1]["text_recognition_model_name"] == "PP-OCRv5_server_rec"
    e_again, _ = stage._engine_for("en", "PP-OCRv5_server_det", "PP-OCRv5_server_rec")
    assert e_again is e_v5 and len(calls) == 2         # cache hit, no rebuild
