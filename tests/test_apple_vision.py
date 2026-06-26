"""Apple Vision engine integration — coord mapping, routing, cheap-tier skip. No deps."""

from __future__ import annotations

from fusion_ocr import config as config_mod
from fusion_ocr.engines import apple_vision
from fusion_ocr.engines.apple_vision import _to_pixel_quads
from fusion_ocr.models import Box, Document, Page, Segment
from fusion_ocr.routing import resolve
from fusion_ocr.stages.fusion import Fusion
from fusion_ocr.stages.vlm_read import _vision_confident


def test_normalized_bottomleft_maps_to_pixel_topleft():
    # one box at x=0.1,y=0.8,w=0.2,h=0.1 on a 1000x2000 image
    quads = _to_pixel_quads([("Hi", 0.9, (0.1, 0.8, 0.2, 0.1))], 1000, 2000)
    assert len(quads) == 1
    pts, text, conf = quads[0]
    xs = [round(p[0]) for p in pts]; ys = [round(p[1]) for p in pts]
    assert text == "Hi" and conf == 0.9
    assert min(xs) == 100 and max(xs) == 300            # x*W .. (x+w)*W
    # y flipped: top = (1-(y+h))*H = (1-0.9)*2000 = 200 ; bottom = (1-y)*H = 400
    assert min(ys) == 200 and max(ys) == 400


def test_vision_langs_cover_supported_scripts():
    for s in ("latin", "thai", "cyrillic", "arabic", "cjk"):
        assert s in apple_vision.VISION_LANGS
    assert "devanagari" not in apple_vision.VISION_LANGS  # Vision has none -> PaddleOCR


def test_route_prefers_vision_when_enabled(monkeypatch):
    monkeypatch.setattr(apple_vision, "available", lambda: True)  # force availability
    cfg = config_mod.Config(prefer_apple_vision=True)
    assert resolve("latin", cfg).engine == "apple_vision"
    assert resolve("thai", cfg).engine == "apple_vision"
    assert resolve("devanagari", cfg).engine == "paddle"        # unsupported -> paddle


def test_route_default_engine_is_paddle():
    assert resolve("latin", config_mod.Config()).engine == "paddle"


def _seg(conf, src):
    return Segment(id="s", page=0, box=Box(points=[(0, 0), (9, 0), (9, 9), (0, 9)]),
                   det_text="text", det_conf=conf, source=src)


def test_vision_confident_gate():
    page = Page(index=0)
    page.segments = [_seg(0.97, "vision"), _seg(0.95, "vision")]
    assert _vision_confident(page, 0.92) is True
    page.segments = [_seg(0.5, "vision")]          # low conf -> not confident -> VLM
    assert _vision_confident(page, 0.92) is False
    page.segments = [_seg(0.99, "paddle")]         # paddle isn't Vision
    assert _vision_confident(page, 0.92) is False


def test_fusion_treats_vision_as_ocr():
    page = Page(index=0)
    s = _seg(0.97, "vision"); s.best_text = ""; s.det_text = "Vision read this"
    page.segments = [s]
    doc = Document(source_path="x", sha256="x", pages=[page])
    Fusion().run(doc, config_mod.Config())
    # no VLM reading -> fusion fills best_text from the Vision det_text
    assert doc.pages[0].segments[0].best_text == "Vision read this"
