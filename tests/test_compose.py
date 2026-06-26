"""Per-region mixed-content composition — pure helpers + integration. No deps.

Covers the case that motivated this: a page that is part machine-readable text layer
(e.g. a header imposed over a scan) and part image — both must be RETAINED and COMBINED
in reading order, with the weaker source kept (superseded) for provenance.
"""

from __future__ import annotations

from fusion_ocr import config as config_mod
from fusion_ocr.compose import (
    classify_regions, in_machine_readable_region, reading_key, region_text_coverage,
)
from fusion_ocr.models import Box, Document, Page, Region, Segment
from fusion_ocr.stages.fusion import Fusion


def _box(x0, y0, x1, y1):
    return Box(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _seg(id, x0, y0, x1, y1, text="", source="paddle", best=None):
    s = Segment(id=id, page=0, box=_box(x0, y0, x1, y1), det_text=text,
                det_conf=0.9, source=source)
    if best is not None:
        s.best_text = best
    return s


def _region(x0, y0, x1, y1, order=0):
    r = Region(box=_box(x0, y0, x1, y1))
    r.reading_order = order
    return r


# ---- pure helpers --------------------------------------------------------

def test_region_text_coverage():
    region = _region(0, 0, 100, 100)            # area 10000
    full = [_box(0, 0, 100, 100)]
    half = [_box(0, 0, 100, 50)]                 # 5000 -> 0.5
    none = [_box(200, 200, 300, 300)]
    assert region_text_coverage(region, full) == 1.0
    assert region_text_coverage(region, half) == 0.5
    assert region_text_coverage(region, none) == 0.0


def test_classify_regions_machine_readable_vs_ocr():
    header = _region(0, 0, 600, 100, order=0)
    body = _region(0, 100, 600, 800, order=1)
    clean = [_seg("h1", 10, 10, 590, 45, source="textlayer", best="Header 1"),
             _seg("h2", 10, 55, 590, 90, source="textlayer", best="Header 2")]
    classify_regions([header, body], clean)
    assert header.source == "textlayer"   # clean text covers ~58% of it
    assert body.source == "ocr"           # no clean text -> image -> OCR


def test_in_machine_readable_region_and_reading_key():
    header = _region(0, 0, 600, 100, order=0); header.source = "textlayer"
    body = _region(0, 100, 600, 800, order=1); body.source = "ocr"
    regions = [header, body]
    in_header = _seg("a", 10, 20, 200, 40)
    in_body = _seg("b", 10, 200, 200, 220)
    assert in_machine_readable_region(in_header, regions) is True
    assert in_machine_readable_region(in_body, regions) is False
    # reading order: header region (0) sorts before body region (1)
    assert reading_key(in_header, regions) < reading_key(in_body, regions)


# ---- integration: the mixed page -----------------------------------------

def _mixed_page():
    """Header = clean text layer; body = scanned image (OCR). Plus one redundant OCR
    box inside the header that must be superseded by the exact text layer."""
    page = Page(index=0, needs_ocr=True, width=600, height=800)
    page.regions = [_region(0, 0, 600, 100, order=0), _region(0, 100, 600, 800, order=1)]
    page.segments = [
        # machine-readable header (clean text layer, has best_text)
        _seg("h1", 10, 10, 590, 45, source="textlayer", best="Header line one"),
        _seg("h2", 10, 55, 590, 90, source="textlayer", best="Header line two"),
        # OCR of the scanned body
        _seg("b1", 10, 200, 300, 230, text="body line one"),
        _seg("b2", 10, 250, 320, 280, text="body line two"),
        # redundant OCR inside the header region -> should be superseded
        _seg("hx", 12, 12, 588, 43, text="Headr llne one"),
    ]
    return Document(source_path="x", sha256="x", pages=[page])


def test_mixed_page_retains_and_combines_both_sets():
    doc = _mixed_page()
    Fusion().run(doc, config_mod.Config())
    segs = doc.pages[0].segments

    # nothing dropped — all five retained
    assert len(segs) == 5
    # the redundant header OCR is superseded; the exact text layer wins
    hx = next(s for s in segs if s.id == "hx")
    assert hx.superseded is True
    # both sets present and primary
    primary = [s for s in segs if not s.superseded]
    sources = {s.source for s in primary}
    assert "textlayer" in sources and "paddle" in sources   # combined
    # regions classified
    assert doc.pages[0].regions[0].source == "textlayer"
    assert doc.pages[0].regions[1].source == "ocr"


def test_mixed_page_reading_order_header_then_body():
    doc = _mixed_page()
    Fusion().run(doc, config_mod.Config())
    primary = [s for s in doc.pages[0].segments if not s.superseded and s.best_text]
    texts = [s.best_text for s in primary]
    assert texts == ["Header line one", "Header line two",
                     "body line one", "body line two"]


def test_contaminated_header_superseded_by_ocr():
    # header text layer is PUA-contaminated (no best_text) over an OCR'd region
    page = Page(index=0, needs_ocr=True, width=600, height=800)
    page.regions = [_region(0, 0, 600, 200, order=0)]   # one region, gets OCR'd
    bad = _seg("bad", 10, 10, 400, 40, source="textlayer")     # contaminated: no best_text
    ocr = _seg("ocr", 12, 12, 398, 38, text="clean ocr text")  # overlaps it
    page.segments = [bad, ocr]
    doc = Document(source_path="x", sha256="x", pages=[page])
    Fusion().run(doc, config_mod.Config())

    assert next(s for s in doc.pages[0].segments if s.id == "bad").superseded is True
    ocr2 = next(s for s in doc.pages[0].segments if s.id == "ocr")
    assert ocr2.superseded is False and ocr2.best_text == "clean ocr text"
