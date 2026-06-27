"""Fusion stage — overlap dedup + line clustering / sequence alignment. No deps."""

from __future__ import annotations

from fusion_ocr import config as config_mod
from fusion_ocr.models import Box, Document, Page, Region, Segment
from fusion_ocr.stages.fusion import (
    Fusion, _cluster_lines, _cluster_within_regions, _nw_align,
)


def _seg(id, x0, y0, x1, y1, det="", source="paddle"):
    return Segment(id=id, page=0, box=Box(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
                   det_text=det, det_conf=0.5, source=source)


def test_cluster_lines_groups_by_row():
    segs = [
        _seg("a", 50, 100, 150, 120, "Dear"),
        _seg("b", 160, 100, 300, 120, "David"),   # same row as a
        _seg("c", 50, 140, 200, 160, "Today"),    # next row
    ]
    clusters = _cluster_lines(segs)
    assert len(clusters) == 2
    assert [s.id for s in clusters[0]] == ["a", "b"]  # left-to-right within row
    assert [s.id for s in clusters[1]] == ["c"]


def test_nw_align_matches_in_order():
    mapping = _nw_align(["Dear David", "Today is polling"],
                        ["Dear David", "Today is polling day"])
    assert mapping == {0: 0, 1: 1}


def test_line_fusion_no_duplication():
    # Two visual lines, each over-segmented into two PaddleOCR boxes.
    page = Page(index=0, needs_ocr=True, width=612, height=792)
    page.segments = [
        _seg("a", 50, 100, 150, 120, "Dea"),
        _seg("b", 160, 100, 300, 120, "Daviid"),
        _seg("c", 50, 140, 200, 160, "Todai is"),
        _seg("d", 210, 140, 320, 160, "poling"),
    ]
    page.vlm_reading = "Dear David\nToday is polling day"
    doc = Document(source_path="x", sha256="x", pages=[page])

    Fusion().run(doc, config_mod.Config())

    fused = doc.pages[0].segments
    assert len(fused) == 2  # collapsed from 4 boxes to 2 line-boxes
    texts = [s.best_text for s in fused]
    assert texts == ["Dear David", "Today is polling day"]
    # each VLM line appears exactly once -> no triple-hit duplication
    assert texts.count("Dear David") == 1
    # merged box spans both source boxes on the line
    x0, _, x1, _ = fused[0].box.bbox
    assert x0 == 50 and x1 == 300
    assert all(s.source == "fused" for s in fused)


def _region(kind, x0, y0, x1, y1, order):
    r = Region(box=Box(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), kind=kind)
    r.reading_order = order
    return r


def test_region_aware_clustering_keeps_columns_apart():
    # Two columns of text at the SAME y-bands: global clustering would merge them;
    # region-aware clustering must keep them separate and ordered left col then right.
    left = [_seg("L1", 50, 100, 150, 116, "left one"),
            _seg("L2", 50, 130, 150, 146, "left two")]
    right = [_seg("R1", 300, 100, 400, 116, "right one"),
             _seg("R2", 300, 130, 400, 146, "right two")]
    regions = [_region("paragraph", 40, 90, 160, 160, order=0),     # left column
               _region("paragraph", 290, 90, 410, 160, order=1)]    # right column

    clusters = _cluster_within_regions(left + right, regions)
    # 4 distinct line-clusters (no cross-column merge), left column first
    assert len(clusters) == 4
    ids = [[s.id for s in c] for c in clusters]
    assert ids == [["L1"], ["L2"], ["R1"], ["R2"]]


def test_unmatched_cluster_keeps_real_engine_source():
    # two Vision lines, but the VLM reading only covers one -> the unmatched cluster
    # must keep source 'vision', not be mislabelled 'paddle' (the provenance bug).
    page = Page(index=0, needs_ocr=True)
    page.segments = [_seg("v1", 50, 100, 200, 116, "alpha", source="vision"),
                     _seg("v2", 50, 140, 200, 156, "beta", source="vision")]
    page.vlm_reading = "alpha"
    doc = Document(source_path="x", sha256="x", pages=[page])
    Fusion().run(doc, config_mod.Config())
    sources = {s.source for s in doc.pages[0].segments}
    assert "fused" in sources and "vision" in sources
    assert "paddle" not in sources


def test_overlap_dedup_prefers_clean_textlayer():
    page = Page(index=0)
    tl = _seg("tl", 50, 100, 300, 120, "exact text", source="textlayer")
    tl.best_text = "exact text"
    pad = _seg("pad", 52, 101, 298, 119, "exatc txet", source="paddle")  # overlaps tl
    page.segments = [tl, pad]
    doc = Document(source_path="x", sha256="x", pages=[page])

    Fusion().run(doc, config_mod.Config())
    segs = doc.pages[0].segments
    assert len(segs) == 2  # nothing dropped — superseded is retained for provenance
    primary = [s for s in segs if not s.superseded]
    assert len(primary) == 1 and primary[0].source == "textlayer"
    assert next(s for s in segs if s.source == "paddle").superseded is True


def test_region_overlap_supersedes_ocr_under_textlayer():
    # The Thai scanned-form duplicate-header bug: a machine-readable header band sits in
    # a large region that doesn't cross the coverage threshold (so the region isn't
    # classified machine-readable), yet the OCR copy directly overlaps the exact text-
    # layer line. It must still be superseded, else the header renders twice.
    page = Page(index=0)
    page.regions = [Region(box=Box(points=[(0, 0), (600, 0), (600, 800), (0, 800)]),
                           kind="header")]
    tl = _seg("tl", 50, 100, 300, 120, "exact header", source="textlayer")
    tl.best_text = "exact header"
    pad = _seg("pad", 52, 101, 298, 119, "exatc heeder", source="paddle")  # overlaps tl
    page.segments = [tl, pad]
    doc = Document(source_path="x", sha256="x", pages=[page])

    Fusion().run(doc, config_mod.Config())
    segs = doc.pages[0].segments
    primary = [s for s in segs if not s.superseded]
    assert len(primary) == 1 and primary[0].source == "textlayer"   # no duplicate
    assert next(s for s in segs if s.source == "paddle").superseded is True
    assert len(segs) == 2                                            # OCR kept (provenance)


def test_fallback_best_text_without_vlm_reading():
    page = Page(index=0)
    s = _seg("a", 50, 100, 150, 120, "raw ocr")
    s.vlm_text = "clean reading"
    page.segments = [s]
    doc = Document(source_path="x", sha256="x", pages=[page])
    Fusion().run(doc, config_mod.Config())
    assert doc.pages[0].segments[0].best_text == "clean reading"
    assert doc.pages[0].segments[0].source == "fused"
