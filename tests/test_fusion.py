"""Fusion stage — overlap dedup + line clustering / sequence alignment. No deps."""

from __future__ import annotations

from fusion_ocr import config as config_mod
from fusion_ocr.models import Box, Document, Page, Segment
from fusion_ocr.stages.fusion import Fusion, _cluster_lines, _nw_align


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


def test_overlap_dedup_prefers_clean_textlayer():
    page = Page(index=0)
    tl = _seg("tl", 50, 100, 300, 120, "exact text", source="textlayer")
    tl.best_text = "exact text"
    pad = _seg("pad", 52, 101, 298, 119, "exatc txet", source="paddle")  # overlaps tl
    page.segments = [tl, pad]
    doc = Document(source_path="x", sha256="x", pages=[page])

    Fusion().run(doc, config_mod.Config())
    kept = doc.pages[0].segments
    assert len(kept) == 1 and kept[0].source == "textlayer"


def test_fallback_best_text_without_vlm_reading():
    page = Page(index=0)
    s = _seg("a", 50, 100, 150, 120, "raw ocr")
    s.vlm_text = "clean reading"
    page.segments = [s]
    doc = Document(source_path="x", sha256="x", pages=[page])
    Fusion().run(doc, config_mod.Config())
    assert doc.pages[0].segments[0].best_text == "clean reading"
    assert doc.pages[0].segments[0].source == "fused"
