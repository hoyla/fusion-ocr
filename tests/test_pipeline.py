"""Walking-skeleton plumbing tests — no models, no network.

They prove the contract that lets us fill stages in one at a time: a PDF flows
end-to-end, the record round-trips, resume works, and the airgap guard bites.
"""

from __future__ import annotations

import socket

import pytest

from fusion_ocr import config as config_mod
from fusion_ocr.models import Box, Document, Page, Segment
from fusion_ocr.pipeline import DEFAULT_PIPELINE, process


def _seg(text: str) -> Segment:
    return Segment(id="s1", page=0, box=Box(points=[(0, 0), (10, 0), (10, 5), (0, 5)]),
                   det_text=text, det_conf=0.9, source="paddle")


def test_document_json_roundtrip():
    doc = Document(source_path="x.pdf", sha256="abc", languages=["en"])
    doc.pages = [Page(index=0, segments=[_seg("hello")])]
    again = Document.from_json(doc.to_json())
    assert again.sha256 == "abc"
    assert again.pages[0].segments[0].det_text == "hello"
    assert again.pages[0].segments[0].box.bbox == (0, 0, 10, 5)


def test_fusion_picks_best_text():
    from fusion_ocr.stages.fusion import Fusion

    doc = Document(source_path="x.pdf", sha256="abc")
    seg = _seg("raw ocr")
    seg.vlm_text = "clean reading"
    doc.pages = [Page(index=0, segments=[seg])]
    Fusion().run(doc, config_mod.Config())
    assert doc.pages[0].segments[0].best_text == "clean reading"
    assert doc.pages[0].segments[0].source == "fused"


def test_airgap_blocks_non_loopback():
    config_mod.enforce_airgap()
    s = socket.socket()
    with pytest.raises(OSError):
        s.connect(("203.0.113.1", 80))  # TEST-NET-3, must be refused
    s.close()


@pytest.mark.skipif(
    pytest.importorskip("fitz", reason="PyMuPDF not installed") is None,
    reason="needs PyMuPDF",
)
def test_end_to_end_emits_artifacts(tmp_path):
    import fitz

    pdf_path = tmp_path / "sample.pdf"
    d = fitz.open()
    page = d.new_page()
    page.insert_text((72, 72), "Born-digital sample line.")
    d.save(str(pdf_path))
    d.close()

    cfg = config_mod.Config(out_dir=tmp_path / "out", airgap=False)
    doc = process(pdf_path, cfg, pipeline=DEFAULT_PIPELINE)

    assert doc.stage_completed == "render"
    assert "segment_index" in doc.artifacts
    assert (tmp_path / "out" / doc.sha256 / "segment_index.json").exists()
    assert (tmp_path / "out" / doc.sha256 / "doc.json").exists()
    assert doc.pages and doc.pages[0].has_text_layer is True
