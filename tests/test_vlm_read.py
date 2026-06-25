"""VLM read stage — mock client, no Ollama/model needed.

Verifies the stage calls the (injected) client per OCR-bound page, stores the raw
reading, aligns lines onto the deterministic boxes, and that fusion then promotes
the VLM text to best_text.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz", reason="needs PyMuPDF")
import fitz  # noqa: E402

from fusion_ocr import config as config_mod  # noqa: E402
from fusion_ocr.models import Box, Document, Page, Segment  # noqa: E402
from fusion_ocr.stages.fusion import Fusion  # noqa: E402
from fusion_ocr.stages.vlm_read import VlmRead, _align_to_boxes  # noqa: E402


class _FakeVLM:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def read(self, image_png, prompt, **opts):
        self.calls += 1
        return self.text


def _box(y):
    return Box(points=[(50, y), (300, y), (300, y + 20), (50, y + 20)])


def test_vlm_read_aligns_lines_to_boxes(tmp_path):
    # One image-only page so the stage actually renders + calls the client.
    pdf = tmp_path / "scan.pdf"
    d = fitz.open(); pg = d.new_page()
    pg.insert_image(pg.rect, pixmap=fitz.open().new_page().get_pixmap(dpi=72))
    d.save(str(pdf)); d.close()

    doc = Document(source_path=str(pdf), sha256="x")
    page = Page(index=0, needs_ocr=True, width=612, height=792)
    page.segments = [
        Segment(id="a", page=0, box=_box(100), det_text="garbage1", source="paddle"),
        Segment(id="b", page=0, box=_box(140), det_text="garbage2", source="paddle"),
    ]
    doc.pages = [page]

    fake = _FakeVLM("First real line\nSecond real line")
    VlmRead(client=fake).run(doc, config_mod.Config())

    assert fake.calls == 1
    assert page.vlm_reading == "First real line\nSecond real line"
    assert page.segments[0].vlm_text == "First real line"
    assert page.segments[1].vlm_text == "Second real line"

    # Fusion promotes the VLM reading over the garbage OCR text.
    Fusion().run(doc, config_mod.Config())
    assert page.segments[0].best_text == "First real line"
    assert page.segments[0].source == "fused"


def test_align_proportional_when_counts_differ():
    page = Page(index=0)
    page.segments = [Segment(id=str(i), page=0, box=_box(100 + 30 * i),
                             source="paddle") for i in range(4)]
    page.vlm_reading = "alpha\nbeta"  # 2 lines onto 4 boxes
    _align_to_boxes(page)
    got = [s.vlm_text for s in page.segments]
    assert got[0] == "alpha" and got[-1] == "beta"
    assert all(t in ("alpha", "beta") for t in got)


def test_born_digital_skips_vlm(tmp_path):
    pdf = tmp_path / "born.pdf"
    d = fitz.open(); d.new_page().insert_text((72, 72), "hi"); d.save(str(pdf)); d.close()
    doc = Document(source_path=str(pdf), sha256="x")
    doc.pages = [Page(index=0, needs_ocr=False)]
    fake = _FakeVLM("should not be called")
    VlmRead(client=fake).run(doc, config_mod.Config())
    assert fake.calls == 0
