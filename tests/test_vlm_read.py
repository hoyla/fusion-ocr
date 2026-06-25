"""VLM read stage — mock client, no Ollama/model needed.

vlm_read now only produces page.vlm_reading; aligning it onto boxes is fusion's job
(see test_fusion).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz", reason="needs PyMuPDF")
import fitz  # noqa: E402

from fusion_ocr import config as config_mod  # noqa: E402
from fusion_ocr.models import Document, Page  # noqa: E402
from fusion_ocr.stages.vlm_read import VlmRead  # noqa: E402


class _FakeVLM:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def read(self, image_png, prompt, **opts):
        self.calls += 1
        return self.text


def test_vlm_read_sets_reading(tmp_path):
    pdf = tmp_path / "scan.pdf"
    d = fitz.open(); pg = d.new_page()
    pg.insert_image(pg.rect, pixmap=fitz.open().new_page().get_pixmap(dpi=72))
    d.save(str(pdf)); d.close()

    doc = Document(source_path=str(pdf), sha256="x")
    doc.pages = [Page(index=0, needs_ocr=True, width=612, height=792)]

    fake = _FakeVLM("First real line\nSecond real line")
    VlmRead(client=fake).run(doc, config_mod.Config())

    assert fake.calls == 1
    assert doc.pages[0].vlm_reading == "First real line\nSecond real line"


def test_born_digital_skips_vlm(tmp_path):
    pdf = tmp_path / "born.pdf"
    d = fitz.open(); d.new_page().insert_text((72, 72), "hi"); d.save(str(pdf)); d.close()
    doc = Document(source_path=str(pdf), sha256="x")
    doc.pages = [Page(index=0, needs_ocr=False)]
    fake = _FakeVLM("should not be called")
    VlmRead(client=fake).run(doc, config_mod.Config())
    assert fake.calls == 0
