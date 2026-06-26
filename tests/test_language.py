"""Language stage — text-layer detection + image-only VLM script probe."""

from __future__ import annotations

import pytest

pytest.importorskip("fitz", reason="needs PyMuPDF")
import fitz  # noqa: E402

from fusion_ocr import config as config_mod  # noqa: E402
from fusion_ocr.models import Box, Document, Page, Segment  # noqa: E402
from fusion_ocr.stages.language import Language, _probe_script  # noqa: E402


class _FakeVLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def read(self, png, prompt, **kw):
        self.calls += 1
        return self.answer


def _img_pdf(tmp_path):
    p = tmp_path / "scan.pdf"
    d = fitz.open(); d.new_page(); d.save(str(p)); d.close()
    return p


def test_probe_parsing():
    assert _probe_script(_FakeVLM("Thai"), b"") == "thai"
    assert _probe_script(_FakeVLM("The script is Cyrillic."), b"") == "cyrillic"  # scans all words
    assert _probe_script(_FakeVLM("Latin"), b"") == "latin"
    assert _probe_script(_FakeVLM("Japanese"), b"") == "cjk"
    assert _probe_script(_FakeVLM("gibberish"), b"") == ""


def test_image_only_page_is_probed(tmp_path):
    doc = Document(source_path=str(_img_pdf(tmp_path)), sha256="x")
    doc.pages = [Page(index=0, needs_ocr=True)]  # no text layer
    fake = _FakeVLM("Thai")
    Language(client=fake).run(doc, config_mod.Config())
    assert fake.calls == 1
    assert doc.pages[0].script == "thai"


def test_textlayer_page_not_probed(tmp_path):
    doc = Document(source_path=str(_img_pdf(tmp_path)), sha256="x")
    page = Page(index=0, needs_ocr=True)
    page.segments = [Segment(id="a", page=0,
                             box=Box(points=[(0, 0), (10, 0), (10, 5), (0, 5)]),
                             det_text="Hello world this is English", source="textlayer")]
    doc.pages = [page]
    fake = _FakeVLM("Thai")
    Language(client=fake).run(doc, config_mod.Config())
    assert fake.calls == 0            # text layer already classified it
    assert doc.pages[0].script == "latin"
