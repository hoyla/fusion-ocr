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


class _SeqVLM:
    """Returns canned responses in order — first the primary read, then escalation."""
    def __init__(self, *responses):
        self.responses = list(responses)
        self.i = 0

    def read(self, png, prompt, **kw):
        r = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return r


def test_confidence_gated_escalation(tmp_path):
    from fusion_ocr.models import Box, Segment
    pdf = tmp_path / "scan.pdf"
    d = fitz.open(); pg = d.new_page()
    pg.insert_image(pg.rect, pixmap=fitz.open().new_page().get_pixmap(dpi=72))
    d.save(str(pdf)); d.close()

    doc = Document(source_path=str(pdf), sha256="x")
    page = Page(index=0, needs_ocr=True, width=612, height=792)
    page.segments = [Segment(id="a", page=0,
                             box=Box(points=[(50, 100), (300, 100), (300, 120), (50, 120)]),
                             det_text="x" * 100, det_conf=0.3, source="paddle")]
    doc.pages = [page]

    cfg = config_mod.Config()
    cfg.vlm.escalation_model = "big-model"
    cfg.vlm.escalate_below = 0.6                      # mean conf 0.3 < 0.6 -> escalate

    # primary refuses; escalation gives a real read
    fake = _SeqVLM("[Image content here]", "A proper full transcription. " * 8)
    VlmRead(client=fake).run(doc, cfg)

    assert fake.i == 2                                 # primary + escalation
    assert page.read_model == "big-model"             # provenance = the escalated model
    assert "proper full transcription" in page.vlm_reading.lower()


def test_refusal_detection():
    from fusion_ocr.stages.vlm_read import _looks_like_refusal
    # empty / placeholder / refusal -> treated as no read
    assert _looks_like_refusal("", 500)
    assert _looks_like_refusal("[Image content here]", 500)
    assert _looks_like_refusal("I'm unable to read this image.", 500)
    # far shorter than what OCR found -> discard
    assert _looks_like_refusal("ก", 500)
    # a genuine full reading -> kept
    assert not _looks_like_refusal("A" * 400, 500)
    assert not _looks_like_refusal("short but no OCR to beat", 0)


def test_born_digital_skips_vlm(tmp_path):
    pdf = tmp_path / "born.pdf"
    d = fitz.open(); d.new_page().insert_text((72, 72), "hi"); d.save(str(pdf)); d.close()
    doc = Document(source_path=str(pdf), sha256="x")
    doc.pages = [Page(index=0, needs_ocr=False)]
    fake = _FakeVLM("should not be called")
    VlmRead(client=fake).run(doc, config_mod.Config())
    assert fake.calls == 0


class _AirgapClient:
    def read(self, image_png, prompt, **opts):
        from fusion_ocr.config import AirgapError
        raise AirgapError("airgap: outbound connection refused")


def test_airgap_refusal_fails_loud_not_silent_det_text(tmp_path):
    # a sealed tier pointed at a remote endpoint must surface, not quietly fall back to
    # det_text (which would hide that the reader was unreachable)
    from fusion_ocr.config import AirgapError
    pdf = tmp_path / "scan.pdf"
    d = fitz.open(); pg = d.new_page()
    pg.insert_image(pg.rect, pixmap=fitz.open().new_page().get_pixmap(dpi=72))
    d.save(str(pdf)); d.close()
    doc = Document(source_path=str(pdf), sha256="x")
    doc.pages = [Page(index=0, needs_ocr=True, width=612, height=792)]
    with pytest.raises(AirgapError):
        VlmRead(client=_AirgapClient()).run(doc, config_mod.Config())
