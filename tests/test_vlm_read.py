"""VLM read stage — mock client, no Ollama/model needed.

vlm_read now only produces page.vlm_reading; aligning it onto boxes is fusion's job
(see test_fusion).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz", reason="needs PyMuPDF")
import fitz  # noqa: E402

from fusion_ocr import config as config_mod  # noqa: E402
from fusion_ocr.models import Box, Document, Page, Segment  # noqa: E402
from fusion_ocr.stages.vlm_read import VlmRead  # noqa: E402


def _ink_seg(text="detected ink", conf=0.9):
    """A detected text box, so the page isn't treated as blank by the no-ink short-circuit."""
    return Segment(id="a", page=0,
                   box=Box(points=[(50, 100), (300, 100), (300, 120), (50, 120)]),
                   det_text=text, det_conf=conf, source="paddle")


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
    page = Page(index=0, needs_ocr=True, width=612, height=792)
    page.segments = [_ink_seg()]                      # a page with detected ink (not blank)
    doc.pages = [page]

    fake = _FakeVLM("First real line\nSecond real line")
    VlmRead(client=fake).run(doc, config_mod.Config())

    assert fake.calls == 1
    assert doc.pages[0].vlm_reading == "First real line\nSecond real line"


def test_blank_page_skips_vlm_no_hallucination(tmp_path):
    # No detected ink -> (near-)blank page. The VLM must NOT be called (it would hallucinate
    # on an empty image); the reading stays empty.
    pdf = tmp_path / "blank.pdf"
    d = fitz.open(); pg = d.new_page()
    pg.insert_image(pg.rect, pixmap=fitz.open().new_page().get_pixmap(dpi=72))
    d.save(str(pdf)); d.close()

    doc = Document(source_path=str(pdf), sha256="x")
    doc.pages = [Page(index=0, needs_ocr=True, width=612, height=792)]   # no segments = blank
    fake = _FakeVLM("$$\\frac{1}{\\sqrt{2}}$$")        # what it hallucinates if asked
    VlmRead(client=fake).run(doc, config_mod.Config())

    assert fake.calls == 0
    assert doc.pages[0].vlm_reading == ""


def test_degenerate_repetition_is_discarded():
    from fusion_ocr.stages.vlm_read import _is_degenerate_repetition, _looks_like_refusal
    loop = "[illegible] " * 200                         # the measured failure mode
    assert _is_degenerate_repetition(loop)
    assert _looks_like_refusal(loop, 0)                  # so fusion falls back to det_text
    # ordinary varied prose is never flagged, even when long
    prose = " ".join(f"word{i}" for i in range(200))
    assert not _is_degenerate_repetition(prose)
    assert not _is_degenerate_repetition("short repeated repeated")   # too short to judge


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
    page = Page(index=0, needs_ocr=True, width=612, height=792)
    page.segments = [_ink_seg()]                      # detected ink, so the VLM is reached
    doc.pages = [page]
    with pytest.raises(AirgapError):
        VlmRead(client=_AirgapClient()).run(doc, config_mod.Config())


class _FailingVLM:
    """A reader that ERRORS (server down / wedged / timeout) — not a refusal response."""
    def read(self, image_png, prompt, **opts):
        raise ConnectionError("connection refused")


def _scan_doc(tmp_path):
    pdf = tmp_path / "scan.pdf"
    d = fitz.open(); pg = d.new_page()
    pg.insert_image(pg.rect, pixmap=fitz.open().new_page().get_pixmap(dpi=72))
    d.save(str(pdf)); d.close()
    doc = Document(source_path=str(pdf), sha256="x")
    page = Page(index=0, needs_ocr=True, width=612, height=792)
    page.segments = [_ink_seg()]                      # detected ink, so the VLM is reached
    doc.pages = [page]
    return doc, page


def test_reader_failure_is_flagged_not_silent(tmp_path):
    # A reader FAILURE (server down/wedged) must set page.read_failed AND fall back to det_text —
    # the old behaviour silently returned "" here, hiding a corpus-wide degradation.
    doc, page = _scan_doc(tmp_path)
    VlmRead(client=_FailingVLM()).run(doc, config_mod.Config())
    assert page.read_failed is True                   # fail loud: visible in the artifact
    assert page.vlm_reading == ""                     # fusion still falls back to det_text
    assert page.read_model == ""


def test_legit_refusal_is_not_flagged_as_failure(tmp_path):
    # A refusal/empty RESPONSE (the model read but declined) is NOT a reader failure — no flag,
    # so a run of hard-but-real pages isn't mistaken for a dead reader.
    doc, page = _scan_doc(tmp_path)
    VlmRead(client=_FakeVLM("[Image content here]")).run(doc, config_mod.Config())
    assert page.read_failed is False
    assert page.vlm_reading == ""


def test_read_failed_defaults_false_and_round_trips(tmp_path):
    # provenance flag defaults False and survives the schema-driven serialiser
    doc, page = _scan_doc(tmp_path)
    VlmRead(client=_FakeVLM("A real reading. " * 8)).run(doc, config_mod.Config())
    assert page.read_failed is False
    assert Document.from_json(doc.to_json()).pages[0].read_failed is False


def test_preflight_reader_ok(monkeypatch):
    from fusion_ocr.vlm import openai_compat
    monkeypatch.setattr(openai_compat.OpenAICompatVLM, "probe", lambda self: None)
    ok, detail = openai_compat.preflight_reader(config_mod.Config())
    assert ok and "ready" in detail.lower()


def test_preflight_reader_reports_not_ready(monkeypatch):
    # a dead/wedged reader is surfaced as (False, why) — the caller warns/aborts instead of
    # silently degrading a whole batch to det_text
    from fusion_ocr.vlm import openai_compat

    def boom(self):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(openai_compat.OpenAICompatVLM, "probe", boom)
    ok, detail = openai_compat.preflight_reader(config_mod.Config())
    assert not ok
    assert "not ready" in detail.lower()
