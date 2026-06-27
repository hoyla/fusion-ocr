"""Reading-view provenance note — the honesty header on document.md.

The markdown is a reading aid; the overlay + segment index are the gated, provenanced
artifacts. The VLM caveat must appear only when a VLM actually read a page (born-digital
and Apple-Vision output is exact / ink-gated, not a model transcription)."""

from __future__ import annotations

from fusion_ocr.models import Document, Page
from fusion_ocr.stages.render import _provenance_note


def _doc(*read_models) -> Document:
    return Document(source_path="x", sha256="s",
                    pages=[Page(index=i, read_model=m) for i, m in enumerate(read_models)])


def test_note_present_when_a_vlm_read_a_page():
    note = _provenance_note(_doc("mlx-community/Qwen3-VL-8B-Instruct-4bit"))
    assert "vision-language model" in note
    assert "overlay.pdf" in note and "segment_index.json" in note


def test_no_note_for_apple_vision_or_born_digital():
    assert _provenance_note(_doc("apple_vision")) == ""   # exact on-device text
    assert _provenance_note(_doc("")) == ""               # born-digital / no read


def test_note_present_if_any_page_used_a_vlm():
    # mixed doc: one Apple-Vision page + one VLM page -> caveat still applies
    assert "vision-language model" in _provenance_note(_doc("apple_vision", "qwen"))
