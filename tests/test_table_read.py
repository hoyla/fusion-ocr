"""Table-class routing — focused VLM read of scanned table regions.

Mock client, no model/server needed. Covers: the stage routes table regions to the
table reader (right prompt + route-aware model + provenance), skips the classes it
should (born-digital, rotated, disabled, already-read), and render places the focused
read at the table's position instead of the monolithic page reading.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz", reason="needs PyMuPDF")
import fitz  # noqa: E402

from fusion_ocr.config import Config  # noqa: E402
from fusion_ocr.models import Box, Document, Page, Region, Segment  # noqa: E402
from fusion_ocr.stages.render import _page_markdown  # noqa: E402
from fusion_ocr.stages.table_read import TableRead  # noqa: E402
from fusion_ocr.vlm.prompts import TABLE, TYPHOON_OCR, select_table_prompt  # noqa: E402

_MD = "| H1 | H2 |\n|---|---|\n| a | b |"


def _box(x0, y0, x1, y1):
    return Box(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _table_region(**kw):
    return Region(box=_box(50, 50, 500, 400), kind="table", **kw)


def _blank_pdf(tmp_path):
    pdf = tmp_path / "scan.pdf"
    d = fitz.open(); d.new_page(width=612, height=792)
    d.save(str(pdf)); d.close()
    return pdf


class _FakeVLM:
    def __init__(self, text=_MD):
        self.text = text
        self.calls = 0
        self.prompts: list[str] = []

    def read(self, image_png, prompt, **opts):
        self.calls += 1
        self.prompts.append(prompt)
        return self.text


def _scanned_doc(tmp_path, **page_kw):
    doc = Document(source_path=str(_blank_pdf(tmp_path)), sha256="x")
    page = Page(index=0, needs_ocr=True, width=612, height=792, **page_kw)
    page.regions = [_table_region()]
    doc.pages = [page]
    return doc


# ---- the stage routes table regions ---------------------------------------

def test_reads_table_region_and_records_provenance(tmp_path):
    doc = _scanned_doc(tmp_path, script="latin")
    fake = _FakeVLM()
    TableRead(client=fake).run(doc, Config())

    region = doc.pages[0].regions[0]
    assert fake.calls == 1
    assert region.table_vlm == _MD                      # clean content stored
    assert region.table_read_by == Config().vlm.model    # provenance = generalist
    assert fake.prompts[0] == TABLE                       # generalist -> table prompt
    assert region.table_html == ""                       # deterministic grid untouched


def test_thai_table_routes_to_typhoon_with_its_prompt(tmp_path):
    doc = _scanned_doc(tmp_path, script="thai")
    fake = _FakeVLM("<table><tr><td>ก</td></tr></table>")
    TableRead(client=fake).run(doc, Config())

    region = doc.pages[0].regions[0]
    assert region.table_read_by == "mlx-community/typhoon-ocr1.5-2b-8bit"
    assert fake.prompts[0] == TYPHOON_OCR                 # specialist prompt


# ---- the classes it must skip ---------------------------------------------

def test_skips_born_digital(tmp_path):
    doc = _scanned_doc(tmp_path, script="latin")
    doc.pages[0].needs_ocr = False                        # born-digital -> text layer
    fake = _FakeVLM()
    TableRead(client=fake).run(doc, Config())
    assert fake.calls == 0


def test_skips_rotated(tmp_path):
    doc = _scanned_doc(tmp_path, script="latin", rotation=90)
    fake = _FakeVLM()
    TableRead(client=fake).run(doc, Config())
    assert fake.calls == 0


def test_disabled_by_config(tmp_path):
    doc = _scanned_doc(tmp_path, script="latin")
    cfg = Config(); cfg.table_vlm_read = False
    fake = _FakeVLM()
    TableRead(client=fake).run(doc, cfg)
    assert fake.calls == 0


def test_idempotent_skips_already_read_region(tmp_path):
    doc = _scanned_doc(tmp_path, script="latin")
    doc.pages[0].regions = [_table_region(table_vlm="already", table_read_by="prev")]
    fake = _FakeVLM()
    TableRead(client=fake).run(doc, Config())
    assert fake.calls == 0
    assert doc.pages[0].regions[0].table_vlm == "already"


def test_no_table_region_is_noop(tmp_path):
    doc = _scanned_doc(tmp_path, script="latin")
    doc.pages[0].regions = [Region(box=_box(0, 0, 600, 90), kind="paragraph")]
    fake = _FakeVLM()
    TableRead(client=fake).run(doc, Config())
    assert fake.calls == 0


# ---- prompt selection -----------------------------------------------------

def test_select_table_prompt():
    assert select_table_prompt("mlx-community/Qwen3-VL-8B-Instruct-4bit") == TABLE
    assert select_table_prompt("mlx-community/typhoon-ocr1.5-2b-8bit") == TYPHOON_OCR


# ---- render places the focused read at the table position -----------------

def _seg(id, best, x0, y0, x1, y1):
    return Segment(id=id, page=0, box=_box(x0, y0, x1, y1), best_text=best, source="fused")


def test_render_prefers_focused_table_over_page_reading():
    # page got a full reading AND a focused table read -> use the block path so the
    # clean table lands at its position; the table's loose segment is suppressed.
    page = Page(index=0, width=600, height=800)
    page.vlm_reading = "Intro heading\nstray cell text inline"
    page.regions = [
        Region(box=_box(0, 0, 600, 90), kind="paragraph", reading_order=0),
        Region(box=_box(0, 100, 600, 300), kind="table", reading_order=1,
               table_vlm=_MD, table_read_by="m"),
    ]
    page.segments = [
        _seg("p", "Intro heading", 10, 10, 500, 40),       # prose, above the table
        _seg("t", "stray cell text", 50, 150, 200, 180),   # inside the table region
    ]
    md = _page_markdown(page)
    assert _MD in md                                       # focused table read placed
    assert "Intro heading" in md                           # prose kept
    assert "stray cell text" not in md                     # table segment suppressed
    assert md.index("Intro heading") < md.index("| H1")    # at its reading position


def test_render_keeps_flat_reading_when_no_focused_table():
    # full page reading, no focused table -> keep the reading whole (status quo).
    page = Page(index=0, width=600, height=800)
    page.vlm_reading = "Full page reading with the table inside it."
    page.regions = [Region(box=_box(0, 100, 600, 300), kind="table", reading_order=1,
                           table_html="<table><tr><td>old</td></tr></table>")]
    md = _page_markdown(page)
    assert md == "Full page reading with the table inside it."
