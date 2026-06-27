"""Document (de)serialization — the resume contract.

from_json is now schema-driven (walks dataclass fields), so a newly-added field
round-trips automatically instead of being silently dropped by a hand-written mapping
someone forgot to update. These tests are the guard: a full-field round-trip would
break the moment a field stopped surviving resume.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from fusion_ocr.models import Box, Document, Page, Region, Segment


def _full_document() -> Document:
    """Every field set to a NON-default value, so a dropped field shows up as a diff."""
    box = Box(points=[(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)])
    seg = Segment(id="s1", page=0, box=box, best_text="hi", source="fused",
                  det_text="h1", det_conf=0.9, vlm_text="hi", read_by="model-x",
                  superseded=True, translations={"en": "hi", "th": "สวัสดี"})
    reg = Region(box=box, kind="table", reading_order=2, source="ocr",
                 table_html="<table><tr><td>a</td></tr></table>", cells=[box, box],
                 table_vlm="| a |\n|---|", table_read_by="model-x",
                 table_engine="find_tables")
    page = Page(index=3, width=600.0, height=800.0, has_text_layer=True, needs_ocr=False,
                rotation=90, script="thai", read_model="model-x", image_ref="p3.png",
                vlm_reading="the reading", regions=[reg], segments=[seg])
    return Document(source_path="a.pdf", sha256="abc123", languages=["en", "th"],
                    pages=[page], artifacts={"overlay": "o.pdf"},
                    stage_completed="render")


def test_round_trip_preserves_every_field():
    doc = _full_document()
    back = Document.from_json(doc.to_json())
    # deep structural equality — any dropped/garbled field surfaces here
    assert asdict(back) == asdict(doc)


def test_box_points_round_trip_as_tuples():
    doc = _full_document()
    back = Document.from_json(doc.to_json())
    pts = back.pages[0].regions[0].box.points
    assert pts == [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)]
    assert all(isinstance(p, tuple) for p in pts)        # not lists, so bbox math works


def test_missing_newer_field_falls_back_to_default():
    # an OLDER doc.json written before a field existed -> default applies, no crash
    d = json.loads(_full_document().to_json())
    del d["pages"][0]["regions"][0]["table_engine"]
    del d["pages"][0]["segments"][0]["translations"]
    back = Document.from_json(json.dumps(d))
    assert back.pages[0].regions[0].table_engine == ""   # dataclass default
    assert back.pages[0].segments[0].translations == {}


def test_unknown_key_is_ignored():
    # a removed/renamed field still present in an old doc.json must not break load
    d = json.loads(_full_document().to_json())
    d["pages"][0]["regions"][0]["legacy_field"] = "ignore me"
    back = Document.from_json(json.dumps(d))
    assert back.pages[0].regions[0].kind == "table"
