"""Hand-labelled eval — manifest loading, the TODO branch, and a scored entry.

These avoid the real pipeline: the unlabelled branch returns before process() is called,
and the scored test stubs process()/page extraction so no models or PDFs are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from fusion_ocr import config as config_mod
from fusion_ocr.eval import labels as labels_mod
from fusion_ocr.eval.labels import evaluate_labelset, load_labelset


def _write_manifest(tmp_path: Path, entries: list[dict]) -> Path:
    manifest = tmp_path / "labelset.json"
    manifest.write_text(json.dumps({"labels": entries}), encoding="utf-8")
    return manifest


def test_load_labelset_resolves_transcript_beside_manifest(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"id": "a", "pdf": "samples/x.pdf", "page": 3, "transcript": "a.txt", "note": "hi"},
    ])
    [lab] = load_labelset(manifest)
    assert lab.id == "a"
    assert lab.pages == [3]                           # single `page` -> one-element span
    assert lab.pdf == Path("samples/x.pdf")          # kept relative (resolved at run time)
    assert lab.transcript == (tmp_path / "a.txt").resolve()   # resolved beside the manifest
    assert lab.note == "hi"


def test_load_labelset_multipage_span(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"id": "letter", "pdf": "samples/x.pdf", "pages": [183, 184], "transcript": "l.txt"},
    ])
    [lab] = load_labelset(manifest)
    assert lab.pages == [183, 184]


def test_load_labelset_render_flag_defaults_false_and_parses(tmp_path):
    manifest = _write_manifest(tmp_path, [
        {"id": "scan", "pdf": "samples/x.pdf", "page": 0, "transcript": "s.txt"},
        {"id": "born", "pdf": "samples/y.pdf", "page": 1, "transcript": "b.txt", "render": True},
    ])
    scan, born = load_labelset(manifest)
    assert scan.render is False        # absent -> default
    assert born.render is True


def test_render_label_forces_image_only_path_not_extract(tmp_path, monkeypatch):
    """A render:true label must render the page to an image-only PDF (force OCR), never the
    as-is _extract_pages copy (which would keep the born-digital text layer and skip OCR)."""
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [
        {"id": "a", "pdf": "samples/x.pdf", "page": 4, "transcript": "a.txt", "render": True},
    ])
    called = {"render": False, "extract": False}
    monkeypatch.setattr(labels_mod, "_render_pages_image_only",
                        lambda *a, **k: called.__setitem__("render", True))
    monkeypatch.setattr(labels_mod, "_extract_pages",
                        lambda *a, **k: called.__setitem__("extract", True))
    monkeypatch.setattr(labels_mod, "recovered_text", lambda page: "hello world")
    monkeypatch.setattr(labels_mod, "_pdf_text", lambda p: "")
    import fusion_ocr.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "process", lambda *a, **k: _doc([_Page()]))

    [res] = evaluate_labelset(manifest, config_mod.Config(), tmp_root=tmp_path / "tmp")
    assert called == {"render": True, "extract": False}
    assert res["status"] == "scored"


def test_empty_transcript_is_reported_unlabelled_without_running_pipeline(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("   \n", encoding="utf-8")   # whitespace only = not done
    manifest = _write_manifest(tmp_path, [
        {"id": "a", "pdf": "samples/x.pdf", "page": 0, "transcript": "a.txt"},
    ])

    # If the pipeline is touched for an unlabelled page, fail loudly.
    import fusion_ocr.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "process",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))

    [res] = evaluate_labelset(manifest, config_mod.Config(), tmp_root=tmp_path / "tmp")
    assert res["status"] == "unlabelled"
    assert "cer" not in res


class _Page:
    """Minimal stand-in for models.Page — only the fields evaluate_labelset reads."""
    def __init__(self, has_text_layer=False, needs_ocr=True):
        self.has_text_layer = has_text_layer
        self.needs_ocr = needs_ocr


def _doc(pages, artifacts=None):
    class _Doc:
        pass
    d = _Doc()
    d.pages = pages
    d.artifacts = artifacts or {}
    return d


def test_scored_entry_compares_transcript_to_recovered_text(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [
        {"id": "a", "pdf": "samples/x.pdf", "page": 0, "transcript": "a.txt"},
    ])

    monkeypatch.setattr(labels_mod, "_extract_pages", lambda *a, **k: None)   # skip real PDF I/O

    import fusion_ocr.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "process", lambda *a, **k: _doc([_Page()]))
    monkeypatch.setattr(labels_mod, "recovered_text", lambda page: "hello world")

    [res] = evaluate_labelset(manifest, config_mod.Config(), tmp_root=tmp_path / "tmp")
    assert res["status"] == "scored"
    assert res["id"] == "a"
    assert res["cer"] == 0.0                  # transcript == recovered text
    assert res["word_recall"] == 1.0
    assert res["word_precision"] == 1.0


def test_searchable_via_overlay_when_one_was_built(tmp_path, monkeypatch):
    """An OCR page's overlay.pdf text is the searchable side, scored vs the reading view."""
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [
        {"id": "a", "pdf": "samples/x.pdf", "page": 0, "transcript": "a.txt"},
    ])
    monkeypatch.setattr(labels_mod, "_extract_pages", lambda *a, **k: None)
    monkeypatch.setattr(labels_mod, "recovered_text", lambda page: "hello world")
    doc = _doc([_Page(needs_ocr=True)], {"overlay_pdf": "/x/overlay.pdf"})
    import fusion_ocr.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "process", lambda *a, **k: doc)
    monkeypatch.setattr(labels_mod, "_pdf_text", lambda p: "hello")  # overlay drops a word

    [res] = evaluate_labelset(manifest, config_mod.Config(), tmp_root=tmp_path / "tmp")
    assert res["searchable_via"] == "overlay"
    assert res["word_recall"] == 1.0                    # reading recovered both words
    assert res["searchable"]["word_recall"] == 0.5      # overlay only has one searchable


def test_searchable_falls_back_to_source_text_layer_when_no_overlay(tmp_path, monkeypatch):
    """No overlay built (born-digital, or mixed where the text layer wins) -> searchability
    comes from the source PDF's text layer, not a 0."""
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [
        {"id": "a", "pdf": "samples/x.pdf", "page": 0, "transcript": "a.txt"},
    ])
    monkeypatch.setattr(labels_mod, "_extract_pages", lambda *a, **k: None)
    monkeypatch.setattr(labels_mod, "recovered_text", lambda page: "hello world")
    doc = _doc([_Page(has_text_layer=True, needs_ocr=False)])    # no overlay_pdf artifact
    import fusion_ocr.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "process", lambda *a, **k: doc)
    # the source page PDF still carries a searchable text layer
    monkeypatch.setattr(labels_mod, "_pdf_text", lambda p: "hello world")

    [res] = evaluate_labelset(manifest, config_mod.Config(), tmp_root=tmp_path / "tmp")
    assert res["searchable_via"] == "text_layer"
    assert res["searchable"]["word_recall"] == 1.0      # text layer carries it
    assert res["word_recall"] == 1.0


def test_searchable_none_when_nothing_findable(tmp_path, monkeypatch):
    """OCR page with no overlay and no text layer -> a genuine miss, scored as recall 0."""
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [
        {"id": "a", "pdf": "samples/x.pdf", "page": 0, "transcript": "a.txt"},
    ])
    monkeypatch.setattr(labels_mod, "_extract_pages", lambda *a, **k: None)
    monkeypatch.setattr(labels_mod, "recovered_text", lambda page: "hello world")
    doc = _doc([_Page(needs_ocr=True)])                          # no overlay, empty source
    import fusion_ocr.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "process", lambda *a, **k: doc)
    monkeypatch.setattr(labels_mod, "_pdf_text", lambda p: "")

    [res] = evaluate_labelset(manifest, config_mod.Config(), tmp_root=tmp_path / "tmp")
    assert res["searchable_via"] == "none"
    assert res["searchable"]["word_recall"] == 0.0


def test_deterministic_pipeline_drops_only_the_vlm_stages():
    from fusion_ocr.pipeline import DEFAULT_PIPELINE, deterministic_pipeline
    names = [s.name for s in deterministic_pipeline()]
    assert "vlm_read" not in names and "table_read" not in names
    assert names == [s.name for s in DEFAULT_PIPELINE
                     if s.name not in ("vlm_read", "table_read")]


def test_no_vlm_runs_the_deterministic_pipeline(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    manifest = _write_manifest(tmp_path, [
        {"id": "a", "pdf": "samples/x.pdf", "page": 0, "transcript": "a.txt"},
    ])
    monkeypatch.setattr(labels_mod, "_extract_pages", lambda *a, **k: None)
    monkeypatch.setattr(labels_mod, "recovered_text", lambda page: "hello world")

    captured = {}

    import fusion_ocr.pipeline as pipeline_mod

    def fake_process(pdf, cfg, pipeline=None, **k):
        captured["names"] = [s.name for s in pipeline] if pipeline is not None else None
        return _doc([_Page()])
    monkeypatch.setattr(pipeline_mod, "process", fake_process)

    evaluate_labelset(manifest, config_mod.Config(), tmp_root=tmp_path / "t", no_vlm=True)
    assert captured["names"] is not None
    assert "vlm_read" not in captured["names"] and "table_read" not in captured["names"]
