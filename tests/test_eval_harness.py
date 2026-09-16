"""The born-digital harness (`harness.evaluate_pdf` / `evaluate`): each selected page is
rendered to an image-only PDF (the pipeline must OCR it), scored against the page's own text
layer, out-of-range and near-empty pages skipped. process() is stubbed to hand back the
text layer as the reading, so the scores are exact and no model runs."""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="needs PyMuPDF")

from fusion_ocr import config as config_mod  # noqa: E402
from fusion_ocr.eval import harness  # noqa: E402
from fusion_ocr.models import Document, Page  # noqa: E402

_LINE = "The quick brown fox jumps over the lazy dog, again and again. "


def _born_digital(path, texts):
    d = fitz.open()
    for t in texts:
        d.new_page().insert_text((72, 72), t)
    d.save(str(path))
    d.close()
    return path


@pytest.fixture
def fake_process(monkeypatch):
    """Stand-in for pipeline.process: asserts the harness's premise (the page it gets is
    image-only) and returns the source page's text layer as the reading."""
    import fusion_ocr.pipeline as pipeline_mod
    calls = []

    def _process(img_pdf, cfg, pipeline=None, **kw):
        with fitz.open(img_pdf) as d:
            assert d.page_count == 1 and d[0].get_text().strip() == ""   # no text layer to cheat from
        _, idx = Path(img_pdf).stem.rsplit("_p", 1)     # "<src stem>_p<index>"
        calls.append({"img": Path(img_pdf), "out": Path(cfg.out_dir),
                      "pipeline": [s.name for s in pipeline] if pipeline else None, "page": int(idx)})
        page = Page(index=0)
        page.vlm_reading = harness.page_text_layer(_process.src, int(idx))
        return Document(source_path=str(img_pdf), sha256="x", pages=[page])
    monkeypatch.setattr(pipeline_mod, "process", _process)
    _process.calls = calls
    return _process


def test_each_page_is_rendered_to_a_scan_and_scored_exactly(tmp_path, fake_process):
    src = _born_digital(tmp_path / "report.pdf", [_LINE * 2, _LINE * 3])
    fake_process.src = src
    cfg = config_mod.Config(out_dir=tmp_path / "unused", airgap=False)
    results = harness.evaluate_pdf(src, cfg, pages=[0, 1, 7], tmp_root=tmp_path / "work")
    assert [r["page"] for r in results] == [0, 1]             # page 7 doesn't exist -> skipped
    assert all(r["pdf"] == str(src) for r in results)
    assert all(r["cer"] == 0.0 and r["word_recall"] == 1.0 for r in results)   # reading == layer
    assert [c["page"] for c in fake_process.calls] == [0, 1]
    assert all(c["img"].parent == tmp_path / "work" for c in fake_process.calls)   # the given work dir
    assert all(c["out"] == tmp_path / "work" / "out" for c in fake_process.calls)  # never cfg.out_dir
    assert all(c["pipeline"] is None for c in fake_process.calls)                  # default pipeline


def test_no_vlm_hands_the_deterministic_pipeline_to_process(tmp_path, fake_process):
    src = _born_digital(tmp_path / "r.pdf", [_LINE * 2])
    fake_process.src = src
    harness.evaluate_pdf(src, config_mod.Config(airgap=False), no_vlm=True, tmp_root=tmp_path / "w")
    names = fake_process.calls[0]["pipeline"]
    assert names and "vlm_read" not in names and "table_read" not in names and "ocr_det" in names


def test_pages_with_too_little_text_are_skipped(tmp_path, fake_process):
    src = _born_digital(tmp_path / "r.pdf", ["cover", _LINE * 2])     # a cover page: < 50 chars
    fake_process.src = src
    results = harness.evaluate_pdf(src, config_mod.Config(airgap=False), tmp_root=tmp_path / "w")
    assert [r["page"] for r in results] == [1]


def test_evaluate_concatenates_across_pdfs_and_pages_default_to_all(tmp_path, fake_process, monkeypatch):
    a = _born_digital(tmp_path / "a.pdf", [_LINE * 2, _LINE * 2])
    b = _born_digital(tmp_path / "b.pdf", [_LINE * 2])
    import fusion_ocr.pipeline as pipeline_mod
    stub = pipeline_mod.process                      # the fixture's stub, keyed on one source

    def per_source(img_pdf, cfg, pipeline=None, **kw):   # point it at whichever PDF this page is from
        fake_process.src = a if Path(img_pdf).stem.startswith("a_") else b
        return stub(img_pdf, cfg, pipeline=pipeline, **kw)
    monkeypatch.setattr(pipeline_mod, "process", per_source)
    results = harness.evaluate([a, b], config_mod.Config(airgap=False), tmp_root=tmp_path / "w")
    assert [(Path(r["pdf"]).name, r["page"]) for r in results] == [("a.pdf", 0), ("a.pdf", 1), ("b.pdf", 0)]


def test_default_work_dir_is_run_scoped_and_removed(tmp_path, fake_process, monkeypatch):
    from fusion_ocr.eval import workdir as wd
    monkeypatch.setattr(wd, "WORK_ROOT", tmp_path / "eval_out" / "_work")
    src = _born_digital(tmp_path / "r.pdf", [_LINE * 2])
    fake_process.src = src
    harness.evaluate_pdf(src, config_mod.Config(airgap=False))
    work = fake_process.calls[0]["img"].parent
    assert work.parent == tmp_path / "eval_out" / "_work" and not work.exists()   # gone with the run
