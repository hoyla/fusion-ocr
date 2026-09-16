"""Eval work directories: run-scoped, under eval_out/_work (never /tmp), removed after the
run unless kept — the review-03 "confidential text in /tmp" fix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusion_ocr.eval import workdir as wd


def test_default_workdir_is_under_work_root_and_removed_after(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "WORK_ROOT", tmp_path / "eval_out" / "_work")
    with wd.workdir("harness") as work:
        assert work.parent == tmp_path / "eval_out" / "_work"
        assert work.name.startswith("harness-")
        (work / "out" / "j").mkdir(parents=True)
        (work / "out" / "j" / "doc.json").write_text("the recovered text")
    assert not work.exists()                       # gone with the run, contents included


def test_workdir_is_removed_even_when_the_run_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "WORK_ROOT", tmp_path / "w")
    with pytest.raises(RuntimeError):
        with wd.workdir("labels") as work:
            (work / "x.pdf").write_bytes(b"%PDF")
            raise RuntimeError("reader down")
    assert not work.exists()


def test_keep_retains_the_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "WORK_ROOT", tmp_path / "w")
    with wd.workdir("labels", keep=True) as work:
        (work / "x").write_text("x")
    assert (work / "x").exists()                   # --keep-work: left for inspection


def test_caller_supplied_directory_is_never_removed(tmp_path):
    mine = tmp_path / "mine"
    mine.mkdir()
    with wd.workdir("harness", given=mine) as work:
        assert work == mine
        (work / "x").write_text("x")
    assert (mine / "x").exists()                   # the caller's to manage (tests pass tmp_path)


def test_new_workdirs_are_unique_within_a_second(tmp_path):
    a = wd.new_workdir("p", root=tmp_path)
    b = wd.new_workdir("p", root=tmp_path)
    assert a != b and a.is_dir() and b.is_dir()


def test_labels_runner_cleans_up_by_default(tmp_path, monkeypatch):
    """End to end on the scored path (process() stubbed): with no tmp_root the run works
    under WORK_ROOT, writes its output there, and nothing is left behind afterwards."""
    from fusion_ocr import config as config_mod
    from fusion_ocr.eval import labels as labels_mod
    import fusion_ocr.pipeline as pipeline_mod

    monkeypatch.setattr(wd, "WORK_ROOT", tmp_path / "eval_out" / "_work")
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    manifest = tmp_path / "labelset.json"
    manifest.write_text(json.dumps({"labels": [
        {"id": "a", "pdf": "samples/x.pdf", "page": 0, "transcript": "a.txt"}]}))

    seen: dict = {}

    class _Page:
        has_text_layer, needs_ocr = False, True

    class _Doc:
        pages, artifacts = [_Page()], {}

    def _process(page_pdf, cfg, **kw):
        out = Path(cfg.out_dir)
        assert out.parent.parent == tmp_path / "eval_out" / "_work"   # under the run dir
        assert Path(page_pdf).parent == out.parent                       # page pdf beside it
        out.mkdir(parents=True)
        (out / "doc.json").write_text("hello world")                     # the sensitive bit
        seen["out"] = out
        return _Doc()

    monkeypatch.setattr(labels_mod, "_extract_pages", lambda *a, **k: None)
    monkeypatch.setattr(labels_mod, "recovered_text", lambda page: "hello world")
    monkeypatch.setattr(pipeline_mod, "process", _process)

    [res] = labels_mod.evaluate_labelset(manifest, config_mod.Config())
    assert res["status"] == "scored" and res["word_recall"] == 1.0
    assert not seen["out"].exists()                                      # cleaned up
    assert list((tmp_path / "eval_out" / "_work").iterdir()) == []       # no run dir left
    assert not list(Path("/tmp").glob("fusion_label_eval_*"))            # and never /tmp
