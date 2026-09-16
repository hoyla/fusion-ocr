"""The eval CLI (`python -m fusion_ocr.eval`): argument routing, engine flags reaching the
config, the scorecard/TODO output and the usage error — with the runners stubbed. The entry
point people actually type had no test (and 0% coverage) before this."""

from __future__ import annotations

import sys

import pytest

from fusion_ocr import config as config_mod
from fusion_ocr.eval import __main__ as cli
from fusion_ocr.eval.metrics import score


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    c = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out", airgap=False)
    monkeypatch.setattr(cli.config_mod, "load", lambda path="config.toml": c)
    return c


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["fusion-ocr-eval", *argv])
    cli.main()


def _scored(id_, ref="hello brave new world", hyp="hello brave new world", **extra):
    return {"id": id_, "status": "scored", **score(ref, hyp), **extra}


def test_no_input_is_a_usage_error(monkeypatch, cfg, capsys):
    monkeypatch.setattr(sys, "argv", ["fusion-ocr-eval"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "give born-digital PDFs, --labels, or --dataset" in capsys.readouterr().err


def test_labels_mode_scorecard_flags_and_todo(monkeypatch, cfg, capsys):
    import fusion_ocr.eval.labels as labels_mod
    seen = {}

    def fake(manifest, c, no_vlm=False, keep_work=False):
        seen.update(manifest=manifest, no_vlm=no_vlm, keep_work=keep_work, cfg=c)
        return [_scored("mandelson-note", searchable=score("hello brave new world", "hello world"),
                        searchable_via="overlay"),
                {"id": "thai-scan", "status": "unlabelled"}]
    monkeypatch.setattr(labels_mod, "evaluate_labelset", fake)

    _run(monkeypatch, "--labels", "eval_labels/labelset.json", "--no-vlm", "--rapidocr", "--keep-work")
    out = capsys.readouterr().out
    assert seen == {"manifest": "eval_labels/labelset.json", "no_vlm": True, "keep_work": True, "cfg": cfg}
    assert cfg.prefer_rapidocr is True                       # the flag reached the config ...
    assert "engine: RapidOCR" in out                         # ... and the banner says so (no VLM)
    assert "mandelson-note" in out and "AGGREGATE" in out and "(1 pages)" in out
    assert " ovl" in out and "sCER" in out                   # searchable columns present
    assert "1 page(s) not yet labelled" in out and "thai-scan" in out
    assert "work dir kept" in out


def test_born_digital_mode_pages_dpi_and_apple_vision(monkeypatch, cfg, capsys):
    import fusion_ocr.eval.harness as harness_mod
    seen = {}

    def fake(pdfs, c, pages=None, dpi=200, no_vlm=False, keep_work=False):
        seen.update(pdfs=[str(p) for p in pdfs], pages=pages, dpi=dpi, no_vlm=no_vlm)
        return [{"pdf": str(pdfs[0]), "page": 0, **score("a b c d e f", "a b c d e f")}]
    monkeypatch.setattr(harness_mod, "evaluate", fake)

    _run(monkeypatch, "report.pdf", "annex.pdf", "--pages", "0,2", "--dpi", "150", "--apple-vision")
    out = capsys.readouterr().out
    assert seen == {"pdfs": ["report.pdf", "annex.pdf"], "pages": [0, 2], "dpi": 150, "no_vlm": False}
    assert cfg.prefer_apple_vision is True
    assert "engine: Apple Vision + VLM" in out               # VLM still on without --no-vlm
    assert "report:0" in out                                 # page tag column


def test_born_digital_mode_with_nothing_scorable(monkeypatch, cfg, capsys):
    import fusion_ocr.eval.harness as harness_mod
    monkeypatch.setattr(harness_mod, "evaluate", lambda *a, **k: [])
    _run(monkeypatch, "cover-only.pdf")
    assert "no scorable pages" in capsys.readouterr().out


def test_dataset_mode_passes_split_and_limit(monkeypatch, cfg, capsys):
    import fusion_ocr.eval.datasets as datasets_mod
    seen = {}

    def fake(source, c, split="test", limit=20, no_vlm=False, keep_work=False):
        seen.update(source=source, split=split, limit=limit, no_vlm=no_vlm)
        return [_scored("X51005", source=source)]
    monkeypatch.setattr(datasets_mod, "evaluate_dataset", fake)

    _run(monkeypatch, "--dataset", "sroie", "--split", "val", "--limit", "3", "--no-vlm")
    out = capsys.readouterr().out
    assert seen == {"source": "sroie", "split": "val", "limit": 3, "no_vlm": True}
    assert "engine: PaddleOCR" in out and "X51005" in out and "sroie" in out


def test_placement_mode_prints_strict_band_and_plain(monkeypatch, cfg, capsys):
    import fusion_ocr.eval.datasets as datasets_mod

    def fake(source, c, split="test", limit=20, no_vlm=False, keep_work=False):
        return [{"id": "f1", "source": source, "placed": 8, "plain": 9, "total": 10, "band_placed": 9},
                {"id": "f2", "source": source, "placed": 4, "plain": 5, "total": 10, "band_placed": 5}]
    monkeypatch.setattr(datasets_mod, "evaluate_placement", fake)

    _run(monkeypatch, "--dataset", "funsd", "--placement")
    out = capsys.readouterr().out
    assert "PLACEMENT (funsd/test, 2 pages, 20 GT words)" in out
    assert "strict 0.600" in out and "band 0.700" in out and "plain 0.700" in out


def test_placement_mode_with_nothing_scorable(monkeypatch, cfg, capsys):
    import fusion_ocr.eval.datasets as datasets_mod
    monkeypatch.setattr(datasets_mod, "evaluate_placement", lambda *a, **k: [])
    _run(monkeypatch, "--dataset", "funsd", "--placement", "--split", "val")
    assert "no scorable items in funsd/val" in capsys.readouterr().out
