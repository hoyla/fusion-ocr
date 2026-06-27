"""Resume / re-run contract. The cache is keyed on (content + recipe), so changing a
prompt/model/route reprocesses instead of silently no-op'ing; rerun_from re-runs from a
chosen stage reusing earlier ones (correct via per-stage snapshots); force ignores cache.

Fake stages (Document-in/out) record which ran and tag the doc, so we can assert exactly
what re-ran without any model."""

from __future__ import annotations

import pytest

from fusion_ocr.config import Config
from fusion_ocr.models import Document
from fusion_ocr.pipeline import process, recipe_fingerprint


class _Stage:
    def __init__(self, name: str, calls: list[str]):
        self.name = name
        self.calls = calls

    def run(self, doc: Document, cfg: Config) -> Document:
        self.calls.append(self.name)
        doc.languages.append(self.name)   # marker: this stage ran, in order
        return doc


def _pipe(calls):
    return [_Stage("a", calls), _Stage("b", calls), _Stage("c", calls)]


def _pdf(tmp_path):
    p = tmp_path / "in.pdf"
    p.write_bytes(b"%PDF-1.4 fake bytes")   # only hashed, never opened by fake stages
    return p


def _cfg(tmp_path):
    return Config(out_dir=tmp_path / "out")


# ---- recipe fingerprint ----------------------------------------------------

def test_recipe_fingerprint_sensitivity(tmp_path):
    calls: list[str] = []
    pipe = _pipe(calls)
    base = recipe_fingerprint(_cfg(tmp_path), pipe)
    assert recipe_fingerprint(_cfg(tmp_path), pipe) == base       # stable

    c = _cfg(tmp_path); c.vlm.model = "other-model"
    assert recipe_fingerprint(c, pipe) != base                    # model change
    c = _cfg(tmp_path); c.prefer_apple_vision = True
    assert recipe_fingerprint(c, pipe) != base                    # flag change
    assert recipe_fingerprint(_cfg(tmp_path), pipe[:2]) != base   # pipeline shape change


# ---- resume / reprocess ----------------------------------------------------

def test_same_recipe_resumes_nothing_reruns(tmp_path):
    cfg, pdf, calls = _cfg(tmp_path), _pdf(tmp_path), []
    process(pdf, cfg, pipeline=_pipe(calls))
    assert calls == ["a", "b", "c"]
    calls.clear()
    doc = process(pdf, cfg, pipeline=_pipe(calls))
    assert calls == []                       # fully cached, nothing re-ran
    assert doc.languages == ["a", "b", "c"]  # state restored from snapshot


def test_changed_recipe_reprocesses(tmp_path):
    cfg, pdf, calls = _cfg(tmp_path), _pdf(tmp_path), []
    process(pdf, cfg, pipeline=_pipe(calls))
    calls.clear()
    cfg.vlm.model = "a-different-model"       # recipe changes
    process(pdf, cfg, pipeline=_pipe(calls))
    assert calls == ["a", "b", "c"]           # NOT a silent no-op


def test_force_reprocesses(tmp_path):
    cfg, pdf, calls = _cfg(tmp_path), _pdf(tmp_path), []
    process(pdf, cfg, pipeline=_pipe(calls))
    calls.clear()
    process(pdf, cfg, pipeline=_pipe(calls), force=True)
    assert calls == ["a", "b", "c"]


# ---- rerun_from ------------------------------------------------------------

def test_rerun_from_reuses_earlier_stages(tmp_path):
    cfg, pdf, calls = _cfg(tmp_path), _pdf(tmp_path), []
    process(pdf, cfg, pipeline=_pipe(calls))
    calls.clear()
    doc = process(pdf, cfg, pipeline=_pipe(calls), rerun_from="b")
    assert calls == ["b", "c"]                # 'a' reused from snapshot
    # rewound to the post-'a' state, so 'b','c' re-ran ON that — not on the final doc
    assert doc.languages == ["a", "b", "c"]   # not ["a","b","c","b","c"]


def test_rerun_from_first_stage_is_full_reprocess(tmp_path):
    cfg, pdf, calls = _cfg(tmp_path), _pdf(tmp_path), []
    process(pdf, cfg, pipeline=_pipe(calls))
    calls.clear()
    doc = process(pdf, cfg, pipeline=_pipe(calls), rerun_from="a")
    assert calls == ["a", "b", "c"]
    assert doc.languages == ["a", "b", "c"]


def test_invalid_rerun_from_raises(tmp_path):
    cfg, pdf, calls = _cfg(tmp_path), _pdf(tmp_path), []
    with pytest.raises(ValueError):
        process(pdf, cfg, pipeline=_pipe(calls), rerun_from="nope")
