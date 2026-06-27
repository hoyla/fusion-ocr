"""OCR eval metrics — pure, deterministic, no deps."""

from __future__ import annotations

from fusion_ocr.eval.metrics import (
    aggregate, cer, edit_ops, insertion_rate, normalize, score, wer,
)


def test_edit_ops_classifies_operations():
    assert edit_ops(list("abc"), list("abc")) == (0, 0, 0)
    assert edit_ops(list("abc"), list("axc")) == (1, 0, 0)   # substitution
    assert edit_ops(list("abc"), list("ac")) == (0, 1, 0)    # deletion (ref 'b' dropped)
    assert edit_ops(list("ac"), list("abc")) == (0, 0, 1)    # insertion (hyp 'b' added)
    assert edit_ops([], list("ab")) == (0, 0, 2)
    assert edit_ops(list("ab"), []) == (0, 2, 0)


def test_cer():
    assert cer("hello world", "hello world") == 0.0
    assert abs(cer("hello world", "hallo world") - 1 / 11) < 1e-9   # 1 sub / 11 chars


def test_wer():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert abs(wer("the cat sat", "the dog sat") - 1 / 3) < 1e-9


def test_insertion_rate_is_the_hallucination_proxy():
    assert insertion_rate("the report", "the report") == 0.0
    # the system invented trailing text not in the reference
    assert insertion_rate("the report", "the report and more invented text") > 0.0


def test_normalize_whitespace_and_unicode():
    assert normalize("a   b\n\n c") == "a b c"
    assert normalize("é") == normalize("é")     # NFC: e+combining == é


def test_aggregate_micro_averages_over_pages():
    results = [score("aaaa", "aaaa"), score("bbbb", "bxbb")]   # 0 + 1 char error / 8 chars
    agg = aggregate(results)
    assert agg["pages"] == 2
    assert abs(agg["cer"] - 1 / 8) < 1e-9
