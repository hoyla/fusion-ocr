"""Evaluation harness — turns "trustworthy/defensible" from a claim into a number.

Ground truth without hand-labelling: a born-digital page carries its own exact text in
the layer, so we render it to an image (stripping the text layer), force the OCR path,
and measure the recovered text against the embedded text. That gives a real CER/WER +
an insertion (hallucination) rate on realistic document content — financial tables,
multi-column layouts — automatically and reproducibly.

Two kinds of number, because they answer different questions:
  - word_recall / word_precision are ORDER-INSENSITIVE: recall = fraction of reference
    words recovered (recognition completeness), precision = fraction of output words that
    are real (1 - precision is the hallucination rate). These isolate recognition.
  - CER / WER are sequence-based, so they fold in reading order too. A high CER with a
    high recall means the text was recognised but mis-ordered (multi-column), not misread
    — trust CER mainly on single-column pages. (First run bore this out: ~95% recall /
    ~96% precision throughout, but CER 0.06 on a single-column page vs 0.3-0.8 on
    multi-column ones — i.e. recognition is solid; linearised reading order on complex
    layouts is the weak point.)

Honest limitations: rendered-clean pages are EASIER than real scans (no scan noise, skew,
or JPEG artifacts), so this is a floor on difficulty; the text-layer order isn't
guaranteed to be true visual order, so CER/WER carry that noise. Genuinely degraded scans
/ handwriting still need a small hand-labelled set (a follow-up). What it buys today: a
defensible recognition + hallucination number and a regression guard, instead of anecdote.
"""

from .metrics import aggregate, cer, edit_ops, insertion_rate, normalize, score, wer

__all__ = ["cer", "wer", "insertion_rate", "edit_ops", "normalize", "score", "aggregate"]
