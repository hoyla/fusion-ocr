# Reading-order measurement — method and first numbers

*Companion to [handover_reading_order.md](handover_reading_order.md). That note set up the
problem; this one records how we measured it and what we found. Status of record is still
[roadmap.md](roadmap.md) / [done.md](done.md).*

## The problem the handover left

Measuring reading order needs a reference whose line order **is** the true reading order. We
had two reference sources and neither qualified:

- **Born-digital text layer** — content-stream order, which scrambles multi-column pages.
- **3rd-party FUNSD annotations** — stored in *annotation* order, not reading order (the
  `eval/datasets.py` loader even warned "trust recall over CER on forms").

The hand-labelled set transcribes in true reading order, but `TestPDFs_01` has no strong
*scanned* multi-column material to label (the multi-column docs in it are born-digital; the
scanned docs are single-column prose, Thai forms, or handwriting). So the hand-labelled route
couldn't reach the cases where reading order actually bites, at any scale.

## The unlock: reconstruct reading order from FUNSD's per-line boxes

FUNSD is ~200 scanned forms with genuine human text GT, and every annotated line carries a
`box`. Forms are read by the universal convention — top-to-bottom, left-to-right within a row
— so we can **reconstruct** the reading order geometrically and use it as the reference:

> `_reading_order` (in `eval/datasets.py`): cluster lines into y-bands ~one line-height tall
> (so a row of side-by-side label/value fields stays together), then sort left-to-right within
> each band.

This is a **constructed** oracle — a principled proxy for human reading order, not a hand
transcription — and the note in the code and the docstring says so. It's defensible because the
sort encodes the same convention a human uses on a form, and because the thing it scores is
**independent** of the sort: the VLM reads from document understanding (it never sees the
boxes), and the deterministic path orders by PP-DocLayoutV2's learned `order` head. So a low
CER against this reference is genuine agreement between two independent orderings, not a sort
compared to itself.

With this reference, the existing recall-vs-CER split (`eval/metrics.py`) becomes interpretable
on forms: **recall** isolates recognition (order-insensitive), **CER** folds in order, and the
lift of CER over the recognition floor is the reading-order error.

## First measurement (FUNSD test split)

Run: `python -m fusion_ocr.eval --dataset funsd [--no-vlm] --limit N`. CER below is now
**reading-order** CER (the loader scores against the reconstructed order); the annotation-order
CER is shown for contrast (it's what the scorecard reported before this change).

| path | n | recall | CER vs annotation-order | CER vs **reading-order** |
| --- | --- | --- | --- | --- |
| deterministic (`--no-vlm`, PaddleOCR + PP-DocLayoutV2 order) | 6 | 0.78 | 0.44 | **0.25** |
| VLM (product default, Qwen3.5-9B reads order) | 6 | 0.79 | 0.33 | **0.10** |
| deterministic (`--no-vlm`) | 16 | 0.82 | — | **0.25** |
| VLM (product default) | 16 | 0.84 | — | **0.15** |

(The n=16 reading-order CER held close to n=6 — VLM 0.15 over 16 forms, recall 0.84 — so the
signal is stable, not a small-sample artifact. Per-page reading-order CER spans 0.018–0.36; the
high end is recognition-limited pages, e.g. recall 0.44 on a degraded form, not mis-ordering.)

Read:

- **Reordering to reading order roughly halves (det) to thirds (VLM) the CER** — confirming
  the annotation-order reference was genuinely scrambled, and that both the learned order head
  and the VLM put these complex forms into near-correct reading order.
- The **VLM result is the stronger evidence**: its ordering is independent of the box sort, and
  it still lands at reading-order CER 0.10–0.15 (one page 0.018). So the answer to the roadmap question — *does the
  reading order hold on multi-column / complex scanned layouts?* — is **yes, on forms**.
- The residual CER is mostly **recognition**, not order: the worst page (`92039708`) has recall
  0.44 (a degraded form), which drags CER regardless of ordering. Where recognition is good,
  CER tracks the recognition floor closely.

## What this does and doesn't settle

- **Settles:** a reproducible, scalable reading-order signal on complex scanned *forms* — no
  hand-labelling, ~200 forms available, regression-guardable.
- **Doesn't settle:** newspaper-style multi-column *prose* reading order (FUNSD is forms, not
  columns of running text). The born-digital `TestPDFs_02` annual reports exercise multi-column
  prose but only via the content-stream-order harness; a hand-labelled multi-column scanned
  prose page is still the gap if we want that case measured directly.
- **Caveat to keep honest:** the FUNSD reading order is geometric, not human-annotated. It's the
  right proxy for forms; don't over-read a CER difference of a few points as a reading-order
  verdict on its own — cross-check with the recall floor.

## Where the code lives

- `eval/datasets.py` — `_reading_order` (the box sort), `funsd_reference` (now reading-order),
  `funsd_reference_annotation_order` (raw order, kept for contrast). Module docstring carries
  the methodology + caveat.
- `eval/metrics.py` — the CER/WER vs recall/precision split (unchanged).
- `tests/test_eval_datasets.py` — `test_funsd_reference_reconstructs_reading_order_from_boxes`.
