# Run manifest — Evidence-plan stream C, box-placement (P1), deterministic

*[evidence_plan.md](../../Docs/dev_notes/evidence_plan.md) stream C — the first-ever number on
**P1**: does a recovered word land on its own line's box (click-a-claim), not just somewhere on
the page? Review 03 named this the single most load-bearing **unmeasured** claim. New metric
`eval/placement.py` (+ `--dataset X --placement`); scored on the gated `page.segments` (=
overlay / segment_index), the artifact a reporter actually clicks.*

## Metric

Each of our segments is assigned to the GT line box it overlaps most (IoU ≥ 0.3, coordinates
scaled px→pt by `page.width/image_width`). A GT word is **well-placed** if it appears in the
segment(s) assigned to *its own* line. Then, micro-averaged:

- `placement_recall` = well-placed GT words / all GT words — **the P1 number**
- `plain_recall` = GT words present anywhere on the page (recognition, ignoring place)
- `placement_gap` = plain − placement — pure mis-placement (read, but pinned to the wrong box)

## Results (deterministic path, full sets, 2026-07-07)

Scored from the saved stream-A `doc.json`s (same run as
[stream_a_deterministic](stream_a_deterministic_2026-07-07.md)). SROIE caseless (matching the
scoring fix).

| dataset | engine | placement_recall | plain_recall | gap |
| --- | --- | --- | --- | --- |
| FUNSD (forms, n=199) | PaddleOCR | 0.601 | 0.808 | 0.207 |
| FUNSD (forms, n=199) | Apple Vision | 0.552 | 0.751 | 0.199 |
| SROIE (receipts, n=973) | PaddleOCR | 0.853 | 0.912 | 0.060 |
| SROIE (receipts, n=973) | Apple Vision | 0.836 | 0.908 | 0.072 |

## Read it carefully — three caveats that bound the claim

1. **This is the DETERMINISTIC path — it tests detector geometry, not fusion.** These segments
   are PaddleOCR/Vision boxes carrying their own det_text. The headline P1 claim — does *fusion*
   put the *VLM's* words on the right boxes — needs the VLM run (stream A VLM pass, queued). This
   is the placement **floor** from geometry alone.
2. **The FUNSD gap (0.21) is an upper bound on true mis-placement.** Forms are dense 2-D grids of
   small label/value boxes, so the metric's own best-IoU line-assignment is itself ambiguous
   (adjacent fields overlap). Some of the "gap" is the *reference/metric* struggling to assign a
   box to the right line, not the pipeline mis-placing. **SROIE is the cleaner signal:** single
   column, boxes map cleanly to lines → a small **~0.06 gap**, i.e. on receipts placement ≈
   recognition (words land where they belong).
3. **First measurement — no pass/fail bar yet** (per the pre-registration). Its job is to exist
   and become the **regression guard** for any fusion/alignment change (the rapidfuzz port, and
   the VLM-fused runs): a change that lifts CER but drops placement is a regression.

## What this settles / doesn't

- **Settles:** P1 is now measurable and has a deterministic floor. On clean single-column scans,
  deterministic placement is strong (gap ~0.06). The metric is committed, tested, and CLI-runnable
  (`--dataset sroie --placement --no-vlm`), so it is a live regression guard, not a one-off.
- **Doesn't settle:** the *fused* (VLM) placement — the actual click-a-claim number — pending the
  VLM run. And the plan's idea of anchoring on the hand-labelled Goldfinch rotated page
  (sRcl 0.65) needs per-line **box** GT, which the transcript-only hand labels lack — a separate
  sourcing step.

## Artifacts

- `src/fusion_ocr/eval/placement.py` — the metric (+ `datasets.evaluate_placement`, `--placement`)
- `tests/test_eval_placement.py` — pure-logic tests (scale, misplacement, caseless)
- scored from `eval_out/stream_a/out/*/doc.json` (the committed stream-A run)
