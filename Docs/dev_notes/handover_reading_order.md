# Handover — Reading-order measurement (Now), and why Tables stayed in Next

*Point-in-time pickup note, 2026-06-30. A new session can start the reading-order work from
here. Status of record is still [roadmap.md](roadmap.md) / [done.md](done.md); this is the
"how to pick it up" companion.*

## Reading order — ready to start (top of Now)

**Why it's possible now.** The hand-labelled eval set transcribes pages in *true visual
reading order*, so CER/WER scored against those transcripts already fold in reading order —
unlike the born-digital text layer, which is content-stream order and scrambles multi-column
pages, so it can't be a reading-order oracle. The harness is already built: the searchability
eval (`--labels`) scores recovered text against these transcripts.

**The metric is already there.** `eval/metrics.py` reports both an order-*sensitive* pair
(`CER`/`WER`) and an order-*insensitive* pair (`word_recall`/`word_precision`). **A high
recall with a high CER = recognised but mis-ordered** — that gap *is* the reading-order
error. So measuring reading order is reading the existing scorecard on the right pages, not
new infrastructure.

**The gap to close — pages, not code.** The current labelled set (4 pages: Mandelson
note/email, rotated Goldfinch, redacted EFTA) is mostly single-column prose, where reading
order barely bites. Reading-order error only shows up on **multi-column / complex layouts**.
So the first step is *labelling*, not coding:

1. Identify a few genuinely **multi-column / complex-layout scanned** pages in the corpus
   (`samples/TestPDFs_01`) and transcribe them in true reading order into the labelled set
   (`eval_labels/labelset.json` + a `.txt` each — both gitignored; see
   [eval-labelling.md](../eval-labelling.md)). Note: the `TestPDFs_02` annual reports are
   *born-digital* multi-column — their order is already testable via the born-digital harness;
   the hand-labelled oracle's unique value is on **scanned** complex layouts with no text layer.
2. Re-run `python -m fusion_ocr.eval --labels eval_labels/labelset.json` and read the
   recall-vs-CER split per page. That quantifies reading order.
3. Interpret against the layout stage: reading order comes from **PP-DocLayoutV2's learned
   `order` head** (see [routing.md](../routing.md)). The measurement tells us whether that
   learned order holds on multi-column / complex pages, or where it needs help.

**Where the code lives:** `eval/labels.py` (`evaluate_labelset`), `eval/__main__.py`
(`--labels`), `eval/metrics.py` (the CER/WER vs recall/prec split), `eval-labelling.md` (the
labelling guide). No new infra expected — it's labelling + reading the metrics already emitted.

## Why Tables was NOT promoted to Now (decision, 2026-06-30)

We explicitly weighed promoting the **Tables** item to Now alongside the output-artifacts doc,
and chose to **keep it in Next**. The reasoning, so a future session doesn't re-promote it
without the missing piece:

- Its headline sub-item — clean per-cell content on **scanned** tables via the focused VLM
  table read — is **blocked**: there is no genuinely *scanned data table* in the test corpus
  to exercise or test the path. In `TestPDFs_01` the scanned docs (Thai forms) are layout-
  classified `paragraph`/`header`/`footer`, not `table`; the `table`-classified docs are
  *born-digital* (handled by `find_tables`). So the path is currently **untested** (the
  2026-06-30 coverage-gap note in the roadmap).
- Promoting a blocked, large item would make "Now" aspirational rather than actionable —
  which is exactly what the Now/Next split exists to prevent.
- The **unblocked slice** — born-digital multi-level-header semantics for `find_tables`, and
  cross-validating `find_tables` vs the vision grid — *could* move up on its own **if** someone
  actively starts tables. Absent a scanned-table sample, the flagship work can't proceed.
- **Trigger to revisit:** a genuinely scanned data table lands in the corpus → then promote,
  source the sample, and build the focused-VLM-table-read test + path together.

## Session state this note hands off from

- **Searchability eval** shipped (PRs #7/#9): the hand-labelled eval also scores what `find()`
  hits in the output PDF (`overlay.pdf` / source text layer) vs the reading view. Quantified
  the word-level fusion fix (searchable recall = reading on handwriting/printed) and surfaced
  the **rotated-dense-print** overlay gap — parked as a search-*precision* refinement, **not**
  data loss (the full reading is in `document.md`/`doc.json`, which Giant indexes as separate
  search views with fallback). See [done.md](done.md).
- **Output-artifacts doc** shipped (PR #10): [outputs.md](../outputs.md).
- All merged to `main`. Services (MLX reader `:8080`, `fusion-ocr-serve` `:9473`, worker) were
  last seen running on merged code.
