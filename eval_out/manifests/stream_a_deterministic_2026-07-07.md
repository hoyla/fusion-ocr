# Run manifest — Evidence-plan stream A, deterministic full gold sets

*[evidence_plan.md](../../Docs/dev_notes/evidence_plan.md) stream A. FULL sets, no sampling:
FUNSD 199 forms + SROIE 973 receipts, every split, PaddleOCR **and** Apple Vision,
deterministic-only (`--no-vlm` equivalent — the recogniser's own text, no VLM). Answers the
three pre-registered stream-A questions. This is the recognition/reading-order floor; the VLM
rows, placement (P1) and hallucination (P2, C/D) are separate streams.*

## Run

- **Completed:** 2026-07-07 (started 07-06, paused for a machine restart, resumed — the runner
  is resumable: one CSV row per item, skips done tuples).
- **Items:** 2,344 scored = (199 FUNSD + 973 SROIE) × {paddle, apple_vision}. 0 errors, 0
  empty-GT skips.
- **Engines:** PaddleOCR (PP-OCRv6_medium det/rec, PP-DocLayoutV2 order) and Apple Vision
  (ocrmac), both via the image→PDF ingest adapter. paddleocr 3.7.0 / paddlepaddle 3.3.1,
  PyMuPDF 1.27.2.3. Recipe fingerprint `933f7fef0409d2d0` (deterministic pipeline).
- **Scoring:** `eval.metrics.score`, CJK-aware tokens (PR #20). Reference = concatenated
  annotation line texts; FUNSD in reconstructed reading order (the geometric oracle), SROIE in
  annotation order (single-column ≈ reading order).
- **Selection protocol:** all items, all splits, no exclusions.
- **Artifacts:** `eval_out/stream_a/results.csv` (per-item, raw metric), saved `doc.json`s
  under `eval_out/stream_a/out/` (the re-score source), runner `eval_out/stream_a.py`.

## The scoring artifact this run exposed (and the fix)

SROIE's aggregate recall has sat at an unexplained **~0.6** since the first benchmark (roadmap
table, n=12). This run diagnosed it — the evidence-plan "diagnose before narrating" rule:

- **SROIE ground truth is 100.0% uppercase** (measured over all 345,189 alphabetic chars in the
  corpus). Its transcribers upper-cased everything; the pipeline reads the receipt's real mixed
  case, so **case-sensitive scoring charged every correctly-cased letter as an error.** FUNSD is
  44.5% uppercase (real case) — unaffected.
- A secondary ~3-point word-recall loss is **label-gluing** — GT `TEL : 03-...` vs our
  `TEL:03-...`, a whitespace-tokenisation difference (CER is immune; only word metrics see it).

Both are *dataset-convention artifacts, not engine weakness.* Fix (this PR): `score(caseless=)`
folds case on both sides, driven by a per-source `datasets._CASELESS_REF = {"sroie"}` flag,
documented with the measured justification. Case stays significant by default (it is OCR
fidelity for a real document); only SROIE, whose reference discards case, is scored caseless.
This is the **third instance** of the harness-artifact class the review flagged (after CJK
tokenisation, PR #20) — same lesson: a low number was the metric, not the engine.

## Results — recall (word, micro-averaged)

Three columns show the artifact peeling away: `raw` (as historically reported) → `casefold`
(case-insensitive, the SROIE convention fix) → `+punct` (also split glued punctuation). All
applied uniformly to both datasets/engines, so the comparison is fair.

| dataset | engine | raw | casefold | +punct |
| --- | --- | --- | --- | --- |
| FUNSD (forms) | PaddleOCR | 0.783 | 0.788 | 0.810 |
| FUNSD (forms) | Apple Vision | 0.705 | 0.711 | 0.749 |
| SROIE (receipts) | PaddleOCR | 0.603 | 0.874 | 0.901 |
| SROIE (receipts) | Apple Vision | 0.610 | 0.872 | 0.911 |

FUNSD barely moves under casefold (its GT is real-case) — confirming the raw FUNSD number is
already honest and the correction is SROIE-specific, not a blanket softening.

## Results — CER + insertion (char, micro-averaged)

CER folds in reading order / sequence / spacing. Casefold removes the uppercase artifact; note
what it leaves.

| dataset | engine | CER raw | CER caseless | insertion_rate |
| --- | --- | --- | --- | --- |
| FUNSD (forms) | PaddleOCR | 0.287 | 0.282 | 0.101 |
| FUNSD (forms) | Apple Vision | 0.329 | 0.322 | 0.091 |
| SROIE (receipts) | PaddleOCR | 0.529 | 0.396 | 0.090 |
| SROIE (receipts) | Apple Vision | 0.537 | 0.408 | 0.091 |

**Read SROIE CER carefully.** Caseless drops it 0.53 → ~0.40 (the uppercase artifact) but it
stays high — because CER also folds in **word-spacing** (`TEL : 03` vs `TEL:03` = deletions) and
**sequence order** against a *flat annotation-order* reference. Recognition *completeness* is
the order-insensitive **recall (~0.90)**; the residual CER is spacing/order disagreement, not
misread characters (the "high recall + high CER = recognised but mis-sequenced" case in
`score`'s docstring). So on receipts, **recall is the recognition metric; CER is not clean**
against this reference. FUNSD CER is barely touched by casefold (real-case GT), so its
reading-order CER stands as reported.

(`insertion_rate` here is the *deterministic engine's*, not the VLM hallucination signal — that
is stream D, on the VLM path. Under casefold the SROIE edit path re-aligns and shifts errors
toward insertions — a composition change, not more hallucination.)

## The three pre-registered questions — answered

**Q1 — Do the n=16 FUNSD numbers hold at n≈200?** **Yes (deterministic path).** Reading-order
CER 0.25 → **0.287**, recall 0.82 → **0.783** at n=199 vs the n=16 figures in
[reading_order_measurement.md](../../Docs/dev_notes/reading_order_measurement.md). Slightly
softer, same regime — the small sample was mildly optimistic, not a fluke. *Caveat:* this run
is deterministic-only; the n=16 **VLM** row (CER 0.15) is not re-tested here — that needs the
stream-A VLM pass (queued).

**Q2 — Does "PaddleOCR out-recognises Apple Vision" survive n≈200 + n≈973?**
**Partly — it is document-type-dependent, and the blanket claim is too strong.**
- **Forms (FUNSD): yes, clearly.** Paddle leads recall 0.783 vs 0.705 (raw) / 0.810 vs 0.749
  (+punct) and CER 0.287 vs 0.329 — a stable ~6–8 point recall gap.
- **Receipts (SROIE): no — a tie, Vision marginally ahead on recall.** Recall 0.901 vs 0.911
  (+punct), 0.874 vs 0.872 (casefold); CER 0.529 vs 0.537 (Paddle a hair better). Inside noise.

So the standing memory note "PaddleOCR out-recognises Apple Vision (measured)" holds on **forms**
but not on **printed receipts**, where they are equivalent. Update the claim to name the
document class rather than assert it globally.

**Q3 — Why is SROIE recall ~0.6 for every engine?** **Solved: a scoring artifact, not the
engine** (see above). True receipt recognition is **~0.90** for both PaddleOCR and Apple Vision.
The old roadmap benchmark table's SROIE row (and its "VLM's big CER win on receipts" reading)
must be re-read once the VLM row gets the same caseless treatment.

## Claim hygiene

Valid from this run: FUNSD reading-order figures at scale; the Paddle-vs-Vision verdict by
document type; the SROIE-artifact diagnosis + true-recognition figure; the case-insensitive
scoring fix. **Not** established here: any VLM-path number, handwriting (IAM, stream B),
placement (P1, stream C), or hallucination rate (P2, stream D) — separate streams. Absolute
recall against these references still carries the geometric-reading-order-oracle caveat (FUNSD)
and the annotation-order assumption (SROIE).
