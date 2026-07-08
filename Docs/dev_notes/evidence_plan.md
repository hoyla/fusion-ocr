# Evidence plan — closing the measurement gap

*Prompted by [review_03](review_03_260705.md): every mechanical finding from reviews 01/02 was
fixed; the evidential findings (calibrate thresholds, measure quant cost, scale the eval) were
the ones dropped — twice. This note pre-registers the measurement campaign the same way
[rapidocr_eval_plan.md](rapidocr_eval_plan.md) pre-registers the engine A/B: questions, methods,
and pass/fail criteria stated **before** running, so "we should measure more" becomes a
checklist with an end state. The architecture is not in question here; whether its promises are
**numbers** is.*

## The two claims that justify the architecture (currently unmeasured)

Everything distinctive about fusion-ocr rests on two promises with no metric today:

- **P1 — placement:** a searched word highlights the *right box* (the click-a-claim promise).
  The current `sRcl` only checks the text is findable *somewhere* in the PDF.
- **P2 — bounded hallucination:** the reading view's invented-text rate is known and small.
  `insertion_rate` exists in `eval/metrics.py` as the proxy and is reported nowhere; the
  ink-gate protects the overlay, not `document.md`.

Until P1 and P2 have numbers, the defensibility advantage over "just run MinerU-hybrid" is an
argument, not a measurement. They are streams C and D below and outrank everything else.

## Standing rules (apply to every stream)

1. **Selection protocol first.** Every run declares its item set up front (all items, or a
   seeded random sample) — no post-hoc exclusions without a logged reason per item.
2. **Durable artifacts.** Every run commits a small manifest to `eval_out/manifests/`:
   run date, machine, model names + versions, `recipe_fingerprint`, per-item scores (CSV),
   aggregate + variance. Numbers stop living only in doc prose on one machine. (Raw outputs
   stay gitignored; manifests are small and text.)
3. **Score the product, not just the VLM.** The harness currently scores the ungated
   `vlm_reading`. Every stream reports **both** the ungated reading and the fused/gated
   output, as separate columns — the gap between them is itself a finding.
4. **Noise floor before deltas.** No default changes on a delta smaller than the measured
   run-to-run variance (stream G). The Qwen3.5 switch (Δ0.005, n=4) is the cautionary case.
5. **Long runs live on the desktop** (always-on); manifests are committed so results are
   readable from either machine.

## Streams, in priority order

### A. Spend the gold data on disk (cheapest, biggest)

> **DETERMINISTIC PASS DONE (2026-07-07)** — full FUNSD (199) + SROIE (973), both PaddleOCR and
> Apple Vision. Manifest: `eval_out/manifests/stream_a_deterministic_2026-07-07.md`. Answers:
> **Q3 SROIE ~0.6 = scoring artifact** (100% uppercase GT; caseless fix shipped; true recall
> ~0.90 both engines). **Q1 FUNSD n=16 held** (det recall 0.82→0.78, RO-CER 0.25→0.29). **Q2
> Paddle-beats-Vision is document-type-dependent** — forms yes (0.81 vs 0.75), receipts a tie
> (Vision +0.01). Roadmap table + memory note updated. **Still queued:** the **VLM rows** (Q1's
> VLM CER 0.15, and the receipt "VLM CER win" claim — both need caseless re-scoring) and the
> **150-DPI re-check**.

> **VLM PASS DONE (2026-07-07)** — FUNSD 199 + SROIE seed-150, full pipeline. Manifest
> `eval_out/manifests/stream_a_vlm_2026-07-07.md`. **VLM out-recognises deterministic on both:**
> FUNSD 0.817/CER 0.170 (Q1's n=16 CER 0.15 HOLDS at scale); SROIE caseless 0.956/0.138 vs det
> 0.864/0.391 — the receipt VLM win is real and bigger corrected, so *with the VLM* receipts are
> VLM-favoured (updates the det-pass Q2 tie). **Fused placement (P1): fusion ≥ deterministic under
> the fair (band) metric** — SROIE 0.924 vs 0.864, FUNSD 0.564 vs 0.538. The scary strict-metric
> result (fused < det) was a granularity artifact (fusion's coarser boxes); `band=` mode added to
> `placement.py`. Residual FUNSD placement gap is shared by both engines (dense-form difficulty),
> not a fusion regression. **Still open:** 150-DPI re-check; gated-vs-ungated insertion (D);
> dense-form placement improvement.

Full-set runs on the two human-GT corpora already wired into the harness:

```bash
python -m fusion_ocr.eval --dataset funsd --no-vlm            # all ~200 forms, deterministic
python -m fusion_ocr.eval --dataset funsd                     # all ~200, VLM path
python -m fusion_ocr.eval --dataset sroie --no-vlm            # all 973 receipts, deterministic
python -m fusion_ocr.eval --dataset sroie --limit 150 --seed 1  # VLM on a seeded sample (cost)
```

Deterministic rows are cheap (no VLM); run them in full. VLM rows: full FUNSD, seeded n=150
SROIE (a full VLM SROIE run is ~days on one Mac — sample first, extend if variance is high).

**Pre-registered questions:**
- Do the n=16 FUNSD numbers (det CER 0.25 / VLM 0.15, recall 0.84) hold at n≈200?
- Does "PaddleOCR out-recognises Apple Vision" survive n≈200 + n≈973? (Current basis: 4 pages
  + one statistical tie.) → add `--apple-vision` rows for the deterministic comparison.
- Why is absolute receipt performance poor for *every* engine (recall ~0.6)? Diagnose before
  narrating: harness artifact (normalisation, matching) vs genuine engine weakness on thermal
  print. This number was published in the roadmap and never explained.
- **150-DPI re-check:** A/B `ocr_det` 150 vs 200 DPI on a seeded n=50 subset of each. The
  shipped default was validated on 5 pseudo-GT pages; this is the real test.
  *Criterion:* keep 150 iff recall delta < noise floor (stream G) on both corpora.

**On completion:** update or retire the affected claims — the
`feedback-paddleocr-is-the-deterministic-baseline` memory note, the roadmap benchmark table,
and the review_03 caveats — citing the manifest.

### B. Unblock IAM — handwriting beyond n=1

The headline capability (handwriting) is proven on one letter; 1,539 IAM pages sit on disk
unusable because the bundled annotations are OCR output (circular). Fix: source the original
FKI/IAM human transcriptions (registration required, free for research; the ASCII
`lines.txt`/`words.txt` carry the human text keyed by line id), write the small adapter to
re-pair them with our images, then run det/VLM rows as in stream A (seeded n=100 first —
IAM pages are dense).

**Pre-registered question:** does VLM handwriting recall generalise from the Mandelson 0.95?
*Interpretation guard:* IAM is clean-ruled English handwriting — a *floor* for the claim, not
proof on degraded FOI material. If sourcing the transcriptions stalls > a week, log it in the
manifest dir and move on — don't let B block C/D.

### C. P1 — box-placement accuracy (new metric, small code)

> **METRIC BUILT + DETERMINISTIC FLOOR MEASURED (2026-07-07).** `eval/placement.py` +
> `--dataset X --placement` + tests; manifest `eval_out/manifests/placement_deterministic_2026-07-07.md`.
> First-ever P1 numbers (deterministic path = detector geometry): **SROIE placement 0.85 /
> plain 0.91, gap 0.06** (single-column, placement ≈ recognition); **FUNSD placement 0.60 /
> plain 0.81, gap 0.21** (dense 2-D — gap is an upper bound, inflated by the metric's own
> line-assignment ambiguity). **Still open — the headline:** the **fused/VLM** placement (does
> fusion pin the VLM's words to the right box) needs the stream-A VLM run. The metric is the
> committed **regression guard** for the rapidfuzz port + fusion changes.

FUNSD (and SROIE) carry per-line **box + text** GT — placement is measurable with data already
on disk. Method (new `eval/placement.py`, ~100 lines):

1. Match each GT line box to fused segment boxes by IoU (Hungarian or greedy-best, IoU ≥ 0.3).
2. For each matched pair, score text agreement (word recall of segment `best_text` vs GT line
   text). A word is **well-placed** if it appears in the segment matched to the GT line that
   contains it.
3. Report **placement precision/recall** per page + aggregate; break out by source
   (`fused` vs det-only fallback) — the fused rows are the ones that test the alignment.

This is scored on `segment_index.json` — i.e. the gated product artifact, exactly what the
overlay renders. Also run it on the 4 hand-labelled pages + the Goldfinch rotated page (the
known-weak case, sRcl 0.65) so the metric's floor is anchored to a documented failure.

**Pre-registered criterion:** none yet — this is the first measurement; its job is to put a
number on P1 and become the regression guard for any fusion/alignment change (including the
rapidfuzz port, stream F). *A fusion change that improves CER but drops placement is a
regression.*

### D. P2 — hallucination measured and reported

1. **Report `insertion_rate`** in every harness/labels/dataset CSV and manifest — both ungated
   (`vlm_reading`) and gated (fused) columns. The gated-vs-ungated insertion gap is the
   measured value of the ink-gate. Zero new metric code; it exists and is dropped on the floor.
2. **Blank/near-blank probe** (targeted, cheap): a dozen synthetic pages — blank, faint
   speckle, a single stamp, a ruled-but-empty form — through the full pipeline. Count invented
   words in `document.md` and in the overlay. This is the documented failure mode of every
   end-to-end VLM (olmOCR shipped it broken); we claim architecture-level immunity in the
   overlay — demonstrate it, and quantify the ungated view's exposure. **Real probes already in
   hand:** the OCR-Quality 1000-run produced genuine instances — blank-page formula hallucination
   (idx 924/967/969) and a `[illegible]` repetition loop (idx 654) — use these alongside the
   synthetic set. **Guard check (post-#21): DONE 2026-07-06** — all four pages re-run through
   the full pipeline (fresh out_dir, live reader): blank pages never reach the VLM (0.0s, empty
   reading, no invented formula in `document.md`); the 654 loop recurred and was discarded, with
   det_text fallback. Results table in `eval_out/manifests/ocrq_full_2026-07-06.md`. Perf note:
   the repetition guard is post-hoc (654 still burned ~93s generating before discard) — a
   max_tokens cap is the refinement if loop pages prove common.
3. **Divergence triage** on stream-A outputs: pages where VLM and det strongly disagree but
   both are confident → human-inspect a seeded sample of 20; classify VLM-wrong / det-wrong /
   both. This is the qualitative anchor for the insertion numbers.

**Pre-registered criteria:** overlay insertion rate on blank probes = **0** (the gate's core
claim — any nonzero is a bug); `document.md` insertion rate gets *reported* with no pass bar
yet (first measurement), but becomes a release-note number — "defensible" means we publish it.

### E. Threshold sensitivity (not full calibration)

Full calibration of ~15 constants is over-engineering at this corpus size; **sensitivity** is
the honest, affordable version. For the four highest-leverage constants —
`fuse_min_sim` (0.34), `fuse_det_conf_trust` (0.80), `_MR_COVERAGE` (0.5),
`_LARGE_IMAGE_FRAC` (0.40) — re-run stream A's deterministic+fused scoring at ±30% of each
value (one-at-a-time, seeded n=50 subset) and plot the metric response.

**Pre-registered interpretation:** flat response → the constant is benign, document that and
stop worrying; steep response → it's load-bearing and earns a real calibration pass + a config
exposure. Either way, **fold the sensitive constants into `recipe_fingerprint`** so tuning one
re-keys the cache (today none are — stale-cache trap flagged in reviews 01 and 03).
Also: **decide `escalate_below`** — a routing-design pillar that ships disabled (0.0). Measure
escalation on stream A's worst-recall decile: if it doesn't pay there, delete the feature
rather than shipping it off (dead pillar = doc drift).

### F. Model/runtime deltas we shipped without measuring

- **4-bit vs 8-bit Qwen3.5-9B** on the labelled set + seeded FUNSD n=50: recall/CER/insertion
  + `t_vlm_read`. *Criterion:* keep 4-bit iff quality delta < noise floor. (Review_01 asked
  for this; it's a config-only A/B.)
- **Qwen3.5-9B vs Qwen3-VL-8B re-test at n≥50** (the default switched on Δ0.005/n=4). Cheap
  to piggyback on the same runs; whichever wins at n=50, the manifest replaces the anecdote.
- *(Engine A/Bs — RapidOCR, PP-OCRv6, PP-DocLayoutV3 — stay in
  [rapidocr_eval_plan.md](rapidocr_eval_plan.md); PP-OCRv6 tiny/small should be added there as
  the null hypothesis RapidOCR must beat. Streams A+C provide the corpus and the placement
  guard those A/Bs should score against.)*

### G. Noise floor (prerequisite for all future "X beats Y")

> **DONE (2026-07-08) — the floor is ZERO.** 3× identical config on a fixed seeded-30 FUNSD set,
> genuinely independent (distinct out_dir/digest → real ~22min VLM runs, not cache hits): all 30
> items **bit-identical**, every metric spread 0.00000 (recall/CER/insertion/placement). MLX
> 4-bit greedy decode (temp 0.0) is deterministic run-to-run. Manifest
> `eval_out/manifests/noise_floor_2026-07-08.md`. **So: single runs are trustworthy — don't
> repeat for variance; the binding constraint on a delta is now SAMPLE SIZE, not noise (the
> Qwen3.5 Δ0.005/n=4 fails on n, not variance). Unblocks the RapidOCR / PP-OCRv6 / quant A/Bs.**
> Caveat: re-measure if MLX/model/machine changes or any sampling (temp>0) is used.

Run the identical config 3× on the labelled set + seeded FUNSD n=30 (temperature is already
0.0; this measures the residual MLX/decode/env variance). Publish the per-metric spread in a
manifest. **Every future default change cites it**: a delta inside the floor is not a result.
If the floor is ~0, say so once and stop paying for repeats.

## LANDED (2026-07-06): the OCR-Quality 1000-doc run + Claude-Vision adjudication

Completed 1000/1000 (durable: `eval_out/ocrq_full/`; **run manifest — the committed record —
`eval_out/manifests/ocrq_full_2026-07-06.md`**, incl. the post-fix re-score of `results.csv` +
`hand_label_queue.csv` from the saved transcripts, 601 values corrected, pre-fix CSV archived,
score-1 agreement independently reconfirmed at 0.934). Our reading scored vs the Qwen-72B
`ocr_text`, then Claude-Vision-adjudicated the worst divergences. Kept to the triage/calibration
framing below — NOT accuracy. What it produced:

- **A harness bug, not an engine one (the headline).** `word_recall`/`WER` split on whitespace,
  meaningless for CJK (no word spaces) — it read ~0 on near-perfect Chinese and made the run look
  like a catastrophic Chinese failure (zh 0.35). Character-level showed 0.94–0.99. **Fixed: PR
  #20** (CJK-aware tokenisation). Corrected agreement by the 72B's rated quality: score-1 **0.93**,
  s2 0.88, s3 0.70, s4 0.35 — monotonic (tracks *reference* quality, as expected). This is a
  *proven example* of the harness-artifact class stream A hypothesises for the SROIE ~0.6 mystery
  — **but it is NOT the SROIE cause**: SROIE is Latin (word-spaced), so the CJK fix leaves it
  unchanged. The SROIE diagnosis stays open under stream A (normalisation / matching / reference
  format), now with the tokenisation class ruled out.
- **Adjudication of the 15 worst score-1 divergences (Vision):** ~half are the 72B *reference's*
  fault (duplication, repetition loops, "Sure, here is…" preamble, LaTeX-vs-plain format), not
  ours — consistent with the pilot. Verdicts: `eval_out/ocrq_full/adjudication.md`.
- **Two genuine OUR failure modes, both Vision-confirmed → both now guarded** (PR #21):
  (a) **blank/near-blank page → hallucinated formula**; (b) **figure-heavy / sparse page →
  `[illegible]` repetition loop** to the token cap.
- **P2 (ungated document.md hallucination) — first MEASURED instance, and it lands right.** On
  blank page 924 the deterministic engine found **0 ink** → the **ink-gate dropped the
  hallucination** → the searchable product (overlay/segment_index) is clean; the invented formula
  survived only in the ungated `document.md`. The moat works as designed. `insertion_rate` on
  `document.md` is the metric that flags this class at scale (stream D — compute + report it).

### Epistemic framing (held throughout — applies to any VLM-as-judge signal)

- **What it is:** a cross-family VLM-as-judge signal. Stronger than the Qwen-72B pseudo-GT
  (different model family judging, so not self-agreement) — but still a model's opinion, not
  ground truth. A VLM judge shares failure modes with VLM readers (plausible-text bias on
  degraded input).
- **Valid uses:** (a) *ranking* — find the worst-scoring pages as the shortlist for
  hand-labelling (feeds streams A/B/D3 with maximally informative items); (b) *divergence
  triage* — pages where judge, VLM reading, and det_text three-way disagree are the
  hallucination-candidate pool for D3; (c) *comparative* engine ordering, where consistent.
- **Invalid use:** promoting agree-rates into accuracy claims ("X% correct per Claude") — that
  repeats the OCR-Quality circularity with a better-dressed judge. Judge-approved ≠ true.
- **Action when it lands:** commit its manifest like any other run; pull the bottom-decile
  pages into the hand-label queue; cross-tab judge verdicts against stream-A gold scores on
  the FUNSD/SROIE overlap (if any) — that cross-tab is the *calibration of the judge itself*,
  and decides how much weight its verdicts get afterwards.

## What stays parked (explicitly, with triggers)

- **Thai accuracy** — still blocked on a Thai reader; placement (stream C) partially applies
  box-level without reading Thai, but transcription GT waits. Trigger: a Thai-literate
  collaborator or a validated Thai gold set.
- **Scanned-table cell accuracy** — still blocked on sourcing a genuinely scanned data table
  (roadmap gap since 2026-06-30). Trigger: first such doc in the corpus; then a small
  cell-level eval rides on stream C's matcher.
- **Office ingest eval** — nothing to measure until the LibreOffice adapter exists.

## Order of execution and cost

| Step | What | Cost (wall-clock, desktop) |
| --- | --- | --- |
| 1 | G noise floor + A deterministic full runs | hours, CPU-bound |
| 2 | D1 insertion reporting + manifest plumbing | ~½ day code |
| 3 | C placement metric | ~1 day code, runs ride on A |
| 4 | A VLM rows (FUNSD full, SROIE n=150) | ~1–2 days compute |
| 5 | D2 blank probes + D3 triage | ~½ day |
| 6 | F quant + model A/Bs | ~1 day compute |
| 7 | E sensitivity sweeps | ~1–2 days compute, automatable |
| 8 | B IAM sourcing + adapter | external dependency; parallel |

Steps 1–3 are the campaign's spine: after them, every subsequent number (including the
in-flight Claude run and the RapidOCR A/B) lands on a corpus with a known noise floor, a
placement guard, and a durable manifest trail.

## End state (definition of done)

- P1 and P2 have first published numbers, and both are regression-guarded in the harness.
- The FUNSD/SROIE headline numbers are full-set (or seeded-sample) figures with variance, in
  committed manifests — not prose.
- Handwriting evidence is n≥100 (IAM) or the blocker is documented.
- The four top thresholds have sensitivity curves; sensitive ones are fingerprinted + exposed.
- `escalate_below` is either evaluated-and-enabled or deleted.
- Every empirical claim in README/routing/roadmap/memory cites a manifest, or is reworded as
  untested. (The docs' per-claim honesty was never the problem — the cross-references were.)
