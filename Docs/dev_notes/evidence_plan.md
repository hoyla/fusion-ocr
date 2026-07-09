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
- **150-DPI re-check — DONE (2026-07-08): keep 150, confirmed on gold.** A/B `ocr_det` 150 vs
  200 DPI, deterministic, seeded n=50 FUNSD + n=50 SROIE (manifest
  `eval_out/manifests/dpi_recheck_2026-07-08.md`). **150 ≥ 200 on both** — FUNSD +0.011 recall /
  −0.009 CER (150 marginally *better*), SROIE +0.0015 (tie). The 5-pseudo-GT-page validation
  behind PR #16 holds on 100 human-GT items; 150 is faster + shares the raster cache. No revisit.

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

> **EXECUTED (2026-07-09) — D1 gated column, D2 blank probes, D3 divergence triage.** Runners
> `eval_out/{insertion_gate,divergence_triage,blank_probes}.py`; manifests
> `eval_out/manifests/{insertion_gate,divergence_triage,blank_probes}_2026-07-09.md`. **Archive
> integrity: 349/349 stream-A docs reproduce `stream_a_vlm/results.csv`.**
>
> - **D2 = PASS** (the one hard bar): **0 gated invented words on all 12** synthetic blank/near-blank
>   probes; the COPY positive control is recovered. Caveat: 11/12 were blank-gated *upstream* (0
>   detector segments → VLM never invoked), so the ink-gate's own drop-VLM-text-past-ink path was
>   not isolated by the synthetics; the real-instance companion stays OCRQ idx-924.
> - **D3 = ZERO candidates** under the pinned rule (VLM and detector agree almost everywhere —
>   median word-F1 0.913; `det_conf ≥ 0.80` is a no-op at min-mean 0.863; the only two sub-0.5-F1
>   items are VLM non-reads, correctly excluded). Corroborates D1 → tripwire (d) clear. No top-up.
> - **D1 = tripwire (b) FIRED** — gated (overlay) char-`insertion_rate` is *higher* than ungated
>   (`vlm_reading`): benefit −0.055 (FUNSD) / −0.086 (SROIE), inverting the pre-registered
>   expectation. **Tripwire (c) clear** (recall cost 0.004 / 0.014 ≪ 0.05 — gating is ~recall-free).
> - **Diagnosis (pending Luke's certification — NOT a settled verdict, per the pins' senior-eyes
>   rule):** a reading-order + det-fragment artifact, not overlay hallucination — the gated text
>   shares **94.4%** of its words with the ungated reading, is 1.06× the reference length, 0
>   superseded; the char-insertion gap exceeds the order-insensitive word gap (1−precision only
>   +0.04). **Same confound family as the strict-vs-band placement artifact** (the campaign's 5th).
>   The ink-gate's insertion *benefit* lives in the hallucination regime (blank/degraded pages),
>   which the blank-gate already covers, so it reads ~0 on these ink-full corpora.
> - **CERTIFIED (Luke, 2026-07-09):** the reading-order-confound diagnosis is accepted; on ink-full
>   corpora the **P2 gated proxy is the word-level figure** (gated `1 − word_precision` 0.184 FUNSD /
>   0.100 SROIE, ~recall-free), and the char-`insertion_rate` gate *benefit* is reserved for the
>   D2 hallucination regime (blank/degraded pages). So the certified P2 result is a **regime split**:
>   the D1 on-content cost + the D2 blank-regime benefit (0 gated invented words), not one headline
>   number. See `manifests/insertion_gate_2026-07-09.md`.

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

#### Operational pins (2026-07-09 — pre-execution)

*The three definitions the prose above leaves open, pinned before running so execution needs
no judgment calls. Field names, loaders, and paths below are verified against the code and the
archived artifacts, not assumed.*

**D1 — the gated column (`eval_out/insertion_gate.py`).** Pure re-scoring, no VLM compute:
the 349 stream-A VLM fused docs are archived at
`/Volumes/CORSAIR/Work_Corsair/fusion-ocr/eval_out/stream_a_vlm/out/<id>/doc.json` (CORSAIR
must be mounted); load with `Document.from_json` (`models.py`). References and caseless
handling exactly as `eval_out/stream_a_vlm.py` (`iter_pairs` / `_annotation_index` /
`_CASELESS_REF`).

- **Ungated hyp** = `page.vlm_reading` joined over pages (what `document.md` carries when
  VLM-read). Recomputing it must reproduce the committed `stream_a_vlm/results.csv` columns —
  a built-in check that the archive is the same run.
- **Gated hyp (the definition):** segments with non-empty `best_text` and not `superseded`,
  sorted by `compose.reading_key`, joined with `"\n"` — i.e. exactly the fallback branch of
  `eval.harness.recovered_text`, applied unconditionally. This is the text of
  `segment_index.json` / the overlay (same filter `placement.py` uses), so it scores the
  product artifact.
- Score both hyps with `metrics.score(ref, hyp, caseless=…)`. **Report the PAIR, per item and
  aggregate, for both columns:** `insertion_rate` (char-level — the pre-registered proxy) AND
  `word_recall`. The gate's *benefit* is ungated−gated insertion; the gate's *cost* is
  ungated−gated recall (true words dropped for lack of ink support). Publishing the benefit
  without the cost flatters the gate — the manifest carries both. (`1 − word_precision` is the
  word-level hallucination companion; it falls out of the same `score()` call, report it too.)
- Items with empty `vlm_reading` (guards discarded the read; ungated≈gated by construction):
  keep them, count them, report the count.

**D2 — the probe set (`eval_out/blank_probes.py`).** Twelve synthetic single-page PDFs,
generated deterministically by the runner with PyMuPDF drawing primitives (commit the script,
not the PDFs): 3 blank (white / off-white / light-grey), 3 speckle (faint dots, seeded RNG, at
increasing density), 2 text-free stamp-like marks (circular rubber-stamp shape; solid dark
box), 1 stamp with the word "COPY" (its ref = `"COPY"`), 3 ruled-but-empty (table grid / lined
page / empty form boxes). Ref = `""` for all but the COPY stamp.

- Full pipeline via `process()`, live MLX reader up, fresh out_dir, distinct digest per probe
  (the guard-check rerun's method). Record wall time and whether the VLM was invoked — true
  blanks should short-circuit at the blank-gate (verified 2026-07-06); the near-blanks are the
  real test.
- **Metric unit is WORDS** (as pre-registered above): invented words per probe =
  `hyp_words − word_overlap` from `score()` (multiset difference), computed for BOTH the
  ungated reading and the gated text (D1's definition). Char `insertion_rate` is also logged
  but degenerates to a raw count when ref is empty (`n = max(len(ref), 1)`) — words are the
  headline unit.
- **Pass/fail applies to D2 only:** gated invented words = **0 on every probe**; any nonzero
  is a gate bug — stop and diagnose. Ungated counts are reported, no bar.
- The four real OCR-Quality probes (blank 924/967/969, loop 654) stand as the companion set —
  already verified in situ 2026-07-06 (`ocrq_full_2026-07-06.md`); cite, don't re-run.

**D3 — the selection rule (`eval_out/divergence_triage.py`).** Universe = the same 349
archived docs. Per item, disagreement = word-multiset F1 between the normalized `vlm_reading`
and the concatenation of non-superseded segments' `det_text`:
`2·overlap / (|vlm_words| + |det_words|)` via `word_tokens` + the `score()` overlap (SROIE
casefolded both sides).

- **Candidates:** F1 < 0.5 AND mean `det_conf` over non-superseded segments ≥ 0.80 (the
  codebase's own det-trust bar, `fuse_det_conf_trust`) AND both sides non-trivial (normalized
  `vlm_reading` ≥ 20 chars; ≥ 3 det segments). "Both confident" operationalized: det by
  confidence; the VLM by having produced a guard-surviving, non-trivial reading (VLMs emit no
  calibrated confidence).
- **Sample:** `random.seed(1)`, `sample(candidates, min(20, len(candidates)))`. If fewer than
  20 qualify, take all and record the count — no threshold-relaxing top-up (that would unpin
  the rule).
- **Verdicts** (gold-anchored — unlike OCR-Quality, these corpora have real references):
  per item, compare each side's `score()` vs gold, inspect the image, classify
  **VLM-wrong / det-wrong / both-wrong / reference-fault** (keep the fourth bucket; the OCRQ
  adjudication showed it is a real class — expect it rare here). Claude-Vision-assisted
  inspection is fine (dev samples are cleared for Vision); Luke certifies the final table.
  Verdicts + per-item notes land in `eval_out/divergence_triage/verdicts.md`.

**Mechanics (all three):** run with the project `.venv` interpreter (the pyenv trap); D1/D3
need no reader, D2 needs MLX on :8080. Durable/resumable CSV-append runners in the
`stream_a_vlm.py` style; one manifest per runner in `eval_out/manifests/`
(`insertion_gate_<date>.md`, `blank_probes_<date>.md`, `divergence_triage_<date>.md`).

**Escalation tripwires (diagnosis triggers, NOT pass bars — D1/D3 remain first-measurement):**
(a) any probe with nonzero *gated* invented words; (b) aggregate gated insertion not lower
than ungated; (c) gate recall cost > 0.05 absolute on either corpus; (d) D3 verdicts
contradicting the D1 story (e.g. det-wrong dominating where the insertion numbers implied VLM
hallucination). Any of these fires → stop and diagnose before writing the manifest verdict —
that diagnosis is the interpretive step (the strict-vs-band placement confound is the
precedent for why it gets senior eyes).

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

> **EXECUTED (2026-07-09)** — runner `eval_out/stream_f_model_ab.py`, manifest
> `eval_out/manifests/stream_f_model_ab_2026-07-09.md`; 4 models × (labelled 5 + FUNSD n=50),
> against the zero noise floor. **Model-gen:** Qwen3.5-9B vs Qwen3-VL-8B is a *tie* on recognition
> at n=55 (the Δ0.005/n=4 recall gap doesn't survive) — keep Qwen3.5-9B on **robustness + speed**
> (Qwen3-VL-8B hit a bare-`.` repetition loop on 1/50 FUNSD → 262k chars, guard-missed; Qwen3.5-9B
> 0 such, ~4s faster). **Quant:** 8-bit is +0.006 recall / medCER-tie at ~23% slower + 2× memory —
> keep 4-bit. **New-gen (added this session):** **Qwen3.6-35B-A3B** (MoE, 3B-active, `qwen3_5_moe`,
> mlx_vlm 0.6.3-compat) **beats the default on quality AND speed** — recall +0.018, medCER −0.012,
> **~28% faster** (13.8 vs 19.2s), 0 runaways — a strong generalist-default candidate pending
> broader validation (n=55/2 corpora here; ~20 GB resident). Decisions surfaced for Luke, no default
> flipped. Also: generalise the repetition guard to catch low-entropy floods (roadmap item).

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
| 2 | D1 insertion reporting + manifest plumbing ✅ (2026-07-09) | ~½ day code |
| 3 | C placement metric ✅ | ~1 day code, runs ride on A |
| 4 | A VLM rows (FUNSD full, SROIE n=150) ✅ | ~1–2 days compute |
| 5 | D2 blank probes + D3 triage ✅ (2026-07-09) | ~½ day |
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
