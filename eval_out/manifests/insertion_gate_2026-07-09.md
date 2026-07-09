# Run manifest — Evidence-plan stream D1, the gated column (ink-gate benefit vs cost)

*[evidence_plan.md](../../Docs/dev_notes/evidence_plan.md) stream D1 — report `insertion_rate`
ungated (`vlm_reading`) **and** gated (fused overlay), the gap being the measured value of the
ink-gate. Pure re-score of the 349 archived stream-A VLM docs (`stream_a_vlm/out/*/doc.json`), NO
VLM compute. Runner `eval_out/insertion_gate.py`.*

**FIRST MEASUREMENT — tripwire (b) fired, was diagnosed, and the diagnosis is CERTIFIED
(Luke, 2026-07-09): the reading-order-confound reading is accepted. P2 decision — on ink-full
corpora the gated proxy is the WORD-LEVEL figure (`1 − word_precision` / invented words); the
char-`insertion_rate` gate *benefit* is reserved for the D2 hallucination regime (blank/degraded
pages, where it is 0). Char-level is still reported alongside, for transparency.**

## Definitions (pinned)

- **Ungated** = `"\n".join(recovered_text(p) for p in doc.pages)` — what `document.md` carries
  when VLM-read (`page.vlm_reading` for VLM-read pages). This is the exact hyp `stream_a_vlm.py`
  scored, so it must reproduce the committed `results.csv`.
- **Gated** = non-superseded segments' `best_text` in `compose.reading_key` order, joined `"\n"`
  (the fallback branch of `recovered_text`, applied unconditionally) = `segment_index.json` / the
  overlay — the searchable product, same filter `placement.py` uses.

**Archive integrity: 349/349 items reproduce `results.csv`** (word_recall / insertion_rate / cer /
ref_chars to 4 dp) — this is the same run. `vlm_empty` = 2 (guards discarded the read; ungated ≈
gated by construction — both fall back to det_text).

## Result (micro-averaged; SROIE caseless both sides)

| corpus | ungated ins | gated ins | **benefit** (u−g) | ungated recall | gated recall | **cost** (u−g) | ungated 1−prec | gated 1−prec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FUNSD (199) | 0.0666 | 0.1211 | **−0.0545** | 0.8174 | 0.8137 | 0.0036 | 0.147 | 0.184 |
| SROIE (150) | 0.0759 | 0.1615 | **−0.0856** | 0.9559 | 0.9419 | 0.0140 | 0.056 | 0.100 |
| overall (349) | 0.0698 | 0.1352 | **−0.0654** | 0.8672 | 0.8598 | 0.0074 | 0.113 | 0.153 |

- **Tripwire (b) — gated insertion is NOT lower than ungated; it is *higher* — fires on both
  corpora** (benefit negative). This inverts the pre-registered expectation ("gated < ungated =
  the gate's benefit").
- **Tripwire (c) — recall cost — does NOT fire** (0.0036 / 0.0140, both ≪ 0.05). The gate drops
  almost no true words: recovery is essentially preserved through gating.

## Diagnosis — reading order + det fragments, NOT overlay hallucination (pending certification)

The negative "benefit" is **not** the overlay inventing text. Evidence:

- **The gated overlay shares 94.4% of its words with the ungated reading** (FUNSD 94.2%, SROIE
  94.6%) — only ~5.6% of gated words are new. It carries essentially the *same content*, reordered.
- **Gated text is only 1.036× the ungated length and 1.059× the reference** — no duplication or
  noise flood. Segment sources: 8653 `fused` + 2107 `paddle`, **0 superseded**.
- **The char-level `insertion_rate` gap (0.055 / 0.086) exceeds the order-insensitive word gap**
  (1−precision rises only 0.036 / 0.045). The excess is reading order: the gated text is the
  segments in `reading_key` (layout) order, which on dense 2-D forms and receipt footers diverges
  from both the VLM's document-order reading and the flat reading-order GT, so char-Levenshtein
  charges transposed blocks as insertions. Concrete: on `funsd/0001209043` both texts open
  identically then the gated version jumps into the scores table ("…Magazine TIME PM6 PROVED
  RECALL COMMENTS SCORE BASE 8 8…") where the reading interleaves label/value fields; on
  `sroie/X51006414631` the gated text pulls the footer totals ("Total Amount: $7.00 GST @6%…")
  ahead of the line items.
- **The genuine ~5.6% / ~4pp-precision component is real but small**: raw-paddle fragments the
  clean VLM reading omitted — form-structure text and OCR noise (`"8 8"`, `"PM6"`, a stray `冰`,
  `"Refationship-Building…"`). This is the overlay honestly carrying every inked mark, not invention.

**This is the same confound family as the strict-vs-band placement artifact** (manifest
`stream_a_vlm_2026-07-07.md`): a char/order-sensitive metric penalising the gated product for a
format mismatch (bag-of-layout-fragments vs flat reading-order reference), not a real defect. It
is the campaign's 5th metric-artifact catch.

## Why the pre-registered "benefit" doesn't appear here — regime

The "gate reduces insertion" claim is a **hallucination-regime** statement. FUNSD/SROIE are clean,
ink-full printed corpora where the VLM barely hallucinates (ungated insertion ~0.07) — there is
almost nothing for the ink-gate to *remove*, while the overlay reassembly *adds* det-fragments and
reorders, so the char proxy inverts. The gate's benefit lives where the VLM actually invents text
with no ink under it — **blank / near-blank / degraded pages**, which is exactly stream **D2**
(synthetic probes) and the real OCR-Quality blank-page instance (idx 924: ungated invented a
formula; the ink-gate found 0 ink and the gated overlay stayed clean —
`manifests/ocrq_full_2026-07-06.md`). D1 on these gold corpora is the wrong regime to see the
benefit; it is the right regime to confirm the **cost is ~0** (tripwire c clear).

## What this settles / doesn't

- **Reads cleanly (high confidence):** the gate is nearly **recall-free** on content (cost
  0.004 / 0.014) and injects almost no new words into the overlay (94.4% shared with the reading);
  archive integrity confirmed.
- **Certified (Luke, 2026-07-09) — option (i):** the reading-order-confound diagnosis is accepted;
  the on-content P2 gated proxy is the **word-level** figure — gated `1 − word_precision`
  **0.184 (FUNSD) / 0.100 (SROIE)** — reported alongside the char-level for transparency. The
  char-`insertion_rate` gate *benefit* is reserved for the D2 hallucination regime.
- **P2 now has first published gated numbers.** The D1 on-content **cost** (gated word-level
  hallucination ~0.18 / ~0.10, ~recall-free) + the D2 blank-regime **benefit** (0 gated invented
  words) are the certified P2 pair — a regime-split number, not a single headline. Note the gated
  word-level rate sits ~0.04 *above* the ungated reading's (the overlay honestly carries the
  detector's real ink fragments), so the ink-gate is not a hallucination-*reducer* on ink-full
  pages; its reduction shows only where the VLM invents past the ink (D2 / OCRQ-924).

## Caveats

- `insertion_rate` is char-level and order-sensitive; on the gated *reassembly* (layout-ordered
  fragments) it conflates reordering with invention. The order-insensitive `1−word_precision` /
  invented-words are the fair hallucination measures on these corpora.
- SROIE is the seeded 150 (VLM cost), not all 973. 2 `vlm_empty` items included (ungated≈gated).

## Artifacts

- `eval_out/insertion_gate/results.csv` — per-item ungated+gated counts (reproduce flag included)
- `eval_out/insertion_gate/summary.json` — per-corpus aggregates + tripwire evaluation
- `eval_out/insertion_gate.py` — the runner (archive-integrity check built in)
