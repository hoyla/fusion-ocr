# D3 divergence triage — verdicts

*Evidence-plan stream D3. Universe = the 349 archived stream-A VLM docs (FUNSD 199 + SROIE 150).
Runner `eval_out/divergence_triage.py`; full per-item F1 table `f1_all.csv`. NO reader / NO VLM
compute. **Pending Luke's certification.***

## Result — the pinned rule selects ZERO candidates (a finding, not a gap)

The selection rule (pinned pre-execution): `F1(vlm_reading, det_text) < 0.5` **AND** mean
`det_conf` ≥ 0.80 **AND** both sides non-trivial (`vlm_reading` ≥ 20 chars, ≥ 3 det segments).
On FUNSD + SROIE **no item qualifies**, so per the pin ("take all and record the count — no
threshold-relaxing top-up") the sample is empty.

Why, from the 349-item distributions (`f1_all.csv`):

| quantity | min | p10 | p50 | p90 | max |
| --- | --- | --- | --- | --- | --- |
| F1(vlm, det) | 0.000 | 0.784 | **0.913** | 0.972 | 1.000 |
| mean det_conf | 0.863 | 0.960 | 0.988 | 0.996 | 0.999 |

- **The VLM and the detector agree almost everywhere** (median word-F1 0.913). Fusion maps the
  VLM's words onto the detector's confident text and they match — there is essentially no
  "both confident, but they disagree" pool on these clean printed corpora.
- **`det_conf` ≥ 0.80 is a no-op filter here** (min mean-conf 0.863): PaddleOCR is confident on
  every FUNSD/SROIE page, so the only thing that pushes F1 < 0.5 is the VLM producing *nothing*.
- **The only two F1 < 0.5 items are VLM non-reads** — empty `vlm_reading`, confident det_text —
  correctly excluded by the `vlm_reading ≥ 20 chars` non-trivial gate (there is no reading to
  triage; this is a *failure mode*, not a disagreement):

  | dataset / id | F1 | vlm_chars | mean det_conf | det_text[:60] |
  | --- | --- | --- | --- | --- |
  | funsd / 80707440_7443 | 0.0 | 0 | 0.988 | `LEAD-IN: New York Football Fans ALTERNATIVES: Los Angeles…` |
  | sroie / X51005719863 | 0.0 | 0 | 0.973 | `SEN LEE HEONG RESTAURANT CD NO.(002083199-T) GST ID NO…` |

  These are the same two `vlm_empty` items D1 reports (ungated ≈ gated by construction — both fall
  back to det_text). They are instances of the **silent VLM-failure** class flagged in review_03
  (reader returns empty, pipeline silently uses det_text). Logged here; the fail-loud fix is
  already a roadmap item, out of D3's scope.

## Verdicts

Four-bucket classification (VLM-wrong / det-wrong / both-wrong / reference-fault): **N/A — no
qualifying items to classify.**

## What it means

- **Consistent with D1, not contradicting it** — so escalation tripwire (d) ("D3 verdicts
  contradicting the D1 story") does **not** fire. D1 found the gated overlay shares 94.4% of its
  words with the ungated reading; D3 explains *why* — the two sources agree, so gating changes
  content only marginally, and the gated-vs-ungated char-insertion gap D1 saw cannot be
  hallucination divergence (there is none) — it is reading-order + det-fragment, as D1 diagnosed.
- **Divergence triage is a noisy-corpus tool.** It was productive on the degraded/blank
  OCR-Quality 1000-set (`ocrq_full/adjudication.md`); on clean FUNSD/SROIE gold the engines agree
  and the pool is empty. That contrast is itself the result — not a shortfall of the run.
- **No rule relaxation.** Lowering the F1 bar or dropping the non-trivial gate to manufacture 20
  items would unpin the rule and select noise; the honest report is zero.

## Artifacts

- `eval_out/divergence_triage/f1_all.csv` — every item's F1, mean det_conf, sizes, candidate flag
- `eval_out/divergence_triage/{sample.json, triage_table.csv}` — empty (no candidates), present for schema
- `eval_out/divergence_triage.py` — the runner (pinned rule, seed=1, no top-up)
