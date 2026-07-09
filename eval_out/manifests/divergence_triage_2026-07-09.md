# Run manifest — Evidence-plan stream D3, divergence triage

*[evidence_plan.md](../../Docs/dev_notes/evidence_plan.md) stream D3 — the qualitative anchor for
the D1 insertion numbers: pages where a confident detector and the VLM strongly disagree, sampled
and gold-adjudicated. Universe = the 349 archived stream-A VLM docs. NO reader / NO VLM compute.
Runner `eval_out/divergence_triage.py`; verdicts `eval_out/divergence_triage/verdicts.md`.
**Pending Luke's certification.***

## Result — the pinned rule selects ZERO candidates

Pinned selection: `F1(vlm_reading, det_text) < 0.5` AND mean `det_conf` ≥ 0.80 AND both sides
non-trivial (`vlm_reading` ≥ 20 chars, ≥ 3 det segments). **No FUNSD/SROIE item qualifies**, so
per the pin the sample is empty — **no threshold-relaxing top-up** (that would unpin the rule).

| quantity (n=349) | min | p10 | p50 | p90 | max |
| --- | --- | --- | --- | --- | --- |
| F1(vlm, det) | 0.000 | 0.784 | **0.913** | 0.972 | 1.000 |
| mean det_conf | 0.863 | 0.960 | 0.988 | 0.996 | 0.999 |

- **The VLM and the detector agree almost everywhere** (median word-F1 0.913): fusion maps the
  VLM's words onto the detector's confident text and they match. There is essentially no
  "both confident, but they disagree" pool on these clean printed corpora.
- **`det_conf ≥ 0.80` is a no-op filter here** (min mean-conf 0.863) — PaddleOCR is confident on
  every page, so only the VLM producing *nothing* pushes F1 below 0.5.
- **The only two F1 < 0.5 items are VLM non-reads** (empty `vlm_reading`, confident det_text:
  `funsd/80707440_7443`, `sroie/X51005719863`), correctly excluded by the non-trivial gate — a
  *failure mode*, not a disagreement. They are the same two `vlm_empty` items D1 reports, and are
  instances of the review_03 **silent VLM-failure** class (reader returns empty → pipeline silently
  uses det_text). Logged; the fail-loud fix is an existing roadmap item, out of D3's scope.

## Verdicts

Four-bucket classification (VLM-wrong / det-wrong / both-wrong / reference-fault): **N/A — no
qualifying items.**

## What it means

- **Corroborates D1 — tripwire (d) does NOT fire.** D1 found the gated overlay shares 94.4% of its
  words with the ungated reading; D3 explains why — the sources agree, so gating changes content
  only marginally, and D1's gated-vs-ungated char-insertion gap **cannot** be hallucination
  divergence (there is none): it is reading-order + det-fragment, as D1 diagnosed.
- **Divergence triage is a noisy-corpus tool.** Productive on the degraded/blank OCR-Quality
  1000-set (`ocrq_full/adjudication.md`); on clean FUNSD/SROIE gold the engines agree and the pool
  is empty. That contrast is the result, not a shortfall.

## Artifacts

- `eval_out/divergence_triage/f1_all.csv` — every item's F1, mean det_conf, sizes, candidate flag
- `eval_out/divergence_triage/verdicts.md` — the verdict write-up (empty sample, reasoning)
- `eval_out/divergence_triage.py` — the runner (pinned rule, seed=1, no top-up)
