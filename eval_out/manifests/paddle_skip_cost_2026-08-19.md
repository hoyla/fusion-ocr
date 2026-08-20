# paddle_skip_vlm counterfactual pricing — 2026-08-19

**Runner:** `eval_out/paddle_skip_cost.py` (committed with the feature; decision rule
pre-registered in its docstring before any number was produced). **Machine:** Luke's laptop
(the noise floor is zero — stream G — so single runs bind). **Data:** the 349 archived
stream-A pre-fusion snapshots (`doc.06-table_read.json`, copied from CORSAIR 2026-08-19;
the stream-E integrity check verified this copy reproduces the committed stream-A columns
on 16/16 probed items). **Method:** per threshold T, any page with mean PaddleOCR
confidence ≥ T has `vlm_reading` blanked pre-fusion (= what the skip does), re-fused at
baseline, gated text + band placement scored vs gold. Zero VLM compute.

## Result: the pre-registered rule says STAY DISABLED — no T clears either bar

| T | ds | n | pages skipped | Δ word_recall | Δ word_precision | Δ band placement |
| --- | --- | --- | --- | --- | --- | --- |
| 0.90 | funsd | 199 | 100.0% | −0.0307 | +0.0175 | −0.0258 |
| 0.90 | sroie | 150 | 99.3% | −0.0777 | −0.0094 | −0.0604 |
| 0.95 | funsd | 199 | 93.5% | −0.0260 | +0.0195 | −0.0211 |
| 0.95 | sroie | 150 | 92.7% | −0.0737 | −0.0060 | −0.0564 |
| 0.98 | funsd | 199 | 55.8% | −0.0083 | +0.0124 | −0.0074 |
| 0.98 | sroie | 150 | 50.7% | −0.0399 | −0.0006 | −0.0307 |

(Bars: Δ > −0.005 on recall AND band per corpus; ≥ 20% skipped. Even the strictest arm,
T=0.98, costs 4 recall points and 3 placement points on receipts.)

## The finding that matters beyond the knob

**PaddleOCR is confidently wrong on a real tail.** Worst skipped items (Δ recall vs own
baseline, mean_conf in brackets): sroie/X51006387953 **−0.40** (0.97),
sroie/X51005663311 −0.29 (0.97), sroie/X51006619760 −0.26 (0.95),
sroie/X51009008095 −0.23 (**0.986**), funsd/88057519 −0.19 (**0.992**). Mean detector
confidence does not discriminate readability on this corpus class (at T=0.90 it would skip
~100% of pages) — the "high confidence ≈ VLM adds nothing" assumption fails per-item, hard.

Two corollaries:

1. **Strategic:** the VLM tier isn't only for handwriting/degraded — it is worth +4–8
   recall points *on clean, high-confidence print receipts*. First direct pricing of the
   "why not deterministic-only on easy pages" question.
2. **Follow-up worth registering:** the Apple Vision cheap tier (`apple_vision_skip_vlm`
   = 0.92, enabled) was adopted on much smaller evidence. The same counterfactual method
   now exists; running it for the Vision tier (Vision confidences, macOS) would either
   confirm that threshold or catch the same confidently-wrong tail there. (Vision's
   confidence distribution is a different engine's — this result does not transfer
   automatically in either direction.)

**Known scope limits (stated pre-run):** FUNSD/SROIE are scanned print; the handwriting
interaction is not measurable here (IAM stream-B CSV carries no det_conf) and stays an
explicit caveat. Skip% is page-count-based, a proxy for VLM-calls-saved.

**Disposition:** `paddle_skip_vlm` ships in the codebase, default 0.0 (disabled), and this
manifest is the evidence that it should stay so on current corpora. Any future enable PR
must carry a re-run of this table (the runner resumes/extends cheaply). Raw per-item rows:
`eval_out/paddle_skip_cost/results.csv`.
