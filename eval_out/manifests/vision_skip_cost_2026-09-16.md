# apple_vision_skip_vlm counterfactual pricing — 2026-09-16

**Runner:** `eval_out/vision_skip_cost.py` (decision rule pre-registered in its docstring before
any number was produced; per-item rows `eval_out/vision_skip_cost/results.csv`). **Machine:**
Luke's laptop (the noise floor is zero — stream G — so single runs bind). **Data:** the 349
archived stream-A pre-fusion snapshots (`stream_a_vlm/out/<id>/doc.06-table_read.json`:
FUNSD 199 full + SROIE 150), whose layout regions and VLM reading are reused — the reading is
engine-independent (same image). **Method:** the page is re-boxed with **Apple Vision** through
the production path (`OcrDet` under `prefer_apple_vision`, 150 DPI, derived PDF regenerated
exactly as the archived ingest did); then per threshold T any page with mean Vision confidence
≥ T has `vlm_reading` blanked pre-fusion (= what the skip does), fused at the certified
constants (0.34 / 0.80), and the gated text + band placement scored vs gold. T=0 is the no-skip
Vision baseline. Zero VLM compute. This is the follow-up the paddle_skip pricing registered on
2026-08-19: the Vision tier shipped **enabled** at 0.92 before its skip had been priced —
"benchmarked before enabling" referred to Vision's recognition quality, not to what replacing
the reading with its text costs.

## Result: the pre-registered rule says DISABLE — every T fails both bars, on both corpora

| T | ds | n | pages skipped | Δ word_recall | Δ word_precision | Δ band placement |
| --- | --- | --- | --- | --- | --- | --- |
| **0.92 (the enabled default)** | funsd | 199 | 94.5% | **−0.0740** | −0.0294 | **−0.0483** |
| **0.92 (the enabled default)** | sroie | 150 | 94.7% | **−0.0737** | −0.0292 | **−0.0632** |
| 0.95 | funsd | 199 | 85.9% | −0.0660 | −0.0262 | −0.0397 |
| 0.95 | sroie | 150 | 82.0% | −0.0581 | −0.0229 | −0.0490 |
| 0.98 | funsd | 199 | 47.2% | −0.0249 | −0.0119 | −0.0159 |
| 0.98 | sroie | 150 | 43.3% | −0.0284 | −0.0106 | −0.0222 |

(Bars: Δ > −0.005 on recall AND band placement per corpus; ≥ 20% skipped. The Paddle skip was
rejected at −0.026/−0.074 recall; the enabled Vision default costs **−0.074 on both corpora**.)

## What the tail says

- **The tier fires almost always.** Mean Vision confidence per page: median 0.978, p10 0.936 —
  94.6% of pages clear 0.92, 45.6% clear 0.98, 19.5% score exactly 1.0. So in a Vision-routed
  deployment the VLM is skipped on ~19 pages in 20, and the product on those pages is Vision's
  `det_text`.
- **Confidence does not discriminate readability.** On the 330 pages skipped at T=0.92 the
  median recall loss is −0.066; 61.5% lose more than 5 points, a third more than 10, and only
  6.4% gain anything. Pages Vision rates ≥ 0.98 still lose 6.4 points on average (n=159); the
  0.92–0.98 band loses 8.9 (n=171). Worst skipped items: funsd/88057519 **−0.32** (conf 0.976),
  sroie/X51006387931 −0.32 (0.984), funsd/83624198 −0.32 (0.978), sroie/X51006619760 −0.27
  (0.956), funsd/01197604 −0.26 (0.982) — several of the same pages that sank the Paddle skip.
- **Same finding as 2026-08-19, second engine:** a deterministic recogniser is confidently wrong
  on a real tail, and its mean confidence cannot find that tail. The VLM read is worth ~7 recall
  points on clean scanned print *whichever* engine boxed the page.

## Context (side output): Vision-routed vs Paddle-routed, no skip, same VLM reading

| ds | recall Vision / Paddle | band Vision / Paddle | precision Vision / Paddle |
| --- | --- | --- | --- |
| funsd | 0.7823 / 0.8137 (**−0.031**) | 0.5774 / 0.5637 (+0.014) | 0.8073 / 0.8162 |
| sroie | 0.9462 / 0.9419 (+0.004) | 0.9331 / 0.9241 (+0.009) | 0.8982 / 0.8996 |

Consistent with stream A's deterministic-only split (Paddle leads on forms, tie on receipts):
with the VLM reading fused on, Vision geometry gives up ~3 recall points on FUNSD and is at
parity on SROIE, with slightly better band placement on both. So `prefer_apple_vision` is a
defensible *geometry* choice for the airgap/no-server tier on receipts-class documents — the
skip is the part that isn't.

## Recommendation (Luke decides; the change lands as its own PR carrying this table)

Set `apple_vision_skip_vlm = 0` (disabled), matching `paddle_skip_vlm` — NB the code needs the
same `threshold <= 0 → never skip` guard the Paddle tier has; without it 0 means *always*
skip (caught while writing the decision PR) — and reword the "cheap tier: Vision's text IS
the reading" framing in `routing.md` / `configuration.md` /
`config.example.toml`: Vision stays available as the deterministic **geometry** engine
(`prefer_apple_vision`), never as the reading. The re-add trigger is the same as for the Paddle
skip — a per-page readability signal that actually discriminates (mean detector confidence is
not it), or a deployment where compute forbids the VLM and the operator accepts ~7 recall
points on print knowingly.

## Scope limits (stated up front in the runner)

FUNSD/SROIE are scanned **print**; the handwriting interaction is not measurable here (IAM rows
carry no Vision confidence). Vision's per-observation confidences are coarse (many pages at
exactly 1.0), which is part of why the threshold can't separate pages. Single machine, single
run (zero noise floor). Not measured: the latency saving the skip buys — irrelevant while it
costs this much.
