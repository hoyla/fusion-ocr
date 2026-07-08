# Run manifest — Evidence-plan stream G, noise floor

*[evidence_plan.md](../../Docs/dev_notes/evidence_plan.md) stream G — the prerequisite for every
"X beats Y" (incl. the RapidOCR A/B): how much do the metrics move run-to-run under the identical
config, i.e. what delta is just noise? Runner `eval_out/noise_floor.py`.*

## Method

The full VLM pipeline (PaddleOCR geometry + Qwen3.5-9B-MLX-4bit reader, temperature 0.0) run
**3× on the same fixed seeded-30 FUNSD set**. Each run used a **distinct out_dir + digest** so the
recipe-fingerprint resume cache could not return an identical cached doc and fake a zero floor —
the 3 runs are genuinely independent re-processings. Verified real: each run took **~22 min
(~45s/item)** of actual VLM calls (a cache hit would be near-instant), across `out_run1/2/3`.

## Result — the floor is exactly ZERO

| metric | run 1 | run 2 | run 3 | spread |
| --- | --- | --- | --- | --- |
| word recall | 0.8328 | 0.8328 | 0.8328 | **0.00000** |
| CER | 0.1571 | 0.1571 | 0.1571 | **0.00000** |
| insertion_rate | 0.0624 | 0.0624 | 0.0624 | **0.00000** |
| placement (strict) | 0.5004 | 0.5004 | 0.5004 | **0.00000** |
| placement (band) | 0.5452 | 0.5452 | 0.5452 | **0.00000** |

**All 30 items are bit-identical across all 3 runs** — zero per-item spread on every metric. MLX
4-bit greedy decode (temperature 0.0) is deterministic run-to-run.

## Implications

- **Per the pre-registration ("if the floor is ~0, say so once and stop paying for repeats"):**
  single runs are trustworthy — **no need to repeat runs for variance**. Every delta measured
  this session (VLM vs deterministic, fused vs deterministic placement, the caseless fixes) is
  **real, not decode noise**.
- **Standing rule 4 is now sharp:** the binding constraint on "is this delta a result?" is no
  longer run-to-run variance (it's 0) but **sample size**. The cautionary Qwen3.5 switch
  (Δ0.005, n=4) fails on *n*, not on noise — the noise floor does not rescue small-*n* deltas.
- **Unblocks the RapidOCR A/B** (and PP-OCRv6 tiny/small, and the 4-bit-vs-8-bit and DPI
  comparisons): a single-run recall/CER/placement delta between engines counts as real, judged
  against a zero floor.

## Caveats

- Determinism is established for **this machine + this MLX build + this model + temperature 0.0**.
  A different MLX version, model, GPU, or any sampling (temperature > 0) can reintroduce variance
  — **re-measure the floor if any of those change** before trusting single-run deltas again.
- The deterministic (PaddleOCR / Apple Vision) path has no randomness, so its floor is 0 by
  construction; this run confirms the *VLM* path (the only plausible variance source) is also 0.

## Artifacts

- `eval_out/noise_floor/results.csv` — per-run, per-item scores (30 × 3)
- `eval_out/noise_floor.py` — the runner (distinct-digest cache bypass)
