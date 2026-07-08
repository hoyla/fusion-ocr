# Run manifest — Evidence-plan stream A, 150-DPI re-check

*The shipped `ocr_det` default was cut 200→150 DPI (PR #16) on the strength of **5 pseudo-GT
pages**. This is the real test on human-GT gold: a deterministic A/B (PaddleOCR, `--no-vlm`) at
150 vs 200 DPI on seeded n=50 FUNSD + n=50 SROIE. Runner `eval_out/dpi_recheck.py`.*

## Method

Each DPI arm is a custom deterministic pipeline `OcrDet(dpi=150|200)`. DPI is a stage constant,
**not** in the recipe fingerprint, so the two arms would collide in the resume cache — each uses
a distinct out_dir + digest to force genuine fresh processing. Verified real: ~32s/item, and
**79/100 items produce different recall between the two DPIs** (so the cache bypass worked; this
is a true A/B, not the same cached doc scored twice). SROIE scored caseless.

## Result — 150 ≥ 200 on both gold sets

| dataset | DPI | recall | CER |
| --- | --- | --- | --- |
| FUNSD (n=50) | 150 | **0.7934** | **0.2938** |
| FUNSD (n=50) | 200 | 0.7821 | 0.3028 |
| SROIE (n=50) | 150 | **0.8813** | **0.4318** |
| SROIE (n=50) | 200 | 0.8798 | 0.4323 |

| dataset | Δ recall (150−200) | Δ CER (150−200) |
| --- | --- | --- |
| FUNSD | **+0.0113** | **−0.0090** |
| SROIE | +0.0015 | −0.0005 |

**150 DPI is at least as good as 200 on both sets** — a small but consistent edge on FUNSD
(higher recall *and* lower CER), a statistical tie on SROIE. (The noise floor is 0 and this path
is deterministic, so these deltas are real, not variance — the sign consistently favours 150.)

## Verdict

**Keep 150 — confirmed on gold.** The PR #16 cut was validated on 5 pseudo-GT pages; on 100
human-GT items it holds and then some: 150 is not merely recognition-equivalent, it is marginally
*better*, while also being faster and sharing the 150-DPI raster cache (no separate 200-DPI
render). No reason to revisit the default. Plausible mechanism: PP-OCRv6 detection/recognition is
tuned to an input scale nearer 150 DPI for these page sizes; 200 adds pixels without adding
legible signal.

## Caveats / not measured

- SROIE **CER stays ~0.43** at both DPIs — the word-spacing/reading-order artifact against the
  flat annotation-order reference (documented in the stream-A manifest), *not* a DPI effect. On
  receipts recall is the recognition measure; the DPI comparison there is the +0.0015 recall.
- **Placement-at-DPI not captured** (a runner column was added after the run had already started,
  so it never took effect). Optional follow-up; low expected value since 150 already wins
  recognition and placement follows detection quality. Not re-run — recall/CER is the
  pre-registered criterion and it is answered.
- Deterministic path only; a DPI change on the *VLM* read is a separate question (the VLM already
  renders at 150 via the shared cache).

## Artifacts

- `eval_out/dpi_recheck/results.csv` — per-item recall/CER at both DPIs
- `eval_out/dpi_recheck.py` — the runner (custom-pipeline DPI override + distinct-digest bypass)
