# Run manifest — Evidence-plan stream A VLM pass + fused placement (P1)

*Full pipeline (PaddleOCR geometry + Qwen3.5-9B-MLX-4bit reader) on FUNSD (all 199) + SROIE
(fixed-seed 150). Completes stream A's open questions AND produces the headline **fused**
box-placement number the deterministic pass could only floor. 349 items, 0 errors. Runner
`eval_out/stream_a_vlm.py`; scores the ungated reading (text) and the gated segments
(placement) per standing rule 3.*

## Text — VLM vs deterministic (matched sets; SROIE caseless both sides)

| dataset | VLM recall / CER | DET recall / CER | ungated insertion |
| --- | --- | --- | --- |
| FUNSD (n=199) | **0.817 / 0.170** | 0.783 / 0.287 | 0.067 |
| SROIE (n=150) | **0.956 / 0.138** | 0.864 / 0.391 | 0.076 |

- **Q1 (does the n=16 FUNSD VLM CER 0.15 hold?) — YES.** 0.170 at n=199. The reader puts complex
  forms into near-correct reading order at scale, as the small sample suggested.
- **The receipt VLM win is real and, corrected, bigger.** The old pilot's "VLM CER win on
  receipts" survives caseless re-scoring and strengthens: VLM CER 0.138 vs det 0.391, recall
  0.956 vs 0.864. **This updates the deterministic pass's Q2:** det-vs-Vision was a *tie* on
  receipts, but with the VLM on top receipts are decisively VLM-favoured.
- `insertion_rate` is the **ungated** reading's hallucination proxy (stream D): low (~0.07) on
  both — no mass invention on these printed corpora. (Gated-vs-ungated gap = a later stream-D step.)

## Placement (P1) — the headline, and a metric-granularity lesson

The strict per-line metric (shipped in #25) gave an alarming result — fused **below**
deterministic — that turned out to be a **granularity artifact**, not a fusion failure. Diagnosis
then correction:

| dataset | metric | FUSED (VLM) | DET (paddle) | plain (fused) |
| --- | --- | --- | --- | --- |
| FUNSD | strict (per-line) | 0.431 | 0.601 | 0.829 |
| FUNSD | **band (fair)** | **0.564** | 0.538 | 0.829 |
| SROIE | strict (per-line) | 0.586 | 0.844 | 0.956 |
| SROIE | **band (fair)** | **0.924** | 0.864 | 0.956 |

**What happened.** Fusion merges PaddleOCR's ~per-line boxes into coarser line/region clusters
(worst FUNSD item: 29 fused segments for 59 GT lines; one box 73pt tall over ~5 lines). The
*strict* metric assigns each segment to a single best-IoU GT line, so a correct-but-tall box wins
one line and its other lines' words score as "misplaced". This under-credits fused output so
badly it **reverses the SROIE ranking** (strict says det 0.84 ≫ fused 0.59, though fused actually
covers the lines *better*). The **band** metric credits a word if it lands in any segment that
*covers* its line (a taller box over the right line is still a correct highlight) — the fair
measure across granularities. It still penalises true displacement (band < plain).

**What it means, honestly:**
- **Fusion places words at least as well as deterministic on both corpora** (band): better on
  receipts (0.924 vs 0.864), level on forms (0.564 vs 0.538). The click-a-claim promise holds —
  on single-column receipts fused placement (0.924) nearly equals recognition (0.956).
- **The residual FUNSD gap (band 0.56 vs plain 0.83) is NOT a fusion regression** — deterministic
  has the same gap (0.54). It is dense-2-D-form difficulty (side-by-side label/value fields,
  narrow GT boxes) plus the band metric's own limits on such layouts. Both engines place ~a
  quarter of recovered form-words outside a box cleanly covering their GT line.
- **Metric lesson (4th of the session):** the strict placement metric from #25 is valid for
  per-line/deterministic output but **confounded on fused (coarse) output** — believe the *band*
  number when comparing fused vs deterministic. #25's deterministic manifest numbers stand
  (fine granularity); this PR adds the `band=` mode + reports both.

## What this settles / doesn't

- **Settles:** VLM clearly out-recognises the deterministic path on both corpora (text); with the
  VLM, receipts are VLM-favoured; fused placement ≥ deterministic under the fair metric — the
  product's central click-a-claim mechanism does not degrade placement, and improves it on
  single-column text.
- **Doesn't settle:** dense-2-D-form placement has a real ~0.25 gap shared by both engines — a
  target for a placement-focused improvement (and a better dense-layout placement metric). SROIE
  is a seeded 150, not all 973 (VLM cost). Handwriting (stream B) and the gated-vs-ungated
  hallucination gap (stream D) are still open.

## Artifacts

- `eval_out/stream_a_vlm/results.csv` — per-item text + placement counts (strict); `out/*/doc.json`
- `src/fusion_ocr/eval/placement.py` — `band=` mode added; `--dataset X --placement` reports both
- comparison recomputed matched (caseless det SROIE; det band placement) from the committed
  stream-A deterministic docs
