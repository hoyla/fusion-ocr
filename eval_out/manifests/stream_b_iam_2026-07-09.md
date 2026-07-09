# Run manifest — Evidence-plan stream B, IAM handwriting beyond n=1

*[evidence_plan.md](../../Docs/dev_notes/evidence_plan.md) stream B — does the handwriting
capability (proven on the single Mandelson letter, recall ~0.95) generalise? IAM's 1,539 pages sat
unusable because the bundled per-image JSON is an OCR engine's output (circular). With the FKI
**human** transcriptions sourced (`ascii/lines.txt`, 2026-07-09) the `datasets.iam_*` adapter pairs
every image to its human reading. Runner `eval_out/stream_b_iam.py`; seeded n=100 across splits
(seed=1), both engines scored against the human transcription, case-sensitive.*

## Method note — crop to the handwriting (the n=1 smoke caught a confound)

An IAM "form" is not bare handwriting: it shows a printed header + the printed prompt the writer
copied, **above** the handwritten version (verified by eye, `a01-020u`). Scoring a full-page OCR
against the handwriting-only reference double-counts the text (the printed prompt hands over every
word for free — recall inflates, CER blows past 1.0). So each image is cropped to the union of its
handwritten line boxes (`datasets.iam_hw_bbox`, from `lines.txt`) before OCR — the eval measures
**handwriting recognition**, not the printed prompt. Adapter + crop are unit-tested (11 tests).

## Result (n=100; 0 errors; case-sensitive)

| engine | recall (raw) | recall (**punct-norm**) | medCER | precision | mean insertion |
| --- | --- | --- | --- | --- | --- |
| **VLM (Qwen3.5-9B-4bit)** | 0.723 | **0.955** | **0.035** | 0.845 | 0.001 |
| deterministic (PaddleOCR) | 0.432 | 0.557 | 0.150 | 0.640 | 0.006 |

- **The headline capability generalises.** Punctuation-normalized VLM recall is **0.955** at n=100 —
  essentially the Mandelson single-letter number (0.95) reproduced at scale. medCER **0.035** (a
  ~4.3× lower character error than the deterministic path's 0.150) says most pages read
  near-perfectly; mean insertion 0.001 (no mass invention).
- **The VLM decisively out-reads the deterministic engine on cursive** — punct-norm recall 0.955 vs
  0.557, medCER 0.035 vs 0.150. PaddleOCR is the geometry/anchor engine, not a handwriting reader
  (as expected).
- **Why report punctuation-normalized recall.** IAM's `lines.txt` tokenises punctuation as separate
  words (`up` `.`), which the VLM writes attached (`up.`) — so *raw* word recall (0.723) is deflated
  by a tokenisation convention, not by missed content. Stripping non-word characters on both sides
  (the punct-norm column) removes the artifact; medCER (char-level) corroborates it independently.

## What this settles / doesn't

- **Settles:** VLM handwriting recall generalises beyond n=1 — 0.955 on 100 unseen IAM pages, ~= the
  Mandelson demo. The handwriting claim is no longer a single anecdote.
- **Interpretation guard (from the plan):** IAM is **clean, ruled, English** handwriting — a
  **FLOOR** for the claim, not proof on degraded FOI material. The number to quote is "≥0.95 on
  clean handwriting at n=100", not "0.95 on any handwriting".
- **Doesn't settle:** degraded / non-English / historical hands; the deterministic path's low
  handwriting recall is by design (geometry engine) and not a target.

## Caveats

- 1 of 100 pages (`n03-082`) skipped the VLM (cheap-tier / short-circuit, `t_vlm`=0) — negligible
  at aggregate; kept in the counts.
- Seeded n=100 (train 79 / test 9 / val 12, proportional to split sizes). Case-sensitive (IAM keeps
  real case). Reader = current default Qwen3.5-9B-4bit (stream F's Qwen3.6-35B-A3B candidate not
  used here, to answer the capability question on the shipped model).

## Artifacts

- `eval_out/stream_b_iam/results.csv` — per-item det + VLM recall/CER/insertion + counts + t_vlm
- `eval_out/stream_b_iam.py` — the runner (crops to handwriting; resumable; `main()`-guarded)
- `src/fusion_ocr/eval/datasets.py` — `iam_line_index` / `iam_hw_bbox` / `iter_pairs('iam')`;
  `tests/test_eval_datasets.py` covers them
