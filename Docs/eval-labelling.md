# Hand-labelling the eval set

## Why this exists

The automatic eval ([`src/fusion_ocr/eval/`](../src/fusion_ocr/eval/)) scores the OCR path
against **born-digital** pages — a digital PDF already carries its correct text, so we
render it, OCR it, and compare. That gives a real recognition number (~95% recall) with no
hand-work, but only on *clean rendered* pages, which **flatter** the system. It says nothing
about the cases the tool actually exists for: **degraded scans and handwriting**, which
carry no machine-readable truth.

For those, the only ground truth is a person reading the page. This is that set: you type
what a handful of hard pages actually say, and the harness scores the pipeline against your
transcripts. It's a small amount of one-off human work that turns "trust me, it reads
handwriting" into a measured number — and a regression guard for the future.

A bonus: because you transcribe in **true reading order**, these labels are also the only
honest reading-order oracle we have (the born-digital text layer is in content-stream order,
which scrambles multi-column pages).

## What you actually do

Everything lives in [`eval_labels/`](../eval_labels/). The manifest
`eval_labels/labelset.json` is already pointed at five hard pages; each has an empty `.txt`
file waiting for its transcript. **Your job is to fill in those `.txt` files.**

1. **Open the page.** For each entry in `eval_labels/labelset.json`, open the `pdf` at the
   given `page` (a **0-based** index — `page: 0` is the first page) in any PDF viewer. A
   document that runs over several pages (a 2-page letter, a multi-page form) uses `pages`
   instead — e.g. `"pages": [183, 184]` — and its transcript covers the whole span in
   reading order.

2. **Type what you see** into the matching transcript file (e.g.
   `eval_labels/mandelson-note-handwritten.txt`). The rules that keep the score honest:
   - **Reading order:** type lines in the order a human reads them, top-to-bottom, column by
     column. Your eye is the oracle here.
   - **Transcribe, don't correct.** Copy the text as written — keep original spelling,
     casing, and obvious errors. You're recording what's on the page, not what it should say.
   - **Don't normalise scripts.** For Thai/Cyrillic/etc. type the real Unicode characters.
   - **Redactions:** write `[REDACTED]` for a blacked-out span, so recall isn't punished for
     ink no one can read.
   - **Skip pure decoration** (logos, page furniture) unless it's real content.
   - Leave a file **empty** to mark that page "not done yet" — it's reported as TODO, never
     scored, so you can label incrementally.

3. **Run the eval.** Handwriting and degraded scans need the VLM reader, so start it first
   (e.g. the MLX server) and *don't* pass `--apple-vision`:

   ```bash
   python -m fusion_ocr.eval --labels eval_labels/labelset.json
   ```

   You'll get a per-page scorecard plus a micro-averaged aggregate. Run it again whenever you
   add a transcript or change a prompt/model — that's the regression guard.

## Reading the scorecard

| Column | Means |
| --- | --- |
| `recall` | fraction of your words the pipeline recovered — **recognition completeness** |
| `prec` | fraction of the pipeline's words that are real — `1 − prec` is the **hallucination** rate |
| `CER` / `WER` | character/word error including reading order — trust most on single-column prose |
| `refchars` | size of your transcript (how much the page weighs in the aggregate) |

Recall/precision isolate recognition; CER/WER fold in order too. A high recall with a high
CER means "read correctly but mis-ordered", not "misread".

## Adding more pages later

Add an entry to `eval_labels/labelset.json` (`id`, `pdf`, `page`, `transcript`, optional
`note`), create the `.txt`, and transcribe. The live manifest and the transcripts are
**gitignored** — they quote source material and the repo is public — so they stay on this
machine; `labelset.example.json` is the committed template that documents the format.
