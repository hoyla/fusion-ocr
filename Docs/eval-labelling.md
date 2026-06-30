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
   (e.g. the MLX server) and run the default:

   ```bash
   python -m fusion_ocr.eval --labels eval_labels/labelset.json
   ```

   You'll get a per-page scorecard plus a micro-averaged aggregate. Run it again whenever you
   add a transcript or change a prompt/model — that's the regression guard.

   **Comparing engines.** Three modes compose over the same labels, so you can A/B which part
   is doing the work — no need to stop the reader server to isolate the deterministic engine:

   | command | what it measures |
   | --- | --- |
   | (default) | PaddleOCR geometry **+ the VLM reader** — the real product output |
   | `--no-vlm` | **PaddleOCR** recognition alone (drops the VLM stages) |
   | `--no-vlm --apple-vision` | **Apple Vision** recognition alone (on-device) |

   `--no-vlm` removes the VLM stages from the pipeline entirely, so the recovered text is the
   deterministic recogniser's own — the honest baseline the VLM is improving on. (PaddleOCR is
   the deterministic spine; Apple Vision is the fast on-device tier, not a stronger recogniser
   — on this set it scores lower than PaddleOCR.)

## Reading the scorecard

| Column | Means |
| --- | --- |
| `recall` | fraction of your words the pipeline recovered — **recognition completeness** |
| `prec` | fraction of the pipeline's words that are real — `1 − prec` is the **hallucination** rate |
| `CER` / `WER` | character/word error including reading order — trust most on single-column prose |
| `refchars` | size of your transcript (how much the page weighs in the aggregate) |
| `sCER` / `sRcl` | the same CER and recall, but for the **searchable** text — what `find()` hits in the *output PDF*, not the reading view |
| `via` | where that searchable text lives: `ovl` = the OCR overlay, `txt` = the source PDF's own text layer, `MISS` = nothing findable |

Recall/precision isolate recognition; CER/WER fold in order too. A high recall with a high
CER means "read correctly but mis-ordered", not "misread".

**Reading view vs searchable text.** The left columns score `document.md` (the clean
reading — what we *recovered*). The `s*` columns score what a reader's find/search actually
hits in the output PDF: `overlay.pdf` when one was built (it carries the source text layer
*plus* the OCR overlay), otherwise the source PDF itself, whose text layer is still
searchable. The `via` column says which:

- `ovl` — an OCR overlay carries it. This is the case the fusion fix is about. Usually it
  matches the reading, but where fusion can't confidently anchor a line to a detected box
  (rotated dense print, badly garbled handwriting) the overlay degrades *honestly* and
  `sRcl` drops below `recall`. **That gap is the number to watch** — the cost of not
  smearing the reading onto guessed positions.
- `txt` — no overlay was added because the page's own text layer already carries the
  content (a born-digital page, or a mixed scan whose exact text layer beat the OCR). Adding
  an overlay would double search hits, so we don't; `sRcl` here is the text layer's recall.
- `MISS` — an OCR page that produced no searchable text at all. A genuine hole; `sRcl` is 0.

## Multi-column reading order, the cheap way (`render: true`)

Hand-transcribing a dense multi-column page is a lot of typing — and the corpus has no strong
scanned multi-column *prose* to begin with (`TestPDFs_01`'s multi-column docs are born-digital;
its scans are single-column / forms / handwriting). There's a shortcut that gives a
multi-column **scan** with a **100%-certain reading order** and almost no transcription:

1. Pick a **born-digital** multi-column page (e.g. a two-/four-column annual-report narrative in
   `TestPDFs_02`). Its exact text is known — no recognition guesswork.
2. Add a label with **`"render": true`**. That renders the page to an image-only PDF (drops the
   text layer) before processing, so the pipeline must **OCR** it — a genuine scan.
3. **Seed** the transcript from the page's own text (copy from a viewer / `get_text`), then do
   the one human step: **certify the reading order** — put the columns in the order you read them
   (column 1 top-to-bottom, then column 2, …). You're checking order, not typing words.

Because the reference text is exact, recognition drops out as a confound: a high `recall` with a
higher `CER` is then **pure reading-order error**. Pick pages where order is unambiguous — clean
multi-column prose, where the content-stream order is often already correct (just confirm it).
Avoid infographics / designed pages (their content-stream order is scrambled and there's no
single "right" reading order to certify). Pages with a side-bar or call-out box are a good
*advanced* target (a Z-order trap), but certify the side-bar's place in the order deliberately.

## Adding more pages later

Add an entry to `eval_labels/labelset.json` (`id`, `pdf`, `page`, `transcript`, optional
`note`, optional `render`), create the `.txt`, and transcribe. The live manifest and the
transcripts are **gitignored** — they quote source material and the repo is public — so they
stay on this machine; `labelset.example.json` is the committed template that documents the format.
