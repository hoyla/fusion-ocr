# Why fusion-ocr exists, and how it's built to serve that

This is the design-rationale doc: the *motivation* behind the tool and the *strategies*
that follow from it. For usage see the [README](../../README.md); for the per-region tool
router see [routing.md](../routing.md); for the build-vs-adopt rule see
[principles.md](../principles.md); for every config setting and the job/config API see
[configuration.md](../configuration.md).

## The problem

The newsroom already has Giant for analysing documents, but its OCR (tesseract /
OCRmyPDF) fails on exactly the documents that matter most:

- **Handwriting** — the handwritten Mandelson→Lammy note that Giant couldn't read (6
  extractable characters) but a vision model transcribes cleanly.
- **Degraded scans** and **non-Latin scripts** (e.g. Thai company filings).
- **Mixed-content pages** — a machine-readable header/footer over a scanned body, where a
  naive check reports the page as "readable" and the scanned content is silently lost.
- **Complex tables and multi-column layouts** — financial statements, forms.

We call these **"Giant rejects"**: documents that went in and came out unusable. They are
the MVP target — process a dozen a day and make them searchable and quotable. The second
failing is **bounding boxes**: even when text is recovered, OCRmyPDF-style overlays are
poor, so a reporter can't reliably click a claim and see it highlighted in the page.

## The constraints that shape everything

Two non-negotiables rule out the easy answers:

1. **The material may be confidential, so everything must run on-estate.** This rules out
   the most mature OCR available — AWS Textract, Google Document AI, Azure. We are limited
   to tools we can run locally / in-VPC, and for the most sensitive material, fully
   airgapped. That constraint is *why this project exists at all*: if we could send
   documents to a cloud OCR API, we largely wouldn't need to build anything.
2. **The output must be defensible.** This is journalism: every claim a reporter makes
   from a document must be drillable back to the source — the exact box, on the exact
   page, read by a named engine/model. See the
   [seven principles](../principles.md) (ingest-broad, defensibility, never-mutate-source,
   append-only, idempotent, look-before-infra, provenance) — they are the editorial
   reason behind most of the engineering below.

## The core strategy: hybrid deterministic + VLM

The insight is that the two things we need come from two different kinds of tool:

- **Trustworthy geometry** (where the ink is) comes from a **deterministic** engine —
  PaddleOCR or Apple Vision. Boxes you can stand behind.
- **Reading the hard cases** (handwriting, degraded scans, unusual scripts) comes from a
  **vision-language model** — the thing tesseract can't do.

So we deliberately bet on *immature* tech (VLMs are non-deterministic and can hallucinate)
for the hard reading, *because* the mature tech fails there — and then we spend the rest of
the engineering making that bet **trustworthy**:

- **Fusion** distributes the VLM's reading onto the deterministic boxes at the **word level**
  (a fuzzy alignment that survives garbled handwriting — spreading a long prose line across the
  visual lines it spans — with a line-level fallback), so a searched word lands on the right
  line. The clean reading itself is never lost: `document.md` is the ungated reading view.
- **The ink-gate** is the anti-hallucination backstop: VLM text with no underlying detected
  ink is dropped. The deterministic side is the gate; the VLM cannot invent geography.
- **Provenance is retained, never overwritten** — `det_text` and `vlm_text` are both kept
  beside the chosen `best_text`; `source`/`read_by`/`superseded` record where every segment
  came from. The segment index is the audit trail.

Making an untrustworthy-but-powerful reader trustworthy enough for journalism is the part
with no off-the-shelf equivalent — it is the actual product.

## The supporting strategies

- **A toolkit, routed per region, not a monolith.** Script detection picks the right
  deterministic recogniser and the right reader per region (Apple Vision for fast on-device
  printed text, PaddleOCR for cross-platform geometry, Qwen3.5-9B as generalist reader,
  Typhoon for Thai). Geometry is always deterministic; only the reader varies. See
  [routing.md](../routing.md).
- **Searchable bbox overlay** — an OCRmyPDF-style invisible text layer, but correct, so the
  recovered text is selectable/searchable in the PDF and highlights land on the right line.
- **Born-digital ≠ scanned.** A born-digital PDF already holds exact text; we don't OCR it.
  Tables there come from PyMuPDF `find_tables` (exact), scanned tables from the vision
  table models + a focused VLM read. Tesseract's "trouble with company reports" is really
  that OCR is the wrong tool for a document whose text is already perfect.
- **Reproducibility is a feature, not an afterthought.** Pinned dependencies + a lockfile,
  a recipe-fingerprinted resume (re-run after changing a prompt actually reprocesses), and
  an eval harness that scores recognition against born-digital ground truth — so
  "trustworthy/defensible" is measured, not asserted.
- **Adopt over build.** OCR/layout/tables/reading-order are solved, hardened problems; we
  orchestrate trusted tools (PaddleOCR PP-DocLayoutV2 + SLANeXt, Apple Vision, the VLMs,
  PyMuPDF) and build only the connective tissue above. See [principles.md](../principles.md).

## Deployment shape

A **standalone sidecar** (not embedded in Giant): other tools (Giant, the transcription
engine, fuel-finder) call it via a job API. The reader endpoint is an OpenAI-compatible
seam, so the runtime is a free variable — local MLX on Apple Silicon now, an in-VPC vLLM
GPU later, **config-only** to switch. Airgap mode seals the process (no egress) for the
most sensitive tier.

## What this tool deliberately does *not* do

It produces faithful, provenanced text + structure — it does **not** do downstream
*analysis* or *understanding* (key-value form extraction, entity linking, "what's the
story"). Per principle 1 (ingest broad, analyse second), that belongs in the consumers,
working off the structured, drillable output this tool emits — not baked into extraction.
