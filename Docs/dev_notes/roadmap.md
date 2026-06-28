# Roadmap

Forward-looking work, roughly by priority. Shipped capabilities are logged in
[done.md](done.md); the rationale behind the architecture is in
[motivation_and_strategy.md](motivation_and_strategy.md).

Ordering principle (principle 6, *look before infra*): the scale-triggered items below are
deliberately **not** built yet — the interfaces are shaped so they drop in when the load is
real, not before.

## Now / near-term

- **Expand the hand-labelled eval set (non-Thai).** Scaffold + first baseline shipped (see
  [done.md](done.md)): `--labels` manifest with multi-page spans, `--no-vlm` to isolate the
  deterministic engine, guide in [eval-labelling.md](../eval-labelling.md). Four hard pages are
  labelled, with the first measured numbers (VLM ~0.92 aggregate recall, handwriting ~0.95). The
  born-digital ~95% is only a difficulty floor, so the work now is **breadth**: label more
  **non-Thai** hard pages (handwriting, degraded scans, the rotations / redactions in test set 1).
  Thai ground truth is parked under Next (no Thai reader); the big-corpus alternative is the
  3rd-party SROIE/FUNSD set under *Later → Input formats*, once images can be ingested.
- **VLM client hardening** (pulled up from Next — no external dependency, the most actionable
  robustness work; [review_01](review_01_260627.md)): a `max_tokens` cap (a pathological page can
  otherwise generate until the 600s timeout — latency/cost on a shared GPU); retry/backoff on a
  transient 5xx (today a 503 becomes a silent empty read → `det_text` fallback); reuse one
  `httpx.Client` (keep-alive) instead of one per page; and **JPEG** rather than PNG base64 (a
  150-DPI page is multi-MB, +33% for base64 — matters on the remote-reader / in-VPC path).

## Next

- **Tables:** multi-level-header semantics for `find_tables`; cross-validate `find_tables` vs
  the vision grid; cleaner per-cell content on scanned tables.
- **Thai ground truth for the eval** — *parked from near-term:* needs a Thai reader. The Thai
  scan was dropped from the labelled set for this reason; pick it up when a reader is available.
- **Thai overlay search reliability** — *parked from near-term:* combining vowels / tone marks,
  NFC vs NFD; reading is solid, reliable highlight is the gap. Also needs a Thai reader to verify.
- **Word-level overlay subdivision** — *parked from near-term, behind a trigger:* revisit only if
  **line-level** highlighting proves inadequate in real reporter use. Honest word boxes would come
  from the Apple Vision per-word API / PyMuPDF `words` (never proportional splitting — that
  manufactures precision), but it adds an alignment step over fusion that can itself err, and
  per-word geometry isn't always available (PaddleOCR detection is line-level). Don't build the
  refinement before the simpler thing is shown wanting (*look before infra*).
- **"Giant rejects" eval at corpus scale** — *parked from near-term:* needs more reject documents
  than test set 1 holds. The *small* old-vs-new comparison is **done** (Mandelson letter:
  tesseract ~0.10 vs our ~0.95 word recall — see [done.md](done.md)); the corpus-scale version
  waits on a larger reject set.
- **Reading order:** actually measure it. The hand-labelled transcripts (in true visual order)
  are now that oracle — CER against them folds in reading order, unlike the born-digital text
  layer — so what remains is scoring multi-column / complex layouts against the labelled set as
  it grows.
- **Qwen3.5-VL re-test** — trigger now met. An earlier look called it a statistical tie with
  Qwen3-VL-8B; a trusted MLX **vision** build now exists (`mlx-community/Qwen3.5-9B-MLX-4bit`,
  apache-2.0, `image-text-to-text`, on HF since 2026-03-02), so a re-test is actionable. Use the
  `mlx-community` build or self-convert from `Qwen/Qwen3.5-9B` — the Qwen3.5 VLM space has many
  abliterated / "Heretic" / uncensored MLX-VLM variants; never run a tampered build on
  confidential material.
- **Result push for non-airgap tiers:** an optional webhook / callback on completion. The
  sealed (airgap) tier stays poll-only by construction — the process can't dial out — so this
  is a tier-gated enhancement, never the default.

## Later — beyond MVP

Capability beyond the MVP target:

- **Rotated-page tables** — the table-structure and focused table-read stages currently skip
  rotated pages ([review_02](review_02_2602627.md) #8). Add support when rotated scans turn
  up in the corpus.

Input formats — an **ingest adapter** that normalises any input to a PDF, after which the
existing pipeline runs unchanged (PDF is the identity case). The original is kept as the
**canonical source**; the PDF is a derived, provenanced artifact. This is the same workflow
as Giant's built-in processor, so reuse its approach for parity.

- **Images (PNG / JPEG / TIFF)** — PyMuPDF opens and `convert_to_pdf`s them (multi-page TIFF
  split via Pillow); they flow straight through the scanned-page path. The most common
  non-PDF input we receive.
- **Office (.docx / .xlsx / .pptx)** — convert via LibreOffice headless
  (`soffice --headless --convert-to pdf`). The existing mixed-content composition then
  separates content *for free*: digital body text → text layer (not OCR'd, exact), embedded
  charts / tables / scans → figure/table regions → VLM read. Scope is the **image-borne**
  text; a pure-text doc is an upstream concern (OCR is the wrong tool for already-digital
  text). Caveats: LibreOffice is a heavyweight optional `office` extra (pre-pull for the
  airgap tier); Office files are untrusted (macros — headless doesn't run them, but sandbox
  it); docx provenance is looser (drill-back is to the rendered page).

> Attach points for the adapter are already marked in the code: the API format gate
> (`api._save_upload`) and the watcher glob (`watcher.scan_once`) — both accept PDF only today.

**Test corpus already on hand for this work** (`samples/file_tests_3rdparty_01/`, gitignored —
~4.3k labelled images, 5.3 GB): a consolidated public OCR benchmark, four sources in a uniform
`category/{train,val,test}/{images,annotations}` layout, each image paired with per-line
**text + box** annotations. Once images can be ingested (the adapter above, or a lighter
eval-only image path) this becomes a recognition + geometry eval far larger than hand-labelling.
What's usable, and the traps:

- **SROIE** (`invoice/`, 973 receipts) and **FUNSD** (`form/`, 199 scanned forms) — genuine human
  GT, in-domain document images. **Use these first.** Ignore their key-value / entity / linking
  labels (that's downstream analysis, not this tool — principle 1); only the text+box GT applies.
- **IAM** (`document/`, 1539 handwriting pages) — the headline-case prize, BUT the bundled json
  annotations carry a per-line `confidence`: they're an OCR engine's *output*, not human
  transcriptions, so scoring our OCR against them is **circular**. Source IAM's original
  transcriptions before using it as ground truth.
- **Total-Text** (`real_life/`, ~1554) — scene text (photos / signage), arguably **out of the
  document domain**; likely set aside.

Each source needs a small annotation→eval-reference loader (FUNSD/SROIE json, Total-Text txt;
pixel coords). Fuller analysis in the `dataset-3rdparty-ocr-benchmark` session memory.

Scale-triggered — don't build until the load is real (principle 6, *look before infra*):

- **Distributed queue adapter** (ElasticMQ / SQS, on-estate) implementing the `JobStore`
  method surface — when workers span machines or need shared durable state.
- **Object-store adapter** (Garage on-estate, or S3-in-VPC for a less-sensitive tier) behind
  `storage.py` — when the local filesystem stops sufficing.
- **Worker options:** an in-process worker for single-box convenience; a multi-worker pool
  (the atomic claim already makes this safe).
- **API at scale:** rate limits beyond the upload cap; richer observability / metrics.

_The seams (`JobStore`, `storage.py`, the OpenAI-compatible reader endpoint) keep the
scale-triggered swaps config-deep, not rewrites._
