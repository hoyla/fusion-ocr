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
  3rd-party SROIE/FUNSD set (now wired — see *Later → Input formats*).
- **Reading order: actually measure it.** *(Largely addressed for scanned forms — 2026-06-30.)*
  Two oracles now fold reading order into CER: the hand-labelled transcripts (true visual order),
  and — the unlock for scale — **FUNSD's per-line boxes reconstructed into reading order**
  (`eval/datasets.py` `_reading_order`; `--dataset funsd`), giving ~200 complex scanned forms
  without hand-labelling. First numbers (n=16, stable vs n=6): reordering to reading order
  roughly **halves** the deterministic CER (0.44→0.25) and the VLM lands at reading-order CER
  **0.15** (recall 0.84) — so the learned order head and the VLM both put complex forms into
  near-correct reading order, the residual being recognition not order. Method + numbers + caveats in
  [reading_order_measurement.md](reading_order_measurement.md). **Remaining gap:** newspaper-style
  multi-column *prose* on a *scanned* page (FUNSD is forms; born-digital multi-column prose is
  already testable via the content-stream harness) — that still wants a hand-labelled page. See
  [handover_reading_order.md](handover_reading_order.md).
- **Revisit reading order with hand-labelled multi-column *prose*.** The FUNSD oracle covers
  complex scanned *forms*; the open case is **scanned running text in multiple columns**, where
  reading order genuinely changes meaning and the geometric proxy could plausibly disagree with
  both the human and the pipeline.
  **Cheapest path — the born-digital-render trick (now built + proven, `render: true`):** take a
  born-digital multi-column page (e.g. a `TestPDFs_02` annual-report narrative), render it to a
  scan (drops the text layer → the pipeline must OCR it), and use its **exact text** as the
  ground truth — so the transcript is *seeded* from the text layer and the only human step is
  certifying the reading order. Recognition drops out as a confound (the reference text is
  certain), making the recall-vs-CER gap pure reading-order error. First proof (Segro 2023 AR
  p66, a 4-column landscape page): VLM CER **0.020** / recall 0.991, deterministic CER 0.029 —
  both read the four columns in correct order, so **clean multi-column prose is handled well**.
  See [eval-labelling.md](../eval-labelling.md) (the `render` method) and [done.md](done.md).
  **So the remaining work is the Z-order traps, not clean columns.** Hand-label (or render +
  certify) a handful of pages where a naive top-to-bottom sort *breaks* — spanning headlines,
  columns interrupted by captions/footnotes, sidebars/call-outs — and read the recall-vs-CER
  split. **Selection principle:** prioritise exactly those break cases; skip clean two-/four-
  column prose (already shown handled — no signal left to gain). The born-digital-render trick
  applies to any born-digital page with such a layout; otherwise source a genuine **scanned**
  page. Good document types (source non-sensitive / public examples, since the transcripts are
  gitignored but quote the page):
  - **Newspaper pages / press clippings** — the canonical stressor (multiple text columns,
    headlines spanning columns, jump/continuation lines, pull-quotes); also realistic journalism
    source material (cuttings in FOI / leak sets).
  - **Two-column academic / journal papers (scanned)** — dense columns with figure/table captions
    and footnotes interrupting the flow, the classic caption-vs-column ordering trap.
  - **Government gazettes / official journals (scanned)** — two-/three-column statutory text;
    public-interest-relevant and freely available.
  - **Newsletters / magazine spreads** — irregular column widths, sidebars and callout boxes that
    a top-to-bottom sort splices into the main column mid-sentence.
  - **Reports with a marginal sidebar** — a main column plus a margin column (key facts / quotes)
    that must not be interleaved into the body.

(The Qwen3.5-VL re-test is **done** — switched the default reader to
`mlx-community/Qwen3.5-9B-MLX-4bit`; see [done.md](done.md).)

## Next

- **Tables:** multi-level-header semantics for `find_tables`; cross-validate `find_tables` vs
  the vision grid; cleaner per-cell content on scanned tables. **Test-coverage gap (2026-06-30):**
  the scanned-table → focused-VLM-table-read path is currently UNTESTED — in test set 1 the
  scanned docs (Thai forms) are layout-classified `paragraph`/`header`/`footer`, not `table`, and
  the `table`-classified docs (sackler) are born-digital (`find_tables`). A genuinely *scanned*
  data table is missing from the corpus; source one to exercise the path.
- **`sha → original filename` manifest** — the `out/` folders are content-hash named, so they're
  opaque to a human ("which job is which?"). The original filename is already recorded inside each
  job (`doc.json` → `source_path`), but a top-level manifest mapping sha → original would save the
  lookup. *(Carved out of the now-done output-artifacts doc — see [done.md](done.md) /
  [outputs.md](../outputs.md).)*
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
- **Improve fusion anchoring on rotated dense print** — *parked; likely acceptable as-is.* The
  searchability eval now measures a real gap on rotated small print: the rotated Goldfinch
  invoice scores **searchable recall 0.65 vs 0.89 reading** — where word-anchoring fails on the
  rotated footer (bank details, codes), the overlay keeps the garbled `det_text` rather than
  smearing the reading onto guessed positions (the honest fallback). **Not data loss:** the full
  clean reading is in `document.md` and `doc.json`, which Giant indexes as separate search views,
  so a query that misses in the Combined (overlay) view still resolves in the text view and takes
  the reader there. So this is a search-*precision* refinement, not a defect — revisit only if
  reporters hit it in real use. (Measured by the searchability eval; see [done.md](done.md).)
- **Result push for non-airgap tiers:** an optional webhook / callback on completion. The
  sealed (airgap) tier stays poll-only by construction — the process can't dial out — so this
  is a tier-gated enhancement, never the default.

## Later — beyond MVP

Capability beyond the MVP target:

- **Rotated-page tables** — the table-structure and focused table-read stages currently skip
  rotated pages ([review_02](review_02_2602627.md) #8). Add support when rotated scans turn
  up in the corpus.
- **Collapse Giant's "text" vs "OCR text" views (integration value).** Giant shows four
  per-document views — original, Combined (PDF + overlay), machine-readable text, OCR text —
  each separately indexed; the two text views confuse users ("which do I read if Combined is
  problematic?"). This tool emits a *single* provenanced output that already composes
  born-digital text and OCR into one artifact (exact text layer kept, OCR only where the page
  needs it — never doubled), so it could let Giant retire the text/OCR split into one view. A
  downstream Giant change, noted here as a reason the structured output earns its keep.

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

- **SROIE** (`invoice/`, 973 receipts) — genuine human GT, in-domain; loader + benchmark wired
  (`eval/datasets.py`, `--dataset sroie`), pairs 98/98 by stem. Ignore its key-value / entity
  labels (downstream scope, principle 1); only the text GT applies.
- **FUNSD** (`form/`, 199 scanned forms) — genuine human GT, in-domain; wired (`--dataset funsd`).
  Quirk: this packaging split train/test/val INDEPENDENTLY for images vs annotations, so an
  image's annotation usually sits in a *different* split folder — the loader pairs by stem across
  all splits (200/203 match). (An earlier note here wrongly called FUNSD "broken"; it isn't.)
- **IAM** (`document/`, 1539 handwriting pages) — the headline-case prize, BUT the bundled json
  annotations carry a per-line `confidence`: they're an OCR engine's *output*, not human
  transcriptions, so scoring our OCR against them is **circular**. Source IAM's original
  transcriptions before using it as ground truth.
- **Total-Text** (`real_life/`, ~1554) — scene text (photos / signage), arguably **out of the
  document domain**; likely set aside.

The ingest adapter + SROIE/FUNSD loaders are now built, so this is partly explored. **First
benchmark run** (via the image→PDF ingest path), aggregate word recall / CER:

| source | PaddleOCR | Apple Vision | Qwen3-VL-8B |
| --- | --- | --- | --- |
| SROIE receipts (n=12) | 0.595 / 0.485 | 0.589 / 0.492 | **0.624 / 0.319** |
| FUNSD forms (n=10 det, n=8 vlm) | **0.816 / 0.397** | 0.717 / 0.448 | 0.782 / 0.388 |

Read: **PaddleOCR is the deterministic baseline and beats Apple Vision on both** (reinforcing
[[feedback-paddleocr-is-the-deterministic-baseline]]). The VLM's value **scales with difficulty**:
transformational on handwriting (the Mandelson 0.10→0.95 story), a modest recall lead + big CER
win on degraded thermal receipts, and **no clear lead on clean forms** (PaddleOCR edges it). So
the product earns its keep exactly where the deterministic tools fail; on clean print the
deterministic engine alone is already strong. Caveats: small n, and the FUNSD VLM row ran on 8 of
the 10 (so not a strictly matched set). To grow this: scale n with matched sets, add per-source
loaders. Fuller analysis in the `dataset-3rdparty-ocr-benchmark` session memory.

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
