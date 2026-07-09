# Roadmap

Forward-looking work, roughly by priority. Shipped capabilities are logged in
[done.md](done.md); the rationale behind the architecture is in
[motivation_and_strategy.md](motivation_and_strategy.md).

Ordering principle (principle 6, *look before infra*): the scale-triggered items below are
deliberately **not** built yet — the interfaces are shaped so they drop in when the load is
real, not before.

## Now / near-term

- **Run the evidence plan** ([evidence_plan.md](evidence_plan.md)) — the pre-registered
  measurement campaign from [review 03](review_03_260705.md): full FUNSD/SROIE runs, the
  box-placement metric (P1), insertion-rate reporting + blank-page probes (P2), noise floor,
  threshold sensitivity, quant A/B, IAM fix. This subsumes the eval-expansion goals below at
  larger scale; the hand-labelling work remains complementary (it covers cases the gold sets
  don't).
  **Status (2026-07-09) — streams A, C, G, D done** (manifests in `eval_out/manifests/`):
  VLM out-recognises deterministic on both corpora; fused placement ≥ deterministic under the band
  metric (P1); noise floor is zero; 150 DPI confirmed on gold; P2 measured — D2 blank probes pass
  (0 gated invented words), D3 divergence triage empty on clean gold, **D1 fired tripwire (b)**
  (gated char-insertion > ungated = a reading-order confound, not hallucination) → the **P2 headline
  framing is pending certification** (see the insertion_gate manifest / evidence_plan §D).
  **Outstanding:** **F** — quant 4-bit-vs-8-bit + Qwen3.5-vs-Qwen3-VL at n≥50 (config-only,
  unblocked by the zero noise floor); **E** — threshold sensitivity (±30%, 4 constants) + the
  `escalate_below` keep-or-delete decision; **B** — IAM handwriting beyond n=1, blocked on the
  external FKI/IAM transcription registration (log-and-move-on if it stalls > 1 week).
- **Fail loud on reader failure (review 03).** `vlm_read` / `table_read` / `language` catch
  every exception and return `""` with no logging anywhere in those stages — a dead MLX server
  or misconfigured model name silently degrades the whole corpus to det_text. Add logging + a
  per-page `read_failed` provenance flag so a degraded run is visible in the artifacts.
- **Job-lifecycle document-loss bugs (review 03):** (a) API uploads are keyed by client
  filename — two same-named uploads with different content overwrite in `in/` and strand the
  first job `queued` forever; key by digest/UUID. (b) A worker killed between `claim` and
  `set_status` orphans the job as `running` forever — add a lease/timeout. (c) The watcher's
  main loop dies if a file vanishes between `iterdir` and `stat`/hash — guard the loop, not
  just `process()`.
- **Eval writes recovered text to `/tmp` (review 03).** `harness`/`labels`/`datasets` use
  `tempfile.mkdtemp` for rendered pages + full text, never cleaned — fine for benchmark data,
  dangerous the day someone runs `--labels` on a confidential document. Use a run-scoped dir
  under `eval_out/` (gitignored) with cleanup.
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

- **Fusion/metrics internals → rapidfuzz (review 03).** The alignment *concept* is ours; the
  *implementation* hand-rolls Needleman-Wunsch twice with a fresh `difflib.SequenceMatcher`
  per DP cell (and a backtrace that recomputes similarity). Port the same algorithms to
  `rapidfuzz` primitives: C speed, and it retires the `_MAX_DP_CELLS` guard whose fallback is
  a *worse* alignment — a quality win, not just perf. Same for `eval/metrics.edit_ops`
  (→ `rapidfuzz`/`jiwer`). Guard the port with the placement metric
  ([evidence_plan.md](evidence_plan.md) stream C) — same numbers before/after.
- **Pipeline the CPU and VLM tracks (perf, review 03).** Everything is serial today: stages in
  sequence, pages in a loop, one job at a time. Overlapping the CPU-bound `ocr_det` (~40%)
  with the GPU/ANE-bound `vlm_read` (~38%) — and/or page-level parallelism — is up to ~2×
  from scheduling alone, zero quality cost. The biggest perf lever after the engine choice.
- **Fusion edge: contaminated text layer can reach `best_text` (review 03).** A PUA-
  contaminated textlayer segment is only superseded if an OCR box overlaps it at IoU ≥ 0.5;
  PaddleOCR's over-segmentation routinely never reaches that, so the garbage can survive as a
  primary segment. Supersede on *coverage of the contaminated span* (many-to-one), not
  pairwise IoU.
- **Per-region script routing (review 03).** Script is decided per page from any text layer,
  so a Latin header (Bates stamp, "EXHIBIT 12") over a Thai scanned body routes the page
  Latin and skips the probe. Decide script per *region* where layout gives regions, or at
  least probe when the text layer is furniture-only. (Same family as the mixed-content
  composition already shipped.)
- **detect_script coverage (review 03):** the hand-rolled Unicode ranges miss Arabic
  presentation forms (U+FB50–FDFF, U+FE70–FEFF — common in real PDF text layers), CJK
  extensions, fullwidth forms. Either extend the ranges or adopt the `regex` module's script
  properties (build-vs-adopt says adopt).
- **Artifact retrieval over the API (review 03).** `GET /jobs/{sha}` returns artifact *names*
  only — a remote consumer (Giant) can't fetch `document.md`/`overlay.pdf` without a shared
  filesystem, which contradicts the "stable contract" framing. Add
  `GET /jobs/{sha}/artifacts/{name}` (and stop listing internal `doc.NN-*.json` resume
  snapshots as artifacts).
- **`PATCH /config` doesn't reach the worker (review 03).** In the two-process deployment it
  mutates only the API process's config; the worker never sees it. Either propagate (config
  version in the job row, worker reloads) or re-document the endpoint as save-and-restart.
- **Airgap: pair the Python seal with an OS-level control (review 03).** The monkeypatch
  covers `connect`/`connect_ex`/`getaddrinfo` but not `gethostbyname`/UDP — and nothing at
  the C level (paddle/onnxruntime natives, subprocesses). Document the seal as a tripwire,
  and ship a sealed-tier recipe (pf rules / network-less user / `sandbox-exec`) in
  [deployment.md](../deployment.md).
- **Candidate new tiers from the 2026 survey (review 03)** — evaluate under the existing
  harness, adopt only on numbers: **PaddleOCR-VL-0.9B** now has an official Apple-Silicon
  path via mlx-vlm — a cheap *structured* reader that could sit between Apple Vision and the
  9B generalist (caveat: its reading is un-cross-checked VLM output, so it slots in as a
  *reader*, never as geometry); and Apple Vision's WWDC25 **`RecognizeDocumentsRequest`**
  adds paragraph/table/list structure to the fast tier we currently use as flat det/rec —
  possibly a free structure signal on the ANE. (Engine A/Bs — PP-OCRv6, RapidOCR,
  PP-DocLayoutV3 — live in [rapidocr_eval_plan.md](rapidocr_eval_plan.md).)
- **Replace the VLM script-probe with a cheaper detector (perf).** The `language` stage IDs the
  dominant script to route the recogniser/reader; for no-text-layer scans it fires a whole 9B-VLM
  image inference per page to do it (the documented "first cut" — [routing.md](../routing.md) flags
  *"a fast langid VLM probe, **or a script classifier**"*). Per-stage profiling (2026-06-30) now
  shows that probe is **~14% of total runtime** — a heavyweight model answering "Latin or CJK?".
  Quick win already shipped (probe 120→72 DPI, ~halves it; see [done.md](done.md)); the proper fix
  is a cheaper detector: **ANE Apple Vision on a crop**, a tiny script classifier, or a
  **known-corpus-script config** to skip the probe when the operator knows the estate's language
  (e.g. all-English). Deterministic/auditable + near-eliminates the cost. (Profiling found
  `ocr_det` PaddleOCR-on-CPU ~40% and `vlm_read` MLX ~38% are the other big costs; PaddleOCR can't
  use the ANE — PaddlePaddle has no Metal backend — so its lever is image size / oneDNN, not a
  device switch.)
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
- **Ingest robustness (review 03):** encrypted/password PDFs are unhandled (fail confusingly
  deep in a stage — detect `needs_pass` at ingest and fail with a clear job error); **HEIC**
  (what an iPhone photo of a document actually is) and WebP aren't sniffed; `image_to_pdf`
  embeds photos at native pixel size (a 20 MP JPEG → ~420 MB pixmap per raster — add a
  downscale cap). Also: page-level triage never OCRs a small embedded scan (<40% image, ≥50%
  text coverage) on a mostly-text page — region-level OCR is the fix, same family as the
  per-region script item under Next.
- **Small-bug sweep from review 03** (each minor alone, worth one pass): overlay
  `granularity="word"` places words at equal-width steps (wrong geometry — fix via per-word
  boxes or drop the mode); `populate_table_html` only fills literal `<td...></td>` (a
  `<td> </td>` yields a silently empty table); raster cache keyed on `pdf.name` (empty for
  in-memory docs — latent collision); `vlm_read`'s refusal length-check counts only
  `source=="paddle"` chars so it's disabled on Apple-Vision-routed pages; overlay `place()`
  and several stage helpers swallow all exceptions silently.
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

The ingest adapter + SROIE/FUNSD loaders are now built. **FULL-SET deterministic benchmark**
(evidence-plan stream A, 2026-07-07 — all 199 FUNSD + 973 SROIE, both engines, `--no-vlm`;
manifest `eval_out/manifests/stream_a_deterministic_2026-07-07.md`). Word recall, case-corrected
(see below):

| source | PaddleOCR | Apple Vision |
| --- | --- | --- |
| SROIE receipts (n=973) | 0.901 | **0.911** |
| FUNSD forms (n=199) | **0.810** | 0.749 |

**This supersedes the old n=10–12 pilot table**, which reported SROIE ~0.59 for every engine and
"PaddleOCR beats Apple Vision on both". The full run corrected two things:

1. **The SROIE ~0.6 was a scoring artifact, not the engines.** SROIE's GT is 100.0% uppercase;
   case-sensitive scoring charged every correctly-cased letter as an error. Corrected (caseless,
   per-source flag — the pilot's numbers were raw), true receipt recognition is **~0.90 for both
   engines**. (Third harness-artifact of this class, after CJK tokenisation.)
2. **"PaddleOCR out-recognises Apple Vision" is document-type-dependent, not global.** Paddle
   leads clearly on **forms** (0.810 vs 0.749) but the two are a **tie on receipts** (Vision
   fractionally ahead). So [[feedback-paddleocr-is-the-deterministic-baseline]] holds for forms /
   structured layouts, *not* as a blanket claim — narrow it to the document class.

The VLM rows (the old table's Qwen column, and the "VLM's CER win on receipts" story) are **not
yet re-run at scale** and must get the same caseless treatment before being trusted — stream A's
VLM pass is queued. Handwriting (the Mandelson 0.10→0.95 story) is still n=1 until IAM is
unblocked (stream B). Fuller analysis in [[dataset-3rdparty-ocr-benchmark]].

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
