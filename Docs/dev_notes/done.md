# Done — shipped capabilities

A running log of what's built. Forward-looking work lives in [roadmap.md](roadmap.md); the
*why* is in [motivation_and_strategy.md](motivation_and_strategy.md). Point-in-time detail for
the big remediations is in the review/plan notes — this is the index, not a duplicate.

## Pipeline (MVP core)
- Triage → layout → language → OCR (PaddleOCR / Apple Vision) → VLM read → tables → fusion →
  render. `Document`-in/out, recipe-fingerprinted resume, per-stage snapshots.
- Deterministic geometry + VLM reading, fused with the **ink-gate**; provenance retained
  (`det_text` / `vlm_text` / `source` / `read_by` / `superseded`).
- **Word-level fusion** (`_word_distribute`): the VLM reading is distributed onto the line-boxes
  by fuzzy word alignment, so a long prose line spreads across the visual lines it spans — the
  fix that made handwriting actually *searchable* (line-level alignment left the body as garbled
  det_text). Degrades honestly: too few anchors → line-level NW fallback; no edge-smearing;
  confidence gate still protects printed det_text. The clean reading is never lost — `document.md`
  is the ungated `vlm_reading` (with the exact text layer preserved in the gated overlay/index).
- **Headline proof:** the handwritten Mandelson→Lammy note — 6 source chars → ~3,185
  searchable chars, fully local. Now *measured* old-vs-new against a hand transcript of the
  2-page letter (word recall): **tesseract ~0.10 → deterministic engines (PaddleOCR / Apple
  Vision) ~0.45 → our VLM pipeline ~0.95** — a ~10× recall gain over the old (Giant) stack,
  which recovered only the printed header/footer and essentially nothing of the handwriting.
  (n=1, the canonical reject.)
- Two-axis routing ([routing.md](../routing.md)): Apple Vision cheap tier, Typhoon Thai
  specialist, **Qwen3.6-35B-A3B generalist** via MLX (a 3B-active MoE — switched from Qwen3.5-9B
  after evidence-plan stream F, 2026-07-09: reads better AND ~28% faster at n=55 vs a zero noise
  floor; ~20 GB resident. Qwen3.5-9B is the one-line rollback, still cached).
- Layout PP-DocLayoutV2 (learned reading order); classified SLANeXt scanned tables;
  born-digital PyMuPDF `find_tables`.
- Searchable bbox overlay (Unicode font), per-language markdown, provenance segment index.
- Eval harness (born-digital text layer = ground truth): **~95% recall / ~96% precision**.
- Hand-labelled eval ([eval-labelling.md](../eval-labelling.md)) for scans / handwriting,
  where there is no born-digital truth: human transcripts, multi-page spans, and `--no-vlm`
  to isolate the deterministic engine. First baseline (n=4 hard pages, identical labels):
  **VLM 0.92 recall / 0.17 CER**; PaddleOCR alone 0.82 / 0.34; Apple Vision alone 0.77 / 0.43.
  So **PaddleOCR is the deterministic spine and out-recognises Apple Vision** (Apple Vision is
  the fast on-device tier, not a stronger reader), and the VLM lifts handwriting recall from
  ~0.45 to ~0.95. (n is tiny — directional, not significant; the labelled set is the way to
  grow it.)
- **Searchability eval** — the hand-labelled eval also scores what `find()` hits in the *output
  PDF* (the searchable deliverable), not just the `document.md` reading view: `overlay.pdf` when
  one was built, else the source PDF's own text layer (`searchable` + `searchable_via` in the
  result; `sCER`/`sRcl`/`via` columns). This quantifies the word-level fusion fix — searchable
  recall **equals** the reading on handwriting (0.973) and printed scans (0.984, via the overlay),
  degrades *honestly* to 0.65 on rotated dense print where anchoring fails, and is carried at 0.86
  by the source text layer on the mixed redacted page (not a miss). A dependency-free regression
  guard on the reading-vs-searchable gap.
- **Output-artifacts doc** ([outputs.md](../outputs.md)) — documents every file in
  `out/<sha256>/`: the three deliverables (`document.md` / `overlay.pdf` / `segment_index.json`),
  `doc.json` (final state), the `doc.NN-<stage>.json` resume snapshots + stage order, and
  `source.pdf` (image inputs only). Linked from the README; the `sha → original filename` manifest
  is carved out to the roadmap.

## Reviews & hardening
- **Review 01** ([review_01_260627.md](review_01_260627.md)): recipe-fingerprint resume,
  pinned deps + lockfile, airgap guard (DNS / connect_ex / full loopback), ungated
  `document.md`, schema-driven `from_json`.
- **Review 02** ([review_02_2602627.md](review_02_2602627.md)): `python-multipart`,
  `overlay_font` in the fingerprint, language-probe airgap re-raise, confidence-gated fusion
  guard, atomic `upsert` + WAL, watcher settle gate + digest passthrough, Apple Vision
  failure logging.
- **VLM client hardening** (review_01 follow-ups): `max_tokens` cap, retry + backoff on
  transient 5xx (an `AirgapError` is never retried — sealed tier fails loud), one keep-alive
  `httpx.Client` across pages, and JPEG (not PNG) image parts. `max_tokens` / `jpeg_quality`
  are fingerprinted; all three settings-registered.

## Input formats & benchmarks
- **Image→PDF ingest** (`ingest.py`): PNG/JPEG/TIFF (incl. multi-page TIFF) normalised to PDF
  via PyMuPDF at full resolution, then the pipeline runs unchanged (PDF = identity). Original
  kept canonical; job keyed by the original's hash; derived PDF at `out/<digest>/source.pdf`.
  Wired into the watcher + API by magic-sniff. (Office formats still out of scope.)
- **3rd-party benchmark** (`eval/datasets.py`, `--dataset sroie|funsd`): scores the pipeline vs
  SROIE/FUNSD ground truth through the ingest path. First run (Qwen3-VL recall/CER): SROIE
  0.62/0.32, FUNSD 0.78/0.39; PaddleOCR leads on clean forms — the VLM's lift scales with
  difficulty (big on handwriting, small on clean print).
- **Reading-order measurement on scanned forms** — FUNSD's per-line boxes reconstructed into
  reading order (`_reading_order`) turn ~200 complex scanned forms into a reading-order oracle
  with no hand-labelling, fixing the loader's old "annotation order isn't reading order" caveat
  (FUNSD CER is now reading-order CER). First numbers (n=16): reordering halves the
  deterministic CER (0.44→0.25); VLM reading-order CER 0.15 at recall 0.84 — the learned order
  head and the VLM both order complex forms near-correctly. Method, numbers, caveats:
  [reading_order_measurement.md](reading_order_measurement.md).
- **Born-digital-rendered reading-order labels** (`render: true`) — a label can render its
  born-digital page(s) to an image-only PDF (text layer dropped) so the pipeline OCRs them: a
  multi-column *scan* whose exact text gives a 100%-certain reading order (transcript seeded from
  the text layer; the human only certifies column order). Covers the multi-column *prose* case
  the corpus lacked. First proof (Segro 2023 AR p66, 4-column landscape): VLM CER 0.020 / recall
  0.991, deterministic 0.029 / 0.988 — both read the columns in correct order. See
  [eval-labelling.md](../eval-labelling.md).
- **Per-stage timing + first profiling** — `process()` records wall-clock per stage into
  `Document.stage_seconds` (in `doc.json`), so "where did the time go?" is answerable from the
  output. Profiling (5 representative OCR-Quality pages) found the cost split: **`ocr_det`
  PaddleOCR-on-CPU ~40%, `vlm_read` MLX ~38%, `language` VLM script-probe ~14%**, rest ~8%.
  Quality-safe speedups applied: `ocr_det` 200→**150 DPI** (recognition within ~0.01 word recall,
  and now shares the 150-DPI raster cache instead of a separate render) and the `language` probe
  120→**72 DPI** (script ID unchanged, ~halves the stage; max_tokens was *not* the cost — measured
  no difference, it's vision-token prefill). Bigger structural fix (cheaper script detector) is in
  the roadmap. PaddleOCR is CPU-only — PaddlePaddle has no Metal/ANE backend, so it can't move to
  the Neural Engine; the VLM already runs on MLX, and Apple Vision is the (lower-recognition) ANE
  tier.
- **OCR-Quality 1000-doc run + Claude-Vision adjudication** ([evidence_plan.md](evidence_plan.md)):
  scored our reading vs the Qwen-72B `ocr_text` over all 1000, then Vision-adjudicated the worst
  divergences. Findings: a **CJK metric bug** (`word_recall` meaningless without word-spaces —
  fixed, CJK-aware tokenisation; corrected score-1 agreement 0.93); ~half the worst divergences
  are the *72B's* fault not ours; and two genuine hallucination failure modes, both quarantined
  from the searchable product by the ink-gate (a measured P2 confirmation).
- **VLM hallucination guards** (from that run): (1) **blank/no-ink short-circuit** — if the
  deterministic engine detected no text boxes, skip the VLM entirely (an empty image makes it
  hallucinate, e.g. inventing `$$1/√2$$`; keys on *detection* so handwriting still reaches the
  reader) — also a saved VLM call; (2) **degenerate-repetition guard** — a read that collapses
  into a loop (`[illegible] [illegible] …` to the token cap) is discarded like a refusal, so
  fusion falls back to det_text. Both validated on the real 1000-run failure pages.
- **Evidence-plan stream D executed** ([evidence_plan.md](evidence_plan.md) §D; manifests
  `eval_out/manifests/{insertion_gate,divergence_triage,blank_probes}_2026-07-09.md`) — the P2
  measurement infrastructure: gated-vs-ungated insertion re-scoring (D1), synthetic blank/near-blank
  probes (D2), gold-anchored divergence triage (D3), all re-scored from the archived stream-A docs
  (349/349 reproduce the committed run). **D2 passes** (0 gated invented words); **D3 is empty** on
  clean gold (engines agree); **D1 fired tripwire (b)** — gated char-insertion > ungated, diagnosed
  as a reading-order confound (gated shares 94.4% of words with the reading). **P2 framing certified
  (Luke, 2026-07-09):** on ink-full corpora the gated proxy is the word-level figure — gated
  `1 − word_precision` **0.18 (FUNSD) / 0.10 (SROIE)**, ~recall-free — and the char-`insertion_rate`
  gate *benefit* is reserved for the D2 blank regime (0 gated invented words). So P2's first
  published numbers are a **regime split** (D1 on-content cost + D2 blank benefit), not one headline;
  the ink-gate is not a hallucination-reducer on ink-full pages (it honestly carries detector ink),
  only where the VLM invents past the ink.
- **Evidence-plan stream F executed** ([evidence_plan.md](evidence_plan.md) §F; manifest
  `eval_out/manifests/stream_f_model_ab_2026-07-09.md`) — model + quant A/Bs at n=55 (labelled 5 +
  FUNSD 50) against the zero noise floor. Keep Qwen3.5-9B over Qwen3-VL-8B (recognition tie; 3-VL-8B
  hit a guard-missed `.`-repetition loop → 262k chars; 3.5-9B faster); keep 4-bit over 8-bit
  (marginal gain, 23% slower + 2× memory). **Qwen3.6-35B-A3B** (MoE, 3B-active) measured **better on
  quality AND speed** than the default (recall +0.018, medCER −0.012, ~28% faster) — a generalist
  default-upgrade candidate pending broader validation (Luke's call; no default flipped).
- **Evidence-plan stream B executed — handwriting generalises** ([evidence_plan.md](evidence_plan.md)
  §B; manifest `eval_out/manifests/stream_b_iam_2026-07-09.md`). FKI human transcriptions sourced;
  IAM adapter (`datasets.iam_line_index` / `iam_hw_bbox` / `iter_pairs('iam')`, unit-tested) pairs
  all 1539 pages, cropping each form to its handwritten-line-box region (IAM forms carry a printed
  prompt above the handwriting — a full-page OCR would double-count). At n=100 the **VLM's
  punctuation-normalized handwriting recall is 0.955 — the Mandelson n=1 (0.95) reproduced at
  scale** (medCER 0.035, precision 0.845), vs deterministic PaddleOCR 0.557 / 0.150. The headline
  capability is no longer a single anecdote (IAM = clean ruled English = a floor, not degraded FOI).

## Config & API
- Settings registry (`settings.py`) → `GET` / `PATCH /config`; secrets masked;
  security/identity fields read-only; output-affecting tunables fingerprinted.
- `POST /config/save` — opt-in persistence (tomli-w).
- Bearer-token auth, **fail-closed**; runs behind a TLS reverse proxy
  (`proxy_headers` / `forwarded_allow_ips`).

## Tier-3 (deploy/scale prep) — complete ([tier3_plan.md](tier3_plan.md))
- Page-raster cache (`raster.py`) — render a page once, share across stages.
- Upload size limit + PDF content sniff.
- Watcher move-processed (`in/processed` | `in/failed`).
- Bearer-token auth (above).
- `process()` off the event loop — later superseded by the async queue (below).

## Serving & deployment
- `fusion-ocr-serve` entrypoint; `api_host` / `api_port`; LAN-serve.
- nginx + TLS groundwork ([deployment.md](../deployment.md), [`deploy/`](../../deploy)).
- **Async job queue:** `POST /jobs` enqueues (202); the watcher is the status-driven worker
  (atomic claim); `GET /jobs` feed. `JobStore` (queue) and `storage.py` (content-addressed
  artifacts) are the swap seams for a distributed queue / object store — see roadmap.
- **Worker log buffering fix:** the watcher line-buffers stdout (+ `PYTHONUNBUFFERED=1` in the
  units) so its `[watch]`/`[done]` progress reaches a redirected log live, not block-buffered.
- **Live-loop validated end-to-end** (2026-06-30): API + worker + MLX up, a real document pushed
  through `POST /jobs` → queue → worker → `GET /jobs/{sha}` → artifacts, under airgap, auth
  enforced. The deployed path (not just `process()`/`--once`) is proven.
