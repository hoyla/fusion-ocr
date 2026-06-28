# Done — shipped capabilities

A running log of what's built. Forward-looking work lives in [roadmap.md](roadmap.md); the
*why* is in [motivation_and_strategy.md](motivation_and_strategy.md). Point-in-time detail for
the big remediations is in the review/plan notes — this is the index, not a duplicate.

## Pipeline (MVP core)
- Triage → layout → language → OCR (PaddleOCR / Apple Vision) → VLM read → tables → fusion →
  render. `Document`-in/out, recipe-fingerprinted resume, per-stage snapshots.
- Deterministic geometry + VLM reading, fused with the **ink-gate**; provenance retained
  (`det_text` / `vlm_text` / `source` / `read_by` / `superseded`).
- **Headline proof:** the handwritten Mandelson→Lammy note — 6 source chars → ~3,185
  searchable chars, fully local. Now *measured*: **0.95 word recall** vs a hand transcript of
  the 2-page letter, where the deterministic engines (PaddleOCR / Apple Vision) manage ≈0.45.
- Two-axis routing ([routing.md](../routing.md)): Apple Vision cheap tier, Typhoon Thai
  specialist, Qwen3-VL generalist via MLX.
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

## Reviews & hardening
- **Review 01** ([review_01_260627.md](review_01_260627.md)): recipe-fingerprint resume,
  pinned deps + lockfile, airgap guard (DNS / connect_ex / full loopback), ungated
  `document.md`, schema-driven `from_json`.
- **Review 02** ([review_02_2602627.md](review_02_2602627.md)): `python-multipart`,
  `overlay_font` in the fingerprint, language-probe airgap re-raise, confidence-gated fusion
  guard, atomic `upsert` + WAL, watcher settle gate + digest passthrough, Apple Vision
  failure logging.

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
