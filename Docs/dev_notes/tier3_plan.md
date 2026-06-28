# Tier-3 plan — harden + scale before exposing the service

> **Status: COMPLETE.** All five items shipped (see the ✅ markers below). This is a finished
> plan kept as a record — the canonical live status is **[done.md](done.md)** /
> **[roadmap.md](roadmap.md)**. Note: item #2's "a real background queue is a later step" was
> subsequently built — the async job queue (see done.md), which supersedes the in-thread
> offload described here.

The review's **Tier-3** items: not bugs in the output, but the things to settle *before the
job API is exposed or the throughput goes up*. Grounded in the code as it stands (commit
`cbd74ef`); file:line references are where each problem actually lives. Tiers 1–2 and the
review-02 follow-ups are already done — see [review_01](review_01_260627.md),
[review_02](review_02_2602627.md) and the memory note.

Five items, in suggested order of execution.

---

## 1. Page-raster cache — collapse the duplicate rasterisation

**✅ Done** — `src/fusion_ocr/raster.py` (LRU Pixmap cache under a byte budget; serves both
the ndarray and PNG consumers and centralises the conversion the three stages had
copy-pasted). All six rasterising stages migrated; `test_raster.py` added (suite 134).

**Problem.** Seven stages each `fitz.open(doc.source_path)` and `get_pixmap()`
independently. The DPIs:

| Stage | DPI | Note |
| --- | --- | --- |
| `stages/layout.py:77` | 150 | full page |
| `stages/table.py:168` | 150 | full page |
| `stages/vlm_read.py:64` | 150 | full page |
| `stages/table_read.py:76` | 150 | **clipped** (a region crop) |
| `stages/ocr_det.py:137` | 200 | full page |
| `stages/language.py:69` | 120 | image-only pages without a script |

So a scanned page is full-page rasterised **at 150 DPI three independent times**
(layout + table + vlm_read), plus once at 200 for OCR. Rasterisation is a real cost; this
is pure recompute.

**Approach.** Add `src/fusion_ocr/raster.py`:

- `page_png(path, page_index, dpi, clip=None) -> bytes` and `page_ndarray(path, page_index, dpi) -> np.ndarray`.
- Backed by an LRU keyed on `(path, mtime, page_index, dpi, clip)`. **mtime in the key**
  keeps it correct if a file is replaced in place.
- Migrate the seven stages off their own `fitz.open`/`get_pixmap` onto these helpers. The
  dpi→points scale math (`scale = dpi / 72.0`) stays in the stages; only the pixel
  acquisition moves.

**Risks / decisions.**
- Pixmaps are large → cache **PNG bytes**, not raw `Pixmap`s, with a bounded `maxsize`
  (a handful of pages × a few DPIs). Cache is per-process and ephemeral.
- The clipped `table_read` raster has a distinct key (the `clip`), so it never collides
  with the full-page one — correct, just not deduplicated against itself (fine).

**Tests.** Cache hit on a repeat `(path, page, dpi)`; key invalidates on mtime change; a
stage (e.g. ocr_det) produces byte-identical boxes before/after the migration.

**Effort:** medium. **Depends on nothing — do this first** (pure perf, no API surface).

---

## 2. Offload `process()` off the event loop

**✅ Done** — `submit` runs the synchronous pipeline via `anyio.to_thread.run_sync`, so a long
job no longer blocks other requests. JobStore concurrency is covered by the WAL + atomic
upsert; the raster cache has a lock.

**Problem.** `api.py:50-52` — `async def submit` calls the **synchronous** `process()`
inline. The whole pipeline (seconds–minutes of CPU + blocking I/O) runs on the event loop,
so one upload blocks every other request, including `GET /jobs` status polls.

**Approach.** Minimal correct fix for the current process-on-submit shape:
`await anyio.to_thread.run_sync(process, ...)` (Starlette already depends on anyio). A real
background queue + worker is a later step — not needed until throughput demands it.

**Risks / decisions.**
- `JobStore` concurrency is already covered (WAL + atomic `INSERT .. ON CONFLICT`).
- A `PATCH /config` mutates the shared `cfg` in place; with concurrent jobs that means a
  config change applies to **subsequently started** jobs, not in-flight ones. Document that
  explicitly; it's the intended semantics, not a race to fix.

**Tests.** Two concurrent submits don't serialise (overlap observable); `GET /jobs`
responds while a job runs.

**Effort:** small–medium.

---

## 3. Upload size limit + content sniff

**✅ Done** — `_save_upload` streams the upload in 1 MiB chunks, sniffs the `%PDF-` header
(415) and enforces `cfg.max_upload_mb` (413) before hashing, cleaning up the partial file on
rejection. New settable setting `max_upload_mb` (default 50).

**Problem.** `api.py:46` — `dest.write_bytes(await pdf.read())` reads the entire upload
into memory with no ceiling and no type check before hashing/processing.

**Approach.** Stream the upload to disk in chunks against a byte ceiling
(`cfg.max_upload_mb`, a new **settable** setting → also surfaced via `GET /config`); abort
with **413** past the limit. Sniff the PDF magic (`%PDF-`) before hashing; **415** otherwise.

**Tests.** Oversized upload → 413; non-PDF → 415; a normal PDF still round-trips.

**Effort:** small.

---

## 4. API auth — static bearer token

**✅ Done** — `FUSION_OCR_API_TOKEN` (env only), checked by an app-level FastAPI dependency on
every route (constant-time compare → 401). **Fail closed:** `create_app` raises if no token
is set, so an unauthenticated API can never be served. `create_app(cfg, token=...)` injects
one for tests.

**Decision (Luke, 2026-06-28): static bearer token; fail closed when the token is unset.**

**Problem.** Every endpoint is open, including `PATCH /config` (which can repoint the reader
endpoint and tune the pipeline).

**Approach.** A bearer token read from config/env (`FUSION_OCR_API_TOKEN`, never written to
`config.toml`), checked by a single FastAPI dependency applied to **all** routes. Missing or
wrong token → **401**. Constant-time compare. When the token is unset, fail closed for
mutating routes (refuse to start, or 503) rather than silently running open — to be decided
at implementation, default **fail-closed**.

**Note.** If the service later sits behind a reverse proxy / mTLS in-VPC, app-level auth may
become redundant — but the bearer token is the right default now and composes fine with a
proxy.

**Tests.** No token → 401; bad token → 401; good token passes; unset-token startup behaves
per the fail-closed decision.

**Effort:** small.

---

## 5. Watcher — move processed files out of `in/`

**✅ Done** — on success a file moves to `in/processed/<sha>.pdf`, on error to
`in/failed/<sha>.pdf` (the glob is non-recursive, so they're not re-scanned). New settable
setting `move_processed` (default on); the watch loop honours it, `--once` never moves.

**Problem.** `watcher.py` globs `in/*.pdf` and `sha256_of`-hashes **every settled file on
every 2s tick**; processed files are never moved, so the whole backlog is re-hashed forever
(the settle gate stops mid-copy grabs but doesn't stop the re-hashing).

**Approach.** On success move `in/x.pdf` → `in/processed/<sha>.pdf`; on error →
`in/failed/`. Exclude those subdirs from the glob. Removes the re-hash-everything cost and
gives an at-a-glance view of state. Opt-in via a setting (`move_processed`, default **on**
for the watch loop); leave `--once` non-moving unless asked, so a user re-running by hand
isn't surprised.

**Tests.** A processed file is moved and not re-scanned next tick; a failing file lands in
`failed/`; `--once` honours the opt-out.

**Effort:** small–medium.

---

## Sequencing

1. **#1 raster cache** — pure win, no API surface, no decisions. Do first.
2. **#3 size limit + #5 watcher move** — small, independent hardening; can land together.
3. **#2 offload + #4 auth** — the "before we expose it" pair; land together when exposure is
   actually on the table.

## Resolved decision

- **Should `PATCH /config` persist to `config.toml`?** **✅ Resolved (2026-06-28).** `PATCH`
  stays in-process only; added an explicit, opt-in **`POST /config/save`** that writes the
  current config (incl. runtime tuning) to disk via `config.save` (tomli-w) — so a transient
  experiment can't silently become the on-disk default. The saved file is generated (no
  comments); `config.example.toml` remains the documented reference.
