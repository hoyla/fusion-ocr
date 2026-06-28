# fusion-ocr

A hybrid **deterministic + Vision-LLM** OCR pipeline for confidential documents.

It pairs a deterministic engine (PaddleOCR / PP-StructureV3) with a self-hosted VLM
so you get the best of both: **trustworthy bounding boxes** from the deterministic
side and **clean, structure-aware reading** (tables → markdown, translation) from the
VLM — fused into one "best of" output with a searchable bbox text overlay.

Everything runs **on your own estate** — local models via Ollama/MLX now, an in-VPC
CUDA endpoint (e.g. vLLM on the transcription GPU) later — with **no third-party LLM
calls**. The most-sensitive tier runs fully air-gapped.

## Core idea

| Concern | Owner |
| --- | --- |
| **Geometry** (boxes, tables, reading order) | deterministic — PaddleOCR / PP-StructureV3 |
| **Semantics** (reading, structure, translation) | VLM |
| **Fusion** | align VLM text onto deterministic boxes; the deterministic layer is the anti-hallucination gate |

Boxes never come from the VLM. That's what makes the searchable overlay reliable and
keeps the VLM from inventing text where there was no ink.

## Status: working pipeline

The real stages are in: triage → layout → language → OCR (PaddleOCR / Apple Vision) →
VLM read → tables → fusion → render. A dropped PDF emits a provenance-carrying segment
index, per-language markdown, and a searchable bbox overlay. On a born-digital
recognition eval the OCR path scores **~95% word recall / ~96% precision**; the headline
case — a handwritten note tesseract recovers 6 characters from — comes through at ~3,185
searchable characters, fully local.

The contract underneath — `Document` in / `Document` out, raw **and** inferred both
retained, serialised between stages for a recipe-fingerprinted resume — is in place and
tested (130 tests). Degraded scans and handwriting still want a small hand-labelled set to
measure against; word-level overlay subdivision and a few table refinements are follow-ups
(see the roadmap).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # bare plumbing — no model deps
cp config.example.toml config.toml

python -m fusion_ocr.watcher --once   # process anything in in/, then exit
# drop a PDF into in/ ; artifacts land in out/<sha256>/
pytest                                # plumbing tests
```

Add capability behind extras as stages are implemented:

```bash
pip install -e ".[ocr,vlm,api]"
# For a byte-for-byte reproducible environment (same OCR output for a given PDF),
# install against the pinned lock — versions are bounded in pyproject, exact in the lock:
pip install -e ".[ocr,vlm,api]" -c constraints.txt
```

| Extra | Brings | For |
| --- | --- | --- |
| `ocr` | PyMuPDF, PaddleOCR, PaddlePaddle | triage + geometry + structure |
| `vlm` | httpx, Pillow | the OpenAI-compatible VLM client |
| `api` | FastAPI, uvicorn, python-multipart | the HTTP job API |

## Serving the readers (the toolkit)

Readers are OpenAI-compatible endpoints, so each tool runs wherever it's fastest and
the router points per-script at the right one. Default setup on Apple Silicon:

```bash
# generalist reader — Qwen3-VL via MLX (~10-40x faster than Ollama on Apple Silicon)
pip install mlx-vlm
python -m mlx_vlm.server --port 8080        # serves mlx-community/Qwen3-VL-8B-Instruct-4bit

# Thai specialist — Typhoon on Ollama
ollama pull scb10x/typhoon-ocr1.5-3b
```

The router sends Latin/handwriting/etc. → Qwen3-VL/MLX (`:8080`), Thai → Typhoon
(`:11434`). One config value moves any reader — Ollama, MLX, or in-VPC vLLM all speak
the same API, so swapping local → transcription-GPU is just a `base_url` change. See
[Docs/routing.md](Docs/routing.md).

## Configuration & the job API

Configure via `config.toml` (copy from `config.example.toml`), or change the
output-affecting tuning knobs on a **running** service over HTTP. The job API (the `api`
extra, `uvicorn fusion_ocr.api:app`) is the stable contract callers use:

| Method & path | Purpose |
| --- | --- |
| `POST /jobs` (multipart `pdf`) | submit a PDF → `{sha256, status}` |
| `GET /jobs/{sha256}` | job status + artifact list |
| `GET /config` | surface every setting (secrets masked) + its constraints |
| `PATCH /config` `{path: value}` | configure the allowlisted settings in-process |
| `POST /config/save` | persist the current config to disk (explicit, opt-in) |

The API is bearer-token auth'd and **fails closed**: set `FUSION_OCR_API_TOKEN` (it refuses
to start without one) and send `Authorization: Bearer …`. Security/identity fields
(`airgap`, `in_dir`, `out_dir`, `routes`) are surfaced but **read-only** — the API can't
unseal the airgap tier or repoint paths. A tuning change re-keys the resume cache, so the
next job reprocesses rather than reusing a stale result.
**Full settings table and endpoint details: [Docs/configuration.md](Docs/configuration.md).**

## Layout

```
src/fusion_ocr/
  models.py          the Document/Page/Region/Segment record
  pipeline.py        Stage protocol + orchestration + resume
  config.py          config + airgap guard
  settings.py        settings registry — what's surfaceable vs runtime-configurable
  raster.py          page-raster cache — render a page once, share across stages
  jobs.py            SQLite job table (idempotent by content hash)
  watcher.py         drop-folder entrypoint
  api.py             HTTP job + config API (stable contract for callers)
  stages/            triage · layout · language · ocr_det · vlm_read · fusion · render
  vlm/               client protocol + OpenAI-compatible impl + prompts
  overlay/           PyMuPDF invisible-text overlay (line- → word-level)
```

## Airgap note

PaddleOCR downloads its models to `~/.paddlex` on first use. For the air-gapped
sensitive tier, **pre-pull the models once on a connected machine** (run any OCR job),
then run sealed — `enforce_airgap()` sets `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` so
Paddle won't attempt its connectivity check.

## Tool routing

The pipeline triages each region and routes it on two independent axes — geometry
(always a deterministic engine — PaddleOCR, or Apple Vision when preferred — never the
VLM) and reading (VLM, specialist varies). See **[Docs/routing.md](Docs/routing.md)**. Thai is the first specialist
route: `paddle_lang=th` reads Thai forms that the generalist VLMs can't, and a
config `[routing.thai]` section points the *reader* at a served Typhoon endpoint when
one's available.

## Evaluation

Accuracy is a number, not a claim. A born-digital page carries its own exact text in the
layer, so the eval renders it to an image (dropping the text layer), forces the OCR path,
and scores the recovered text against the embedded text — no hand-labelling:

```bash
python -m fusion_ocr.eval report.pdf --pages 6,9,24 --apple-vision
```

It reports **word recall / precision** (order-insensitive: recognition completeness and
the hallucination rate) alongside **CER / WER** (sequence-based, so reading order on
multi-column pages inflates them — trust CER on single-column text). Caveat: rendered-clean
pages are a floor on difficulty; genuinely degraded scans / handwriting need a small
hand-labelled set. See `src/fusion_ocr/eval/`.

## Design principles

Why this tool exists and how its architecture serves that is narrated in
**[Docs/dev_notes/motivation_and_strategy.md](Docs/dev_notes/motivation_and_strategy.md)**.

Build-vs-adopt, determinism vs learned models, and auditing model defaults are codified in
**[Docs/principles.md](Docs/principles.md)** — the short version: orchestrate trusted tools,
build only the connective tissue and the journalism-specific guarantees no component
provides, and prefer a library's public surface over its internals.

## Roadmap

1. **PaddleOCR geometry** — boxes + text + confidence (`stages/ocr_det.py`). ✅ *done — 2.x & 3.x; per-language recogniser via the router.*
2. **VLM read + fusion** — read via the swappable client; cluster boxes + sequence-align VLM lines onto them; ink-gate. ✅ *done — proven on the handwritten note (6 → 3,185 searchable chars).*
3. **Tool router** — script detection → per-region `{PaddleOCR recogniser + VLM reader}`; provenance; generalist-refusal fallback. ✅ *done.*
4. **Readers** — default generalist Qwen3-VL via **MLX** (`mlx_vlm.server`, ~10-40× faster than Ollama on Apple Silicon); **Typhoon** specialist for Thai (Ollama); confidence-gated escalation to a stronger model. ✅ *done.*
5. **Layout + tables (PP-DocLayout)** — region detection + region-aware clustering (columns don't merge) + deterministic table-cell extraction (HTML grid + cell boxes). ✅ *done.*
6. **Script detection** — text-layer Unicode classify + one-word VLM probe for image-only pages. ✅ *done.*
7. **Overlay** — Unicode font so non-Latin search works ✅; word-level subdivision for precise highlights *(follow-up)*.
8. **Follow-ups** — true multi-column reading order (PP-StructureV3); per-cell table content; the "Giant rejects" eval (old vs new); Qwen3.5-VL re-test when its MLX build lands.
