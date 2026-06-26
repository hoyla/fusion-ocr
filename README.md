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

## Status: walking skeleton

The plumbing runs end-to-end with **every model stage stubbed as a passthrough** —
a dropped PDF flows triage → … → render and emits a (currently trivial) segment
index, markdown, and overlay. Stages get filled in one at a time, starting with
PaddleOCR. The contract that makes that safe — `Document` in / `Document` out, raw +
inferred both retained, serialised between stages for resume — is already in place
and tested.

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
```

| Extra | Brings | For |
| --- | --- | --- |
| `ocr` | PyMuPDF, PaddleOCR, PaddlePaddle | triage + geometry + structure |
| `vlm` | httpx, Pillow | the OpenAI-compatible VLM client |
| `api` | FastAPI, uvicorn | the HTTP job API |

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

## Layout

```
src/fusion_ocr/
  models.py          the Document/Page/Region/Segment record
  pipeline.py        Stage protocol + orchestration + resume
  config.py          config + airgap guard
  jobs.py            SQLite job table (idempotent by content hash)
  watcher.py         drop-folder entrypoint
  api.py             HTTP job API (stable contract for callers)
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
(PaddleOCR, always on, recogniser language varies) and reading (VLM, specialist
varies). See **[Docs/routing.md](Docs/routing.md)**. Thai is the first specialist
route: `paddle_lang=th` reads Thai forms that the generalist VLMs can't, and a
config `[routing.thai]` section points the *reader* at a served Typhoon endpoint when
one's available.

## Roadmap

1. **PaddleOCR geometry** — boxes + text + confidence (`stages/ocr_det.py`). ✅ *done — 2.x & 3.x; per-language recogniser via the router.*
2. **VLM read + fusion** — read via the swappable client; cluster boxes + sequence-align VLM lines onto them; ink-gate. ✅ *done — proven on the handwritten note (6 → 3,185 searchable chars).*
3. **Tool router** — script detection → per-region `{PaddleOCR recogniser + VLM reader}`; provenance; generalist-refusal fallback. ✅ *done.*
4. **Readers** — default generalist Qwen3-VL via **MLX** (`mlx_vlm.server`, ~10-40× faster than Ollama on Apple Silicon); **Typhoon** specialist for Thai (Ollama); confidence-gated escalation to a stronger model. ✅ *done.*
5. **Layout + tables (PP-DocLayout)** — region detection + region-aware clustering (columns don't merge) + deterministic table-cell extraction (HTML grid + cell boxes). ✅ *done.*
6. **Script detection** — text-layer Unicode classify + one-word VLM probe for image-only pages. ✅ *done.*
7. **Overlay** — Unicode font so non-Latin search works ✅; word-level subdivision for precise highlights *(follow-up)*.
8. **Follow-ups** — true multi-column reading order (PP-StructureV3); per-cell table content; the "Giant rejects" eval (old vs new); Qwen3.5-VL re-test when its MLX build lands.
