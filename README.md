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

## The VLM seam (local now, CUDA later)

One config value moves the heavy work. Ollama, MLX, and vLLM all speak the same
OpenAI-compatible API:

```toml
[vlm]
base_url = "http://localhost:11434/v1"            # Ollama, today
# base_url = "http://transcription-gpu.internal:8000/v1"   # vLLM, later — nothing else changes
```

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

## Roadmap

1. **PaddleOCR geometry** — real boxes + text + confidence (`stages/ocr_det.py`).
2. **PP-StructureV3 layout** — regions, tables, reading order (`stages/layout.py`).
3. **VLM read** — wire `vlm_read` to the client; tables → markdown; translation.
4. **Fusion** — sequence-align VLM text onto boxes; ink-gate; best-of.
5. **Overlay** — line-level first, then word-level subdivision for selection fidelity.
6. **Eval** — run the "Giant rejects" pile; side-by-side old vs new text.
