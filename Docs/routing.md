# Tool routing — two-axis "horses for courses"

The pipeline does not rely on one OCR engine or one VLM. It **triages each region and
routes it to the right tools.** This document is the design of record for that routing.

## The core idea: two independent axes per region

For every region we make **two** independent tool choices — they are orthogonal:

| Axis | Owner | Always on? | What varies |
| --- | --- | --- | --- |
| **Geometry** (boxes) | deterministic — **PaddleOCR** | **yes, always** | which *recogniser language* (Thai / Latin / Cyrillic / CJK …) |
| **Reading** (semantics) | a **VLM** | when OCR is needed | which *reader* — generalist (Qwen) or a specialist (Typhoon for Thai, …) |

Routing to a specialist VLM **never** removes PaddleOCR. Geometry is always
deterministic; only the *reader* changes. The Thai route is:

```
Thai region → PaddleOCR detection (boxes) + PaddleOCR-Thai recogniser (det_text)
            → Typhoon (authoritative reading)
            → fusion aligns Typhoon's lines onto PaddleOCR's boxes, ink-gated
```

This is the identical hybrid proven on the handwritten note (PaddleOCR boxes + Qwen
reading) — with the Thai recogniser and Typhoon dropped into their slots.

### Why keep PaddleOCR's *language* recogniser on, not just detection

Beyond geometry, using the **Thai recogniser** gives us Thai `det_text`, which
directly sharpens fusion's sequence-alignment: instead of matching the VLM's Thai
lines against garbage English-on-Thai det_text (a weak fuzzy match), we match **Thai
against Thai**, so the reader's lines land on the correct boxes far more reliably.
Picking the right PaddleOCR language model is not a fallback — it improves placement.

## Triage already routes (binary)

The `triage` stage is the first routing decision: per page it chooses **text-layer vs
OCR** from deterministic signals (text-layer quality, PUA contamination, image
coverage). Born-digital pages skip OCR and the VLM entirely. The router generalises
this from binary to N-way.

## The router

A single, config-driven decision point maps **region features → a Route**:

```
Route = { script, paddle_lang, vlm_model, vlm_base_url }
```

Features used (cheap, deterministic, auditable):

- **script** — Thai / Cyrillic / CJK / Arabic / Latin (detected by Unicode-range
  classification of available text; see below)
- **layout class** — paragraph / table / figure (from PP-StructureV3, future) →
  table prompt vs prose
- **rotation**, and **PaddleOCR confidence** — low confidence ≈ handwriting/degraded →
  escalate to a stronger/specialist reader (confidence-gated escalation)

Unmatched scripts fall through to the **generalist default** (Qwen2.5-VL). Adding a
tool is a config row, not a code change.

### Design principles

1. **Prefer deterministic routing signals over a model deciding for itself.** Route on
   *measured* features (script detected, layout class, confidence), not by asking a
   model "what should read you?" — cheaper and auditable. Use a classifier model only
   where deterministic signals genuinely can't distinguish (e.g. handwriting vs print).
2. **Record the routing decision as provenance.** Each region logs *which tools read
   it* — geometry engine (`paddle:th`) and reader (`typhoon-ocr`). Every passage backs
   not just to a box and a model, but to the *reason* that model was chosen
   (defensibility).
3. **Confidence-gated escalation.** Cheap deterministic signals first; escalate to
   expensive specialists only where needed — the same cascade shape used elsewhere in
   the estate (cheap span model → big model only for ambiguous cases).

## Script detection

First cut (deterministic, auditable): classify by **Unicode block counts** over the
text we already have — the page's text layer (even a partial Thai header/footer
counts) or born-digital text. Dominant non-Latin block wins, else Latin.

Pure image-only pages with **no** text layer can't be classified this way; for now
they take the default route. Robust image-only script detection (a fast langid VLM
probe, or a script classifier) is a follow-up.

## Serving specialists — endpoint-agnostic

Because the VLM client speaks the OpenAI-compatible API, a specialist is just a
different `model` name and/or `base_url`. A specialist can be served by **any** means:

- a GGUF in **Ollama** (`ollama pull hf.co/<repo>:<quant>`) — needs the vision mmproj
  projector for VL models;
- a **vLLM** endpoint (the natural in-VPC / transcription-GPU path);

…and the router just points at it. Typhoon OCR (a Qwen2.5-VL fine-tune for Thai/English
documents) is the first specialist; GGUF quants exist (`*/typhoon-ocr-7b-GGUF`, a 3b,
and a newer `typhoon-ocr1.5`).

## Single-Mac vs VPC

You can't keep many 6 GB VLMs hot on one Mac — Ollama swaps them in/out (slow). So
aggressive per-language **VLM** routing is naturally a VPC/CUDA concern (parallel
specialist endpoints). On the Mac MVP, lean on PaddleOCR's per-language recognisers
(small, coexist fine) and accept VLM model-swap cost for the low daily volume. The
router lives behind the job API, so callers don't change as the backend grows from one
model to a fleet.

## Typhoon serving — status (2026-06-25)

Typhoon OCR exists as GGUF with the vision `mmproj` projector
(`kunato/typhoon-ocr1.5-3b-gguf`, `mradermacher/typhoon-ocr-7b-GGUF`). **Blocker:**
`ollama pull hf.co/kunato/typhoon-ocr1.5-3b-gguf` downloaded both blobs but Ollama
0.30.10 returned **`Error: 400`** creating the manifest — its direct HF *vision*-GGUF
import is the fiddly bit. llama.cpp (`llama-server`, the clean Mac path) is not
installed. Options to finish (need a decision / heavier setup): bump/patch Ollama and
retry, hand-author a Modelfile with the GGUF + mmproj, run `llama-server`, or serve via
vLLM on the CUDA path. The router is **ready**: set `[routing.thai] vlm_model/base_url`
to the served endpoint and it's wired — no code change.

**Meanwhile the Thai route already works via PaddleOCR-th.** On the Thai form both
Qwen VLMs failed (2.5 refused, 3 timed out); routing to `paddle_lang=th` reads it at
0.95–1.00 confidence, and the refusal guard discards the generalist's "[Image content
here]" so the Thai `det_text` carries the output (searchable overlay + clean Thai
markdown). Typhoon is now an *enhancement* (better reading/structure), not a
prerequisite.

## Status / roadmap

- [x] Binary triage (text-layer vs OCR) — `triage` stage
- [x] `route`/`language` stage: Unicode-range script detection → Route
- [x] Per-language PaddleOCR recogniser selection in `ocr_det` (Thai verified)
- [x] Per-route VLM model selection in `vlm_read`
- [x] Provenance: `read_by` per segment + per-page script/read_model in the index
- [x] Generalist-refusal guard → fall back to routed det_text (Thai works today)
- [~] Thai route: PaddleOCR-Thai **done**; Typhoon reader blocked on local serving
- [ ] Confidence-gated escalation
- [ ] Image-only script detection (no text layer → currently defaults to Latin)
- [ ] Layout-class routing (with PP-StructureV3)
