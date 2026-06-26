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

## Mixed-content pages — retain & combine both sets (2026-06-26)

A page is often part machine-readable text layer and part image — e.g. a scanned
document with a digital header/footer/stamp imposed over it (the Thai "scan" docs). A
naive "has a text layer? → born-digital" check reads the header and misses the scanned
body. Instead the pipeline combines BOTH, deciding per layout region:

- `compose.classify_regions` tags each region **machine-readable** (clean text covers
  ≥ 50% of it → use the text layer verbatim) or **ocr** (image → read it);
- fusion keeps the clean text layer for machine-readable regions and the OCR/VLM for
  image regions, **superseding** (not dropping) the weaker source where they overlap —
  redundant OCR over a machine-readable region, or contaminated text layer over an
  OCR'd region. Superseded segments are retained with `superseded: true` for provenance
  (principle: never discard source);
- the combined output is emitted in **reading order** (region order, then position).

Verified on the Thai scan: 4 regions machine-readable (header/footer), 17 OCR (body);
output combines `textlayer` + `vision`; 7 superseded segments retained. Every segment
carries its `source`, so any span backs to "was selectable in the PDF" vs "OCR'd from
the image".

## Apple Vision — the fast on-device tier (macOS, 2026-06-26)

A deterministic engine (boxes + text + confidence, like PaddleOCR) but **fast**
(Neural Engine, sub-2s/page), **free** (no model download), **fully on-device** (no
server, ideal for the airgap tier), across **~30 languages** (en, Thai, Cyrillic,
Arabic, CJK, …). It needs the script as a language hint — which the router already
detects. It loses to the VLM on handwriting and table *structure*, so it's the cheap
printed-text tier, not a VLM replacement.

Benchmarked: clean print 0.75s @ 0.97; Montenegrin (diacritics) 0.68s @ 0.92; Thai
1.84s @ 0.95 (≈ Typhoon, ~20× faster); handwriting — garbled (VLM wins).

Wiring (`engines/apple_vision.py`, `vision` extra = `ocrmac`):
- `run.prefer_apple_vision = true` → the router uses Vision as the deterministic
  engine for Vision-supported scripts on macOS (Devanagari / non-Mac → PaddleOCR);
- **cheap tier**: when a page's mean Vision confidence ≥ `run.apple_vision_skip_vlm`,
  the VLM read is **skipped** (Vision's text IS the reading); harder pages fall through
  to the VLM (and confidence-gated escalation). Verified: the Thai scan runs **7.8s with
  no VLM call** (vs ~84s via Typhoon), fully on-device.
- Per-route override: `[routing.<script>] engine = "apple_vision"`.

## Runtime — MLX vs Ollama on Apple Silicon (2026-06-26)

The reader is endpoint-agnostic, so the *serving runtime* is a free variable — and on
Apple Silicon it matters a lot. An earlier bake-off found Qwen3-VL "6–10× slower,
timed out" — but that was **Ollama (llama.cpp)**. Re-run via **MLX** (`mlx-vlm`, whose
`mlx_vlm.server` is OpenAI-compatible — point `vlm.base_url` at it, no code change):

| Page | Ollama qwen3-vl:8b | MLX Qwen3-VL-8B-Instruct-4bit |
| --- | --- | --- |
| handwriting | 127s | **10.5s** (cleanest read of all) |
| Thai form | failed | **28.9s** (reads Thai ≈ Typhoon) |
| table | timed out | **15.0s** (correct values) |

~10–40× faster and accurate across the board, Thai included. **The earlier verdict was
a runtime artifact, not the model.** Takeaways:

- **MLX is the preferred local runtime on Apple Silicon**; serve readers via
  `python -m mlx_vlm.server` and set `vlm.base_url = http://localhost:8080/v1`.
- **Qwen3-VL-8B-Instruct (MLX)** is a strong, fast, multilingual generalist — a
  candidate to replace the Qwen2.5-VL/Ollama default, and a viable Thai reader
  alongside Typhoon. (Use **Instruct**, not Thinking, for OCR; recommended sampling
  temp 0.7 / top-p 0.8 / top-k 20, though greedy is fine for transcription.)
- Qwen3.5-VL exists (text MLX builds published; VL MLX build not yet) — re-test when
  available.
- Lesson (again): benchmark the *runtime*, not just the model.

## Single-Mac vs VPC

You can't keep many 6 GB VLMs hot on one Mac — Ollama swaps them in/out (slow). So
aggressive per-language **VLM** routing is naturally a VPC/CUDA concern (parallel
specialist endpoints). On the Mac MVP, lean on PaddleOCR's per-language recognisers
(small, coexist fine) and accept VLM model-swap cost for the low daily volume. The
router lives behind the job API, so callers don't change as the backend grows from one
model to a fleet.

## Thai route — WORKING (2026-06-26)

Thai is the first fully-wired specialist route:

- **geometry**: PaddleOCR `lang=th` reads the Thai form at 0.95–1.00 confidence
  (both Qwen VLMs had failed it — 2.5 refused, 3 timed out);
- **reading**: **Typhoon OCR** (`ollama pull scb10x/typhoon-ocr1.5-3b` — SCB 10X's
  own community-published Ollama model; this is the correct install, *not* a raw HF
  GGUF pull, which Ollama 0.30.10 won't import). Default reader for `script=thai`.

Typhoon is fine-tuned on its own instruction, so a generic transcribe prompt makes it
echo its template — `vlm/prompts.py::select_prompt(model)` sends the Typhoon-format
prompt (literal-text variant: tables/page-numbers preserved, figure-DESCRIPTION/chart-
ANALYSIS dropped — that's inference, not OCR). Verified end-to-end: clean structured
Thai markdown, Typhoon lines aligned onto PaddleOCR-th boxes (Thai↔Thai fuzzy match),
`read_by` provenance recorded.

If Typhoon isn't installed the call fails → the **refusal guard** falls back to
PaddleOCR-th `det_text`, so the route degrades gracefully (no hard dependency).

**Thai overlay search — FIXED (2026-06-26).** Root cause was the overlay *font*, not
normalisation: the base-14 "helv" font can't encode Thai, so `search_for` missed (0/4
terms) even though `get_text` returned the Thai. Using a broad Unicode font (Arial
Unicode on macOS, or a configured TTF via `run.overlay_font`) makes all terms
searchable (verified end-to-end: หนังสือบริคณห์สนธิ / บริษัท / กระทรวงพาณิชย์ all
hit). Text is NFC-normalised too. Falls back to helv (Latin only) if no Unicode font
is present.

## Status / roadmap

- [x] Binary triage (text-layer vs OCR) — `triage` stage
- [x] `route`/`language` stage: Unicode-range script detection → Route
- [x] Per-language PaddleOCR recogniser selection in `ocr_det` (Thai verified)
- [x] Per-route VLM model selection in `vlm_read`
- [x] Provenance: `read_by` per segment + per-page script/read_model in the index
- [x] Generalist-refusal guard → fall back to routed det_text
- [x] Per-model prompt selection (`select_prompt`) — Typhoon's own instruction
- [x] Thai route: PaddleOCR-Thai geometry + **Typhoon reader** — done & verified
- [x] Thai overlay search reliability — Unicode overlay font + NFC (verified)
- [x] Layout detection (PP-DocLayout) → regions + region-aware line clustering
- [x] Table-cell extraction (TableStructureRecognition) → HTML grid + cell boxes
- [x] Confidence-gated escalation (low PaddleOCR conf / refusal → stronger model)
- [x] Image-only script detection (one-word VLM probe for no-text-layer pages)
- [ ] True multi-column reading order (PP-StructureV3 reading-order model)
- [ ] Layout-class routing (route table regions to a table-aware reader)
- [ ] Per-cell table content (intersect cell boxes with OCR/VLM text)
