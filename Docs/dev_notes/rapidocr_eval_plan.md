# RapidOCR (ONNX) vs PaddleOCR — eval plan

*Scaffolding is wired (this branch); the engine inference is a stub. This note is the plan to
flesh it out and decide — by measurement, not assertion — whether to adopt RapidOCR for some or
all of the deterministic engine. Prompted by a perf argument (Gemini) + today's profiling
(`ocr_det` PaddleOCR-on-CPU ≈ 40% of runtime; PaddlePaddle has no Metal/ANE backend).*

## The hypothesis (to test, not assume)

RapidOCR runs the same PP-OCR model **family** exported to ONNX, served by `onnxruntime` —
leaner than PaddlePaddle's CPU path, and able to use the CoreML EP (ANE/GPU). Claim: faster on
Apple Silicon at equal recognition quality, and it sheds the heavy `paddlepaddle` dependency.

This is **not a migration** — it's a third deterministic engine behind the existing routing
seam (`engine = "paddle" | "apple_vision" | "rapidocr"`), adopted **per component** only where
it's *both* faster *and* quality-equal.

## Null hypothesis first: PP-OCRv6 tiny/small (added 2026-07-05, review 03)

**PP-OCRv6 shipped ~June 2026** (PPLCNetV4 backbone; tiny/small/medium tiers): vendor numbers
claim +5.1% recognition / +4.6% detection over PP-OCRv5_server, and — the relevant one — a
**6.1× speedup on Apple M4 for the tiny tier** (0.96s vs 5.82s). If that holds under our eval,
the speed this plan chases may be available *inside the engine we already trust*, with zero
port risk (same PaddleOCR API, no ONNX re-validation, no geometry re-check, layout/table
untouched). So the A/B becomes three-way:

```bash
# add a PP-OCRv6 tiny/small row alongside the existing two, same harness
python -m fusion_ocr.eval --labels ... --no-vlm                    # current Paddle models
python -m fusion_ocr.eval --labels ... --no-vlm  # + PP-OCRv6 tiny/small via model name config
python -m fusion_ocr.eval --labels ... --no-vlm --rapidocr         # RapidOCR
```

**Decision update:** RapidOCR is adopted for det/rec only if it beats *PP-OCRv6 on Paddle*,
not just our current models — the cheaper experiment runs first. Vendor numbers are Baidu's
own; trust the harness, not the release notes. (Layout note: **PP-DocLayoutV3** also now
exists — instance segmentation with a *jointly-trained* reading-order head, superseding V2's
decoupled pointer network. Same "audit the defaults" upgrade path, independent of the engine
question.)

## Make-or-break verification (do FIRST — cheap, decides scope)

1. **det/rec** (DBNet + CRNN/SVTR) — the low-risk, mature part of the ONNX ports. Confirm
   `rapidocr-onnxruntime` exposes our per-script recognisers (RAPID_LANGS), incl. **CJK** and
   the non-Latin scripts we route (Thai/Cyrillic/Arabic), or document the gaps.
2. **Layout reading order** — PP-DocLayoutV2 is RT-DETR + a learned **pointer-network reading
   -order head**. A layout ONNX port may convert the *detector* but **not** the order head.
   Our reading-order quality (Segro 4-col CER 0.02; the FUNSD forms) depends on that head, so
   **verify rapid-layout actually outputs reading order**, not just regions. If it doesn't,
   layout stays on PaddleOCR — det/rec can still move.
3. **Table structure** — confirm rapid-table ships the SLANeXt (wired/wireless) structure we
   use, not only older SLANet.
4. **Model currency** — ONNX ports lag Paddle releases; confirm versions ≈ ours (PP-OCRv6_medium
   / PP-DocLayoutV2 / SLANeXt) or note the delta.

## Flesh-out steps (tomorrow)

1. `pip install -e ".[rapid]"` (adds `rapidocr-onnxruntime` + `onnxruntime`).
2. Implement `engines/rapid.recognize()` — the reference impl is in that file's docstring;
   **check box origin/shape against PaddleOCR on one page** before trusting geometry.
3. (If layout/table move too) add rapid-layout/rapid-table behind the layout/table stages,
   same seam — but only after verification #2/#3 pass.

## Benchmark (reuses the existing eval harness — no new tool)

Engine A/B is already wired into the eval via `--rapidocr` (sets `prefer_rapidocr`). Compare the
deterministic engines head-to-head with `--no-vlm` so the VLM doesn't mask the recogniser:

```bash
# recognition quality + reading order, on the hand-labelled set (Segro 4-col, etc.)
python -m fusion_ocr.eval --labels eval_labels/labelset.json --no-vlm              # PaddleOCR
python -m fusion_ocr.eval --labels eval_labels/labelset.json --no-vlm --rapidocr   # RapidOCR

# recognition vs human GT, on the 3rd-party gold sets
python -m fusion_ocr.eval --dataset funsd --no-vlm [--rapidocr] --limit 30
python -m fusion_ocr.eval --dataset sroie --no-vlm [--rapidocr] --limit 30
```

**Speed** is read from the new per-stage timing (`Document.stage_seconds` in `doc.json`, and
the OCR-Quality batch CSV's `t_ocr_det` column) — compare `ocr_det` seconds between engines on
the same pages. **Equivalence** is the recall / CER columns (and `sRcl`/reading-order on the
labelled set). Add a direct box-IoU check between the two engines' segments if geometry is in
doubt (the overlay + ink-gate depend on it).

## Decision criteria

Adopt RapidOCR for a component **iff**: (a) measurably faster on this hardware, AND (b) recall
within ~0.01 + reading order preserved (no regression on the labelled set) + box geometry
equivalent. Otherwise keep PaddleOCR for that component. det/rec and layout/table are decided
**independently** — a likely outcome is "det/rec on RapidOCR, layout/table stay PaddleOCR" if
the reading-order head doesn't port.

## Caveats already known

- Outputs aren't bit-identical (RapidOCR reimplements resize / DB postprocess / CTC decode) →
  re-validate, don't swap-and-trust; these feed the overlay geometry + the ink-gate.
- CoreML/ANE speedup is **conditional** — OCR uses dynamic input shapes and some ops fall back
  to CPU (can even add partition round-trips). The realistic gain may be "leaner CPU ONNX +
  partial ANE", so measure the EP actually helps before claiming Metal acceleration.
  *(Corroborated by 2026 practitioner reports — review 03's survey: dynamic-shape graphs force
  CPU fallback or subgraph splitting on the CoreML EP. Benchmark the CPU EP first; treat
  CoreML EP as a bonus experiment, not the premise.)*
- Verification #2/#3 update (review 03 survey): **RapidAI/RapidDoc** ships ONNX conversions of
  PP-DocLayoutV3/V2 + table models (SLANet_plus/UNITABLE) with XY-cut-based order recovery —
  so a Paddle-free structure stack *exists*, but it's young (~195 stars, v0.9.x) and whether
  the learned reading-order head survives conversion (vs falling back to XY-cut — the thing we
  deleted) is exactly what #2 must verify before layout moves.
- The `pymupdf4llm` + RapidOCR integration is **out of scope** — it bypasses our fusion /
  ink-gate / provenance architecture (the anti-hallucination design that is the product).
- DPIs/engine are not in `recipe_fingerprint`; if RapidOCR is adopted, fold the engine choice
  into the fingerprint so resume re-keys on it.
