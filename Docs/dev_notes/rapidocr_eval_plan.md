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

## Make-or-break verification (do FIRST — cheap, decides scope)

1. **det/rec** (DBNet + CRNN/SVTR) — the low-risk, mature part of the ONNX ports. **Language
   support is confirmed** (RapidOCR model list): the single `rapidocr` package covers all our
   scripts — `LangRec` has `latin`, `th`, `cyrillic`, `arabic`, `ch` (+ `japan`/`korean`),
   `devanagari` — so **no per-language pip extra** is needed; the recogniser is chosen by
   `params={"Rec.lang_type": ...}` and its ONNX model **auto-downloads from ModelScope** on first
   use (RAPID_LANGS already maps our scripts). Confirm each language actually resolves on the
   installed version, and note any that need a specific `Rec.ocr_version` (e.g. Thai → PP-OCRv5).
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

1. `pip install -U -e ".[rapid]"` — the extra pins **`rapidocr>=3.9,<4`** (the *current* unified
   package; the old `rapidocr-onnxruntime` 1.4.4 is frozen from Jan 2025 — don't use it) plus
   `onnxruntime`. Run `python -c "import rapidocr; print(rapidocr.__version__)"` and confirm it's
   the latest 3.x (was 3.9.0, 2026-06-23; check PyPI for newer) — the API can shift between
   majors, so re-check the result-object shape (`.boxes`/`.txts`/`.scores`) against the docstring.
2. Implement `engines/rapid.recognize()` — the reference impl (RapidOCR 3.x) is in that file's
   docstring; **check box origin/shape against PaddleOCR on one page** before trusting geometry,
   and keep the None/empty-return guard.
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
- **Airgap: RapidOCR auto-downloads models from ModelScope on first use** — that's egress the
  sealed tier can't do. If RapidOCR is adopted, the per-language ONNX models (det + each script's
  rec) must be **pre-pulled and cached before sealing**, a RapidOCR pre-pull list analogous to the
  PaddleX one (PP-OCRv5_server_det, per-lang recognisers, PP-DocLayoutV2, SLANeXt…). Point
  RapidOCR at the local model paths (or prime its cache) so no download is attempted under airgap.
- The `pymupdf4llm` + RapidOCR integration is **out of scope** — it bypasses our fusion /
  ink-gate / provenance architecture (the anti-hallucination design that is the product).
- DPIs/engine are not in `recipe_fingerprint`; if RapidOCR is adopted, fold the engine choice
  into the fingerprint so resume re-keys on it.
