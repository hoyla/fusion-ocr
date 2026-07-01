"""RapidOCR engine (ONNX Runtime) — WIRED BUT NOT YET IMPLEMENTED (eval scaffolding).

RapidOCR runs the same PP-OCR model FAMILY as PaddleOCR, but exported to ONNX and served by
`onnxruntime` (leaner than PaddlePaddle's CPU path, and able to use the CoreML execution
provider / ANE). The hypothesis we want to TEST — not assume — is that it's faster on Apple
Silicon at equal recognition quality, letting us shed the heavy `paddlepaddle` dependency.

This module is a THIRD deterministic engine behind the existing routing seam (`engine =
"paddle" | "apple_vision" | "rapidocr"`), so it's an A/B option, not a migration. `recognize()`
is a STUB on purpose: the wiring (config flag, routing, ocr_det dispatch, eval `--rapidocr`,
the `rapid` extra) is in place so tomorrow's work is just (1) `pip install -e ".[rapid]"`,
(2) flesh out `recognize()` below, (3) run the benchmark. See
Docs/dev_notes/rapidocr_eval_plan.md for the verification checklist + decision criteria.

CAVEATS to settle during the eval (why this is det/rec-first, not a wholesale swap):
- det/rec (DBNet + CRNN/SVTR) is the low-risk, mature part of the ONNX ports — start here.
- LAYOUT (`PP-DocLayoutV2`) carries a learned reading-order POINTER NETWORK, not just region
  detection; a layout ONNX port may convert the detector but NOT the order head. Our
  reading-order quality (Segro 4-col CER 0.02, the FUNSD forms) depends on that head — so the
  layout/table swap must PROVE it keeps reading order before we believe "identical".
- Outputs are not bit-identical (RapidOCR reimplements resize / DB postprocess / CTC decode),
  so re-validate against the eval set — these feed the overlay geometry + the ink-gate.
"""

from __future__ import annotations

# script -> RapidOCR/PP-OCR recogniser language key (same script hints the router detects).
# Filled to match whatever the chosen ONNX port exposes; mirrors PaddleOCR's per-script lang.
RAPID_LANGS = {
    "latin": "en",
    "thai": "th",
    "cyrillic": "cyrillic",
    "arabic": "arabic",
    "cjk": "ch",
    "devanagari": "devanagari",
}


def available() -> bool:
    """True if RapidOCR is importable. False keeps routing on PaddleOCR, so selecting
    `--rapidocr` before `pip install -e .[rapid]` is a silent no-op, not a crash. Prefers the
    current unified `rapidocr` (3.x); the legacy `rapidocr_onnxruntime` name is a fallback."""
    try:
        import rapidocr  # noqa: F401  (current package, 3.x)
        return True
    except ImportError:
        try:
            import rapidocr_onnxruntime  # noqa: F401  (legacy 1.x, frozen — avoid)
            return True
        except ImportError:
            return False


def recognize(pil_image, script: str | None = None) -> list[tuple[list, str, float]]:
    """[(quad_points_px, text, confidence), ...] in pixel coords (top-left origin) — the SAME
    shape PaddleOCR/Apple Vision return, so ocr_det's coordinate handling stays unchanged.

    NOT IMPLEMENTED — this is the one piece tomorrow fills in. Reference impl (RapidOCR 3.x;
    confirm against the installed `rapidocr.__version__`, the API moves between majors):

        from rapidocr import RapidOCR
        import numpy as np
        engine = RapidOCR()                        # cache per-process (models load here)
        result = engine(np.asarray(pil_image))     # RapidOCROutput | None
        out = []
        if result is not None and getattr(result, "boxes", None) is not None:
            # result.boxes: np.ndarray (N, 4, 2) px, top-left origin; .txts / .scores parallel
            for box, text, score in zip(result.boxes, result.txts, result.scores):
                if not text:
                    continue
                out.append(([(float(x), float(y)) for x, y in box], text, float(score)))
        return out

    Notes: guard the None/empty return (RapidOCR returns None on a blank page). To pick the
    recogniser language, configure RapidOCR with the model for RAPID_LANGS[script]. Verify box
    origin/shape against PaddleOCR on one page before trusting the geometry (see the eval plan).
    """
    raise NotImplementedError(
        "RapidOCR engine is wired but not implemented — flesh out engines/rapid.recognize() "
        "and `pip install -e .[rapid]` first (Docs/dev_notes/rapidocr_eval_plan.md)."
    )
