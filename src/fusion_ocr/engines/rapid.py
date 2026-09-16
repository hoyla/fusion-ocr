"""RapidOCR engine (ONNX Runtime) — the A/B engine (implemented 2026-08-20; not adopted).

RapidOCR runs the same PP-OCR model FAMILY as PaddleOCR, but exported to ONNX and served by
`onnxruntime` (leaner than PaddlePaddle's CPU path, and able to use the CoreML execution
provider / ANE). The hypothesis TESTED — not assumed — is that it's faster on Apple Silicon at
equal recognition quality, letting us shed the heavy `paddlepaddle` dependency. Numbers and the
verdict: eval_out/manifests/engine_ab_2026-08-20.md (runner eval_out/engine_ab.py).

This module is a THIRD deterministic engine behind the existing routing seam (`engine =
"paddle" | "apple_vision" | "rapidocr"`), so it's an A/B option, not a migration: inert
unless `prefer_rapidocr` is set AND the `rapid` extra is installed. Its segments carry
`source="rapid"` and are treated like any OCR box downstream (models.OCR_SOURCES). See
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
    """True if a RapidOCR ONNX runtime is importable. False keeps routing on PaddleOCR, so
    selecting `--rapidocr` before `pip install -e .[rapid]` is a silent no-op, not a crash."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        try:
            import rapidocr  # noqa: F401  (newer package name)
            return True
        except ImportError:
            return False


_ENGINE = None  # per-process cache, like ocr_det._engine_for
_REC_TIER: str | None = None  # None = the package default (v6 small)
_DET_TIER: str | None = None  # None = the package default (v6 small)


def set_rec_tier(tier: str | None) -> None:
    """Select the recognition-model tier ('tiny'|'small'|'medium'|None=default) for the
    NEXT engine build — the engine A/B knob (rapidocr >= 2 only; the RapidAI zoo ships
    SHA256-pinned PP-OCRv6 tiers). Clears the cached engine so the change takes effect.
    NB: an EVAL-side switch, invisible to the recipe fingerprint — adoption of a non-default
    tier must promote it to a fingerprinted config knob (rapidocr_eval_plan.md)."""
    global _ENGINE, _REC_TIER
    if tier != _REC_TIER:
        _ENGINE, _REC_TIER = None, tier


def set_det_tier(tier: str | None) -> None:
    """Same knob for the DETECTION model tier (the n=30 A/B's medrec arm kept the small
    detector; the full set showed a detection-side tail on forms — engine_ab_verdict.py)."""
    global _ENGINE, _DET_TIER
    if tier != _DET_TIER:
        _ENGINE, _DET_TIER = None, tier


def _engine():
    global _ENGINE
    if _ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR   # legacy package name (1.x)
            _ENGINE = RapidOCR()                        # 1.x: no tier selection
        except ImportError:
            from rapidocr import ModelType, OCRVersion, RapidOCR  # current name (>= 2)
            params = {}
            if _REC_TIER:
                params.update({"Rec.model_type": ModelType(_REC_TIER),
                               "Rec.ocr_version": OCRVersion.PPOCRV6})
            if _DET_TIER:
                params.update({"Det.model_type": ModelType(_DET_TIER),
                               "Det.ocr_version": OCRVersion.PPOCRV6})
            _ENGINE = RapidOCR(params=params)
    return _ENGINE


def recognize(pil_image, script: str | None = None) -> list[tuple[list, str, float]]:
    """[(quad_points_px, text, confidence), ...] in pixel coords (top-left origin) — the SAME
    shape PaddleOCR/Apple Vision return, so ocr_det's coordinate handling stays unchanged.

    Implemented 2026-08-20 for the engine A/B (rapidocr_eval_plan.md); geometry verified
    against PaddleOCR on a FUNSD page (box-IoU check in the A/B runner) before the benchmark
    was trusted. Verification #1 gap, documented: `rapidocr_onnxruntime` 1.x ships the
    default en/ch det+rec models only — the per-script recognisers in RAPID_LANGS
    (Thai/Cyrillic/Arabic/devanagari) require supplying per-language rec model files, which
    the pip package does not bundle. So `script` is currently unused: non-Latin routes must
    stay on PaddleOCR until per-language ONNX rec models are sourced and validated.
    """
    import numpy as np

    raw = _engine()(np.asarray(pil_image))
    # rapidocr >= 2 returns a RapidOCROutput (.boxes/.txts/.scores); the legacy 1.x
    # package returned ([[box, text, score], ...], elapse). Normalise both.
    if hasattr(raw, "boxes"):
        if raw.boxes is None:
            return []
        triples = zip(raw.boxes, raw.txts or [], raw.scores or [])
    else:
        triples = raw[0] or []
    out = []
    for box, text, score in triples:
        if not text:
            continue
        # box is 4 [x, y] points already in pixel space (top-left origin) — pass through
        out.append(([(float(x), float(y)) for x, y in box], text, float(score)))
    return out
