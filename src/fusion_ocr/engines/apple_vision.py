"""Apple Vision OCR engine (macOS, on-device).

A deterministic engine — boxes + text + confidence, like PaddleOCR — but fast
(Neural Engine), free (no model download), fully on-device (ideal for the airgap
tier), and strong on PRINTED text across ~30 languages. It needs the script as a
language hint, which the router already detects. Loses to the VLM on handwriting /
table structure, so it's the cheap printed-text tier, not a VLM replacement.

`recognize()` returns pixel-space quads in the SAME shape as the PaddleOCR path, so
ocr_det's coordinate handling (scale + derotation) is unchanged.
"""

from __future__ import annotations

import sys

# script -> Apple Vision recognition language codes (it needs the hint; Vision has no
# Devanagari, so that script falls back to PaddleOCR).
VISION_LANGS = {
    "latin": ["en-US"],
    "thai": ["th-TH"],
    "cyrillic": ["ru-RU", "uk-UA"],
    "arabic": ["ar-SA"],
    "cjk": ["zh-Hans", "ja-JP", "ko-KR"],
}


def available() -> bool:
    """True only on macOS with ocrmac installed."""
    if sys.platform != "darwin":
        return False
    try:
        import ocrmac  # noqa: F401
        return True
    except ImportError:
        return False


def recognize(pil_image, languages=None) -> list[tuple[list, str, float]]:
    """[(quad_points_px, text, confidence), ...] in pixel coords (top-left origin)."""
    from ocrmac import ocrmac

    width, height = pil_image.size
    annotations = ocrmac.OCR(
        pil_image, recognition_level="accurate",
        language_preference=languages or None,
    ).recognize()
    return _to_pixel_quads(annotations, width, height)


def _to_pixel_quads(annotations, width: int, height: int) -> list[tuple[list, str, float]]:
    """Map ocrmac (text, conf, (x,y,w,h) normalised bottom-left) -> pixel quads
    (top-left origin), matching the PaddleOCR engine's output shape."""
    out: list[tuple[list, str, float]] = []
    for text, conf, (x, y, w, h) in annotations:
        if not text:
            continue
        px0, px1 = x * width, (x + w) * width
        # Vision is bottom-left origin; flip to top-left to match the image pixels.
        py0, py1 = (1 - (y + h)) * height, (1 - y) * height
        out.append((
            [(px0, py0), (px1, py0), (px1, py1), (px0, py1)], text, float(conf),
        ))
    return out
