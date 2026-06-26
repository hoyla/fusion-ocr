"""Tool routing — pick a (PaddleOCR recogniser, VLM reader) pair per region.

Deterministic and auditable: a region's script is detected by Unicode-range counting,
then mapped to a Route. PaddleOCR is ALWAYS the geometry engine; only its recogniser
language varies. The VLM reader varies by specialist (generalist default, e.g. Typhoon
for Thai). See Docs/routing.md.

Config overrides: a [routing.<script>] table in config.toml may set paddle_lang,
vlm_model, vlm_base_url for any script (e.g. point Thai at a served Typhoon endpoint).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Route:
    script: str
    paddle_lang: str = "en"
    vlm_model: str | None = None       # None -> use cfg.vlm.model (generalist)
    vlm_base_url: str | None = None     # None -> use cfg.vlm.base_url
    engine: str = "paddle"              # deterministic engine: "paddle" | "apple_vision"


# script -> default route. vlm_model stays None (generalist) until a specialist is
# served and wired via config; paddle_lang is set per script (safe, improves det_text).
# The toolkit: per-script PaddleOCR recogniser (geometry) + reader (semantics).
# Default reader (unmatched scripts) = the generalist in cfg.vlm (Qwen3-VL/MLX @ 8080).
# Thai is special-cased to the Typhoon specialist, served on Ollama (its own endpoint);
# Typhoon needs its own prompt (handled by vlm/prompts.select_prompt). If Typhoon isn't
# running the call fails and the refusal guard falls back to PaddleOCR-th det_text.
# Override / add tools via [routing.<script>] in config.toml.
_OLLAMA = "http://localhost:11434/v1"
DEFAULT_ROUTES: dict[str, Route] = {
    "latin":      Route("latin", "en"),
    "thai":       Route("thai", "th",
                        vlm_model="scb10x/typhoon-ocr1.5-3b:latest", vlm_base_url=_OLLAMA),
    "cyrillic":   Route("cyrillic", "cyrillic"),
    "arabic":     Route("arabic", "arabic"),
    "cjk":        Route("cjk", "ch"),
    "devanagari": Route("devanagari", "devanagari"),
}

# (name, lo, hi) Unicode ranges. Order matters only for disjoint ranges.
_BLOCKS = [
    ("thai", 0x0E00, 0x0E7F),
    ("cyrillic", 0x0400, 0x04FF),
    ("arabic", 0x0600, 0x06FF),
    ("devanagari", 0x0900, 0x097F),
    ("cjk", 0x4E00, 0x9FFF),
    ("cjk", 0x3040, 0x30FF),   # hiragana / katakana
    ("cjk", 0xAC00, 0xD7AF),   # hangul
]

_NONLATIN_MIN_SHARE = 0.10  # a non-Latin script must be >=10% of letters to win


def detect_script(text: str) -> str:
    """Classify the dominant script of ``text`` by Unicode-block counts.

    Latin (incl. Latin-1/Extended diacritics, e.g. Montenegrin Š/Ž/Č) is the default;
    a non-Latin block must clear a share threshold to win, so a stray foreign glyph
    doesn't misroute an otherwise-Latin page."""
    counts: dict[str, int] = {}
    latin = 0
    for ch in text:
        o = ord(ch)
        for name, lo, hi in _BLOCKS:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            if ("a" <= ch.lower() <= "z") or (0x00C0 <= o <= 0x024F):
                latin += 1
    total = latin + sum(counts.values())
    if total == 0:
        return "latin"
    if counts:
        top = max(counts, key=lambda k: counts[k])
        if counts[top] / total >= _NONLATIN_MIN_SHARE:
            return top
    return "latin"


def resolve(script: str, cfg=None) -> Route:
    """Resolve a script to a Route, applying any config [routing.<script>] overrides."""
    base = DEFAULT_ROUTES.get(script, DEFAULT_ROUTES["latin"])
    overrides = getattr(cfg, "routes", {}) or {}
    o = overrides.get(script, {})
    engine = o.get("engine") or _auto_engine(base.script, cfg) or base.engine
    return Route(
        script=base.script,
        paddle_lang=o.get("paddle_lang", base.paddle_lang),
        vlm_model=o.get("vlm_model", base.vlm_model),
        vlm_base_url=o.get("vlm_base_url", base.vlm_base_url),
        engine=engine,
    )


def _auto_engine(script: str, cfg) -> str | None:
    """Prefer Apple Vision (fast, on-device) when enabled and the script is supported
    on this machine; else None (keep the route's default engine)."""
    if cfg is None or not getattr(cfg, "prefer_apple_vision", False):
        return None
    from .engines import apple_vision
    if script in apple_vision.VISION_LANGS and apple_vision.available():
        return "apple_vision"
    return None
