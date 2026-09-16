"""Tool routing — pick a (PaddleOCR recogniser, VLM reader) pair per region.

Deterministic and auditable: a region's script is detected by Unicode-range counting,
then mapped to a Route. Geometry always comes from a DETERMINISTIC engine — PaddleOCR,
or Apple Vision when prefer_apple_vision is set (macOS) — never the VLM; only the
recogniser/engine varies. The VLM reader varies by specialist (generalist default, e.g.
Typhoon for Thai). See Docs/routing.md.

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
    engine: str = "paddle"              # deterministic engine: "paddle" | "apple_vision" | "rapidocr"


# script -> default route. vlm_model stays None (generalist) until a specialist is
# served and wired via config; paddle_lang is set per script (safe, improves det_text).
# The toolkit: per-script PaddleOCR recogniser (geometry) + reader (semantics).
# Default reader (unmatched scripts) = the generalist in cfg.vlm (Qwen3-VL/MLX @ 8080).
# Thai's specialist reader = Typhoon-OCR 1.5 (2B) served by mlx-vlm on the SAME MLX
# server (vlm_base_url=None -> cfg.vlm.base_url) — ~2x faster than the old Ollama 3B and
# one server for everything. Typhoon needs its own prompt (vlm/prompts.select_prompt).
# Override / add tools via [routing.<script>] in config.toml.
DEFAULT_ROUTES: dict[str, Route] = {
    "latin":      Route("latin", "en"),
    "thai":       Route("thai", "th", vlm_model="mlx-community/typhoon-ocr1.5-2b-8bit"),
    "cyrillic":   Route("cyrillic", "cyrillic"),
    "arabic":     Route("arabic", "arabic"),
    "cjk":        Route("cjk", "ch"),
    "devanagari": Route("devanagari", "devanagari"),
}

# (script, lo, hi) Unicode ranges — every block a real PDF text layer or an OCR output can
# carry for the script, not only its core block (review 03: the core-only table missed the
# Arabic PRESENTATION FORMS most Arabic PDF text layers actually hold, the CJK extensions
# and compatibility ideographs, and the halfwidth/fullwidth forms). Ranges are disjoint, so
# order doesn't matter. Kept as a table rather than adopting the `regex` module's script
# properties: the sealed tier pins its dependencies, and the table is small and testable.
_BLOCKS = [
    # Thai
    ("thai", 0x0E00, 0x0E7F),
    # Cyrillic: base + supplement + extended-A/B/C + phonetic extensions
    ("cyrillic", 0x0400, 0x052F),
    ("cyrillic", 0x1C80, 0x1C8F),
    ("cyrillic", 0x2DE0, 0x2DFF),
    ("cyrillic", 0xA640, 0xA69F),
    # Arabic: base + supplement + extended-B/A + presentation forms A/B
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("arabic", 0x0870, 0x08FF),
    ("arabic", 0xFB50, 0xFDFF),
    ("arabic", 0xFE70, 0xFEFF),
    # Devanagari: base + extended + extended-A
    ("devanagari", 0x0900, 0x097F),
    ("devanagari", 0xA8E0, 0xA8FF),
    ("devanagari", 0x11B00, 0x11B5F),
    # CJK: ideographs (unified + extension A + compatibility + the SMP extensions B–I),
    # radicals, kana (+ phonetic extensions), bopomofo, hangul (syllables + all jamo blocks),
    # CJK punctuation, and the halfwidth/fullwidth forms (fullwidth Latin/digits and
    # halfwidth kana/hangul are CJK-typeset text, so they count towards CJK routing).
    ("cjk", 0x1100, 0x11FF),     # hangul jamo
    ("cjk", 0x2E80, 0x2FDF),     # CJK radicals supplement, Kangxi radicals
    ("cjk", 0x3001, 0x303F),     # CJK symbols and punctuation (U+3000 ideographic space excluded)
    ("cjk", 0x3040, 0x30FF),     # hiragana / katakana
    ("cjk", 0x3100, 0x312F),     # bopomofo
    ("cjk", 0x3130, 0x318F),     # hangul compatibility jamo
    ("cjk", 0x31F0, 0x31FF),     # katakana phonetic extensions
    ("cjk", 0x3400, 0x4DBF),     # CJK extension A
    ("cjk", 0x4E00, 0x9FFF),     # CJK unified ideographs
    ("cjk", 0xA960, 0xA97F),     # hangul jamo extended-A
    ("cjk", 0xAC00, 0xD7AF),     # hangul syllables
    ("cjk", 0xD7B0, 0xD7FF),     # hangul jamo extended-B
    ("cjk", 0xF900, 0xFAFF),     # CJK compatibility ideographs
    ("cjk", 0xFF01, 0xFFEF),     # halfwidth and fullwidth forms
    ("cjk", 0x20000, 0x323AF),   # CJK extensions B–I
]

# Latin letters beyond ASCII: Latin-1 supplement + Extended-A/B + IPA (00C0–024F) and
# Latin Extended Additional (1E00–1EFF — Vietnamese, transliteration diacritics).
_LATIN_EXTRA = [(0x00C0, 0x024F), (0x1E00, 0x1EFF)]

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
            if ("a" <= ch.lower() <= "z") or any(lo <= o <= hi for lo, hi in _LATIN_EXTRA):
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
    """Pick a non-default deterministic engine when its opt-in flag is set and it's usable
    on this machine; else None (keep the route's default engine). RapidOCR is checked first,
    so an explicit `--rapidocr` / `prefer_rapidocr` benchmark wins over Apple Vision."""
    if cfg is None:
        return None
    if getattr(cfg, "prefer_rapidocr", False):
        from .engines import rapid
        if script in rapid.RAPID_LANGS and rapid.available():
            return "rapidocr"
    if getattr(cfg, "prefer_apple_vision", False):
        from .engines import apple_vision
        if script in apple_vision.VISION_LANGS and apple_vision.available():
            return "apple_vision"
    return None
