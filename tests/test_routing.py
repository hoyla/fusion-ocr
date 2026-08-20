"""Routing — script detection + route resolution. No deps."""

from __future__ import annotations

from types import SimpleNamespace

from fusion_ocr.routing import DEFAULT_ROUTES, detect_script, resolve


def test_detect_latin_with_diacritics():
    # Montenegrin: Latin + Š/Ž/Č diacritics -> still latin.
    assert detect_script("RJEŠENJE DRUŠTVO Skraćeni naziv PRIVATE FAMILY OFFICE") == "latin"


def test_detect_thai():
    assert detect_script("สำเนาเอกสารฉบับนี้ กรมพัฒนาธุรกิจการค้า") == "thai"


def test_detect_cyrillic():
    assert detect_script("Пореска управа Црна Гора Подгорица") == "cyrillic"


def test_detect_empty_and_punctuation_default_latin():
    assert detect_script("") == "latin"
    assert detect_script("   123 .,;  ") == "latin"


def test_resolve_defaults():
    r = resolve("thai", cfg=None)
    assert r.paddle_lang == "th"
    assert "typhoon" in r.vlm_model.lower()  # Thai specialist reader by default
    assert resolve("latin", cfg=None).vlm_model is None  # generalist
    assert resolve("latin", cfg=None).paddle_lang == "en"
    assert resolve("klingon", cfg=None).paddle_lang == "en"  # unknown -> latin default


def test_resolve_config_override():
    cfg = SimpleNamespace(routes={
        "thai": {"vlm_model": "typhoon-ocr", "vlm_base_url": "http://gpu:8000/v1"}
    })
    r = resolve("thai", cfg)
    assert r.paddle_lang == "th"            # kept from default
    assert r.vlm_model == "typhoon-ocr"     # from override
    assert r.vlm_base_url == "http://gpu:8000/v1"


def test_default_routes_cover_expected_scripts():
    for s in ("latin", "thai", "cyrillic", "arabic", "cjk"):
        assert s in DEFAULT_ROUTES


def test_rapidocr_engine_seam(monkeypatch):
    # Default: PaddleOCR, regardless of whether rapid is importable.
    assert resolve("latin", cfg=None).engine == "paddle"
    assert resolve("latin", SimpleNamespace(prefer_rapidocr=False)).engine == "paddle"

    # prefer_rapidocr routes to "rapidocr" only when the engine reports available; if the extra
    # isn't installed it stays on PaddleOCR (a silent no-op, not a crash).
    from fusion_ocr.engines import rapid
    monkeypatch.setattr(rapid, "available", lambda: True)
    assert resolve("latin", SimpleNamespace(prefer_rapidocr=True)).engine == "rapidocr"
    monkeypatch.setattr(rapid, "available", lambda: False)
    assert resolve("latin", SimpleNamespace(prefer_rapidocr=True)).engine == "paddle"


def test_rapidocr_recognize_returns_engine_shape():
    # Implemented 2026-08-20 (engine A/B): recognize() must return the shared engine shape
    # [(quad_points_px, text, conf), ...] — a blank page yields an empty list (engine loads,
    # inference path runs, nothing detected), never an exception.
    import pytest

    pytest.importorskip("rapidocr", reason="rapid extra not installed")
    from PIL import Image

    from fusion_ocr.engines import rapid
    lines = rapid.recognize(Image.new("RGB", (200, 100), "white"))
    assert isinstance(lines, list)
    for pts, text, conf in lines:   # shape check on anything detected
        assert len(pts) == 4 and isinstance(text, str) and 0.0 <= conf <= 1.0
