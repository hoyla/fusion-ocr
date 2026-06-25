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
    assert r.vlm_model is None  # generalist until a specialist is wired
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
