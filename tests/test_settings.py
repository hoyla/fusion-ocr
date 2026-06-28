"""Settings registry — surface (read every setting) + apply (configure the allowlist),
and the GET/PATCH /config endpoints that wrap them."""

from __future__ import annotations

import pytest

from fusion_ocr import config as config_mod
from fusion_ocr import settings as settings_mod
from fusion_ocr.pipeline import recipe_fingerprint


class _Stage:
    def __init__(self, name):
        self.name = name


def _surfaced(cfg):
    return {e["path"]: e for e in settings_mod.surface(cfg)}


def test_surface_covers_every_setting_with_constraints():
    by_path = _surfaced(config_mod.Config())
    assert by_path["fuse_min_sim"]["value"] == 0.34
    assert by_path["fuse_min_sim"]["settable"] is True
    assert by_path["fuse_min_sim"]["min"] == 0.0 and by_path["fuse_min_sim"]["max"] == 1.0
    assert by_path["granularity"]["choices"] == ["line", "word"]


def test_surface_masks_secrets_and_marks_readonly():
    cfg = config_mod.Config()
    cfg.vlm.api_key = "super-secret"
    by_path = _surfaced(cfg)
    assert by_path["vlm.api_key"]["value"] == "***"      # never leak the key
    assert by_path["airgap"]["settable"] is False         # surfaced but read-only
    assert by_path["in_dir"]["settable"] is False
    assert by_path["routes"]["settable"] is False
    assert by_path["api_host"]["settable"] is False       # bind address is startup-only
    assert by_path["forwarded_allow_ips"]["settable"] is False


def test_apply_sets_allowlisted_value():
    cfg = config_mod.Config()
    out = settings_mod.apply(cfg, {"fuse_min_sim": 0.5})
    assert cfg.fuse_min_sim == 0.5
    assert out == {"fuse_min_sim": 0.5}


def test_apply_sets_nested_vlm_field():
    cfg = config_mod.Config()
    settings_mod.apply(cfg, {"vlm.base_url": "http://localhost:9001/v1"})
    assert cfg.vlm.base_url == "http://localhost:9001/v1"


def test_apply_rejects_readonly_field():
    cfg = config_mod.Config()
    with pytest.raises(ValueError, match="read-only"):
        settings_mod.apply(cfg, {"airgap": False})       # the footgun, refused
    assert cfg.airgap is True


def test_apply_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown setting"):
        settings_mod.apply(config_mod.Config(), {"nope": 1})


def test_apply_rejects_out_of_range_and_wrong_type():
    cfg = config_mod.Config()
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        settings_mod.apply(cfg, {"fuse_min_sim": 1.5})
    with pytest.raises(ValueError, match="true or false"):
        settings_mod.apply(cfg, {"prefer_apple_vision": "yes"})
    with pytest.raises(ValueError, match="one of"):
        settings_mod.apply(cfg, {"granularity": "paragraph"})


def test_apply_is_all_or_nothing():
    # a batch with one bad field must mutate nothing (validate fully, then set)
    cfg = config_mod.Config()
    with pytest.raises(ValueError):
        settings_mod.apply(cfg, {"fuse_min_sim": 0.5, "airgap": False})
    assert cfg.fuse_min_sim == 0.34                       # the good field was not applied


def test_tunable_change_rekeys_recipe_fingerprint():
    # a runtime tuning change must invalidate the resume cache (defensibility): the same
    # PDF re-processes instead of silently reusing a result built with old thresholds.
    pipe = [_Stage("fusion")]
    before = recipe_fingerprint(config_mod.Config(), pipe)
    tuned = config_mod.Config()
    settings_mod.apply(tuned, {"fuse_min_sim": 0.5})
    assert recipe_fingerprint(tuned, pipe) != before


def test_config_save_round_trips(tmp_path):
    pytest.importorskip("tomli_w", reason="needs the api extra")
    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out", airgap=False,
                            api_host="0.0.0.0", api_port=9000,
                            forwarded_allow_ips="10.0.0.0/8")
    settings_mod.apply(cfg, {"fuse_min_sim": 0.5, "vlm.base_url": "http://localhost:9001/v1"})
    back = config_mod.load(config_mod.save(cfg, tmp_path / "config.toml"))
    assert back.fuse_min_sim == 0.5
    assert back.vlm.base_url == "http://localhost:9001/v1"
    assert back.airgap is False                          # read-only fields persist too
    assert back.api_host == "0.0.0.0" and back.api_port == 9000
    assert back.forwarded_allow_ips == "10.0.0.0/8"


def test_config_endpoints_roundtrip(tmp_path):
    pytest.importorskip("fastapi", reason="needs the api extra")
    from fastapi.testclient import TestClient

    from fusion_ocr.api import create_app

    cfg = config_mod.Config(in_dir=tmp_path / "in", out_dir=tmp_path / "out",
                            airgap=False)            # injected -> no socket seal
    client = TestClient(create_app(cfg, token="t"),
                        headers={"Authorization": "Bearer t"})

    got = client.get("/config").json()["settings"]
    paths = {e["path"] for e in got}
    assert "fuse_min_sim" in paths and "vlm.api_key" in paths

    ok = client.patch("/config", json={"fuse_min_sim": 0.42})
    assert ok.status_code == 200 and ok.json() == {"fuse_min_sim": 0.42}
    assert cfg.fuse_min_sim == 0.42                  # mutated the live config

    bad = client.patch("/config", json={"airgap": False})
    assert bad.status_code == 400 and "read-only" in bad.json()["detail"]
