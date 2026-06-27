"""The settings registry — a single source of truth for what is *surfaceable* (every
setting, so a consumer / the API can read how the service is configured) and what is
*configurable* at runtime (a safe allowlist).

Why a registry rather than ad-hoc endpoints: the API needs to expose tuning knobs (e.g.
the fusion gate thresholds) without also exposing footguns. Two fields are deliberately
read-only here even though they live on Config:
  - `airgap`  — flipping the seal off over HTTP would defeat the most-sensitive tier.
  - `in_dir`/`out_dir` — paths are identity-critical (jobs/artifacts are keyed off them);
    changing them mid-process desynchronises the store.
`routes` is surfaced read-only too (a nested mapping — edit it in config.toml).

Secrets (`vlm.api_key`) are surfaced masked. Everything output-affecting that IS settable
is also in `recipe_fingerprint` (pipeline.py), so a runtime change re-keys the cache and
re-processes rather than silently reusing a stale result — the same defensibility rule
that governs the rest of resume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_MASK = "***"


@dataclass(frozen=True)
class Setting:
    path: str                       # dotted attribute path on Config, e.g. "vlm.base_url"
    kind: str                       # "bool" | "float" | "str" | "dict"
    settable: bool
    lo: float | None = None         # inclusive float bounds
    hi: float | None = None
    choices: tuple[str, ...] | None = None
    secret: bool = False            # masked when surfaced
    help: str = ""


SETTINGS: tuple[Setting, ...] = (
    Setting("airgap", "bool", settable=False,
            help="sealed (no-egress) tier; read-only — never toggle the seal over HTTP"),
    Setting("in_dir", "str", settable=False, help="input drop dir (identity-critical)"),
    Setting("out_dir", "str", settable=False, help="artifact dir (identity-critical)"),
    Setting("granularity", "str", settable=True, choices=("line", "word"),
            help="overlay box granularity"),
    Setting("overlay_font", "str", settable=True,
            help="path to a Unicode TTF for the overlay; '' = auto-detect"),
    Setting("prefer_apple_vision", "bool", settable=True),
    Setting("apple_vision_skip_vlm", "float", settable=True, lo=0.0, hi=1.0,
            help="skip the VLM read when mean Apple Vision confidence >= this"),
    Setting("table_vlm_read", "bool", settable=True,
            help="route scanned table regions to a focused VLM read"),
    Setting("fuse_min_sim", "float", settable=True, lo=0.0, hi=1.0,
            help="fusion gate: min det<->VLM similarity to accept an aligned line"),
    Setting("fuse_det_conf_trust", "float", settable=True, lo=0.0, hi=1.0,
            help="fusion gate: only refuse a dissimilar line above this det confidence"),
    Setting("vlm.model", "str", settable=True, help="default generalist reader model"),
    Setting("vlm.base_url", "str", settable=True,
            help="OpenAI-compatible reader endpoint (the runtime is a free variable)"),
    Setting("vlm.api_key", "str", settable=True, secret=True),
    Setting("vlm.escalate_below", "float", settable=True, lo=0.0, hi=1.0,
            help="re-read a page with escalation_model when mean confidence < this (0 = off)"),
    Setting("vlm.escalation_model", "str", settable=True),
    Setting("vlm.escalation_base_url", "str", settable=True),
    Setting("routes", "dict", settable=False,
            help="per-script routing overrides; edit in config.toml"),
)

_BY_PATH = {s.path: s for s in SETTINGS}


def _get(cfg, path: str):
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set(cfg, path: str, value) -> None:
    parts = path.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _jsonable(v):
    return str(v) if isinstance(v, Path) else v


def _present(s: Setting, value):
    if s.secret and value not in ("", None):
        return _MASK
    return _jsonable(value)


def _coerce(s: Setting, value):
    """Validate and coerce an incoming value for setting `s`, or raise ValueError."""
    if s.kind == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError(f"{s.path} must be true or false")
    if s.kind == "float":
        try:
            fv = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{s.path} must be a number")
        if (s.lo is not None and fv < s.lo) or (s.hi is not None and fv > s.hi):
            raise ValueError(f"{s.path} must be in [{s.lo}, {s.hi}]")
        return fv
    # str
    if not isinstance(value, str):
        raise ValueError(f"{s.path} must be a string")
    if s.choices and value not in s.choices:
        raise ValueError(f"{s.path} must be one of {list(s.choices)}")
    return value


def surface(cfg) -> list[dict]:
    """Every setting with its current value (secrets masked) and its constraints —
    the read side of get/set."""
    out = []
    for s in SETTINGS:
        entry: dict = {"path": s.path, "value": _present(s, _get(cfg, s.path)),
                       "kind": s.kind, "settable": s.settable}
        if s.lo is not None:
            entry["min"] = s.lo
        if s.hi is not None:
            entry["max"] = s.hi
        if s.choices:
            entry["choices"] = list(s.choices)
        if s.help:
            entry["help"] = s.help
        out.append(entry)
    return out


def apply(cfg, updates: dict) -> dict:
    """Validate and apply {path: value} updates in place, returning the new (masked)
    values for the touched paths. Raises ValueError on any unknown / read-only / invalid
    field — and applies nothing in that case (validate fully before mutating)."""
    if not isinstance(updates, dict) or not updates:
        raise ValueError("body must be a non-empty object of {setting: value}")
    coerced: dict = {}
    for path, value in updates.items():
        s = _BY_PATH.get(path)
        if s is None:
            raise ValueError(f"unknown setting {path!r}")
        if not s.settable:
            raise ValueError(f"{path!r} is read-only (surfaced but not configurable)")
        coerced[path] = _coerce(s, value)
    for path, value in coerced.items():       # mutate only after everything validated
        _set(cfg, path, value)
    return {p: _present(_BY_PATH[p], _get(cfg, p)) for p in coerced}
