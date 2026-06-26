"""Configuration loading + the airgap guard.

`airgap = true` is the contract for the most-sensitive tier: the process must make
no outbound connections. We enforce it defensively by monkeypatching socket
creation to refuse any non-loopback connection, so a stray `requests.get` or model
download can't silently leak. The local VLM endpoint (loopback) still works.
"""

from __future__ import annotations

import socket
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5vl:7b"  # Ollama's name has no hyphen
    api_key: str = "not-needed-locally"


@dataclass
class Config:
    in_dir: Path = Path("in")
    out_dir: Path = Path("out")
    airgap: bool = True
    granularity: str = "line"
    overlay_font: str = ""  # path to a Unicode TTF for the overlay; "" -> auto-detect
    vlm: VLMConfig = None  # type: ignore[assignment]
    # per-script routing overrides: {script: {paddle_lang, vlm_model, vlm_base_url}}
    routes: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.vlm is None:
            self.vlm = VLMConfig()
        if self.routes is None:
            self.routes = {}


def load(path: str | Path = "config.toml") -> Config:
    p = Path(path)
    if not p.exists():
        # Fall back to defaults (handy for the walking skeleton / tests).
        return Config()
    raw = tomllib.loads(p.read_text())
    run = raw.get("run", {})
    vlm = raw.get("vlm", {})
    return Config(
        in_dir=Path(run.get("in_dir", "in")),
        out_dir=Path(run.get("out_dir", "out")),
        airgap=run.get("airgap", True),
        granularity=run.get("granularity", "line"),
        overlay_font=run.get("overlay_font", ""),
        vlm=VLMConfig(
            base_url=vlm.get("base_url", "http://localhost:11434/v1"),
            model=vlm.get("model", "qwen2.5vl:7b"),
            api_key=vlm.get("api_key", "not-needed-locally"),
        ),
        routes=raw.get("routing", {}),
    )


_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def enforce_airgap() -> None:
    """Refuse any non-loopback socket connection. Idempotent.

    Also tells PaddleOCR not to phone home: its model-source connectivity check
    would otherwise hit the network (and be refused below, aborting OCR). In airgap
    mode the OCR models must already be present in ~/.paddlex — pre-pull them once
    on a connected machine, then run sealed.
    """
    import os

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    if getattr(socket.socket, "_fusion_airgapped", False):
        return
    orig_connect = socket.socket.connect

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in _LOOPBACK:
            raise OSError(
                f"airgap: outbound connection to {host!r} refused. "
                "Set run.airgap = false to allow remote endpoints."
            )
        return orig_connect(self, address)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket._fusion_airgapped = True  # type: ignore[attr-defined]
