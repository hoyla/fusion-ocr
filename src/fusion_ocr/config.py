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
    # Default generalist reader = Qwen3-VL-8B-Instruct served by mlx-vlm (MLX is far
    # faster than Ollama/llama.cpp on Apple Silicon). Start the server with:
    #   python -m mlx_vlm.server --port 8080
    # Specialists (e.g. Typhoon for Thai) are routed to their own endpoints — see
    # routing.py / [routing.<script>]. If the server is down the call fails and fusion
    # falls back to PaddleOCR det_text.
    base_url: str = "http://localhost:8080/v1"
    model: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    api_key: str = "not-needed-locally"
    # Confidence-gated escalation: when set, a page whose mean PaddleOCR confidence is
    # below `escalate_below` (or whose primary read looks like a refusal) is re-read by
    # `escalation_model`. 0.0 / "" disables it.
    escalate_below: float = 0.0
    escalation_model: str = ""
    escalation_base_url: str = ""


@dataclass
class Config:
    in_dir: Path = Path("in")
    out_dir: Path = Path("out")
    airgap: bool = True
    granularity: str = "line"
    overlay_font: str = ""  # path to a Unicode TTF for the overlay; "" -> auto-detect
    # Apple Vision (macOS, on-device) as the fast deterministic engine for supported
    # scripts. When its mean confidence on a page is >= apple_vision_skip_vlm, the VLM
    # read is skipped (Vision's text IS the reading — the cheap tier); harder pages
    # still escalate to the VLM.
    prefer_apple_vision: bool = False
    apple_vision_skip_vlm: float = 0.92
    # Route detected table regions on scanned pages to a focused VLM table read (crop +
    # table prompt), regardless of the page-level read — tables are structure that line-
    # OCR/Apple Vision handle poorly. Geometry still comes from the deterministic grid;
    # this supplies clean cell content. Born-digital tables are left to the text layer.
    table_vlm_read: bool = True
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
        prefer_apple_vision=run.get("prefer_apple_vision", False),
        apple_vision_skip_vlm=run.get("apple_vision_skip_vlm", 0.92),
        table_vlm_read=run.get("table_vlm_read", True),
        vlm=VLMConfig(
            base_url=vlm.get("base_url", "http://localhost:8080/v1"),
            model=vlm.get("model", "mlx-community/Qwen3-VL-8B-Instruct-4bit"),
            api_key=vlm.get("api_key", "not-needed-locally"),
            escalate_below=vlm.get("escalate_below", 0.0),
            escalation_model=vlm.get("escalation_model", ""),
            escalation_base_url=vlm.get("escalation_base_url", ""),
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
