"""Configuration loading + the airgap guard.

`airgap = true` is the contract for the most-sensitive tier: the process must make
no outbound connections. We enforce it defensively by monkeypatching socket
creation to refuse any non-loopback connection, so a stray `requests.get` or model
download can't silently leak. The local VLM endpoint (loopback) still works.
"""

from __future__ import annotations

import ipaddress
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
    # Fusion anti-misalignment gate (see stages/fusion.py). Needleman-Wunsch always pairs
    # a detected cluster with *some* VLM line rather than gapping both, so a confident OCR
    # cluster can be handed a dissimilar line (a reading off-by-one). When the aligned line
    # resembles the ink below fuse_min_sim AND the detector was at least fuse_det_conf_trust
    # sure, that's misalignment not correction -> keep det_text. Gating on confidence is
    # what protects the handwriting path (garbled det_text at low conf, VLM is the truth).
    fuse_min_sim: float = 0.34
    fuse_det_conf_trust: float = 0.80
    # Drop-folder watcher: move a file out of in_dir once handled — processed/ on success,
    # failed/ on error — so the backlog isn't re-hashed on every scan. The watch loop
    # honours this; `--once` never moves (a manual re-run shouldn't disturb the folder).
    move_processed: bool = True
    # API ingest guard: reject an upload larger than this (MB) with 413, before it's hashed.
    max_upload_mb: float = 50.0
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
        fuse_min_sim=run.get("fuse_min_sim", 0.34),
        fuse_det_conf_trust=run.get("fuse_det_conf_trust", 0.80),
        move_processed=run.get("move_processed", True),
        max_upload_mb=run.get("max_upload_mb", 50.0),
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


class AirgapError(OSError):
    """Raised when the airgap guard refuses an outbound connection or DNS lookup.

    Distinct from a generic network error so callers can fail LOUD on a misconfigured
    sensitive tier (e.g. a remote VLM endpoint) instead of silently degrading to
    det_text — which would hide that the reader was unreachable."""


def _is_loopback_host(host) -> bool:
    """True for localhost and the whole loopback range (127/8, ::1, ::ffff:127.0.0.1) —
    not just three literal strings."""
    h = str(host)
    if h == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None) is not None:   # ::ffff:127.0.0.1
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _is_ip_literal(host) -> bool:
    try:
        ipaddress.ip_address(str(host))
        return True
    except ValueError:
        return False


_AIRGAP_ORIG: dict = {}


def enforce_airgap() -> None:
    """Seal the process: refuse any non-loopback connection AND DNS lookup. Idempotent.

    Patches connect, connect_ex (a code path that previously bypassed the guard) and
    getaddrinfo (so a non-loopback hostname can't egress a DNS query before connect
    would refuse it). Only AF_INET/AF_INET6 are guarded — AF_UNIX is local IPC and must
    keep working. Refusals raise AirgapError so callers can surface them loudly.

    Also tells PaddleOCR not to phone home: its model-source connectivity check would
    otherwise hit the network. In airgap mode the OCR models must already be present in
    ~/.paddlex — pre-pull them once on a connected machine, then run sealed.
    """
    import os

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    if getattr(socket.socket, "_fusion_airgapped", False):
        return
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_getaddrinfo = socket.getaddrinfo
    _AIRGAP_ORIG.update(connect=orig_connect, connect_ex=orig_connect_ex,
                        getaddrinfo=orig_getaddrinfo)

    def _refuse_if_remote(sock, address) -> None:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return  # AF_UNIX etc. — local IPC, allowed
        host = address[0] if isinstance(address, tuple) else address
        if not _is_loopback_host(host):
            raise AirgapError(
                f"airgap: outbound connection to {host!r} refused. "
                "Set run.airgap = false to allow remote endpoints.")

    def guarded_connect(self, address):
        _refuse_if_remote(self, address)
        return orig_connect(self, address)

    def guarded_connect_ex(self, address):
        _refuse_if_remote(self, address)
        return orig_connect_ex(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        # IP literals and loopback names resolve locally; a non-loopback hostname would
        # send a DNS query (egress) before connect could refuse it.
        if (isinstance(host, str) and host
                and not _is_ip_literal(host) and not _is_loopback_host(host)):
            raise AirgapError(
                f"airgap: DNS resolution of {host!r} refused (would egress a query). "
                "Use a loopback endpoint, or set run.airgap = false.")
        return orig_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = guarded_connect          # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex    # type: ignore[method-assign]
    socket.getaddrinfo = guarded_getaddrinfo         # type: ignore[assignment]
    socket.socket._fusion_airgapped = True           # type: ignore[attr-defined]


def _disable_airgap() -> None:
    """Restore the patched socket functions. For tests only — production stays sealed
    for the life of the process; this stops the guard leaking across the test run."""
    if not getattr(socket.socket, "_fusion_airgapped", False):
        return
    socket.socket.connect = _AIRGAP_ORIG["connect"]
    socket.socket.connect_ex = _AIRGAP_ORIG["connect_ex"]
    socket.getaddrinfo = _AIRGAP_ORIG["getaddrinfo"]
    del socket.socket._fusion_airgapped
