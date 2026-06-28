"""HTTP job API (extra: api). The stable contract consumers (Giant, etc.) call —
identical whether the service runs on this desktop or later in a VPC, so moving it
is invisible to callers.

  POST  /jobs            (multipart pdf)  -> 202 {sha256, status}   (enqueue; worker drains)
  GET   /jobs            [?status=done]   -> {jobs: [...]}          (queue / 'out' feed)
  GET   /jobs/{sha256}                    -> {status, error, artifacts}
  GET   /config                           -> {settings: [...]}      (every setting, surfaced)
  PATCH /config          {path: value}    -> {path: value}          (configure the allowlist)
  POST  /config/save                      -> {saved: <path>}        (persist to disk, opt-in)

POST /jobs only ENQUEUES — run a worker (`fusion-ocr` watcher) to drain the queue; clients
poll GET /jobs/{sha256}. Run: `fusion-ocr-serve` (the API) alongside `fusion-ocr` (the worker).
"""

# NB: no `from __future__ import annotations` here. The route handlers are closures
# inside create_app() and import UploadFile locally; stringised annotations would leave
# FastAPI with an unresolvable ForwardRef (it resolves against module globals). Eager
# annotations bind UploadFile to the real class at def-time. (str | None still evaluates.)

import os
import secrets
from pathlib import Path

from . import config as config_mod
from . import settings as settings_mod
from . import storage
from .jobs import JobStore
from .pipeline import sha256_of


def _safe_name(filename: str | None) -> str:
    """Strip any directory components from a client-supplied upload name. Prevents a
    `filename="../../x"` from escaping in_dir (path traversal)."""
    name = Path(filename or "").name
    return name if name and name not in (".", "..") else "upload.pdf"


def _is_sha256(s: str) -> bool:
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s.lower())


_PDF_MAGIC = b"%PDF-"
_UPLOAD_CHUNK = 1 << 20   # 1 MiB


async def _save_upload(pdf, dest: Path, max_mb: float, http_exc) -> None:
    """Stream an upload to `dest`, enforcing a size cap and a PDF content sniff so a huge or
    non-PDF body is rejected before it's hashed/processed. Reads in chunks (never the whole
    file into memory), raises http_exc(413) past the cap and http_exc(415) if the bytes
    aren't a PDF, and removes the partial file on any rejection."""
    max_bytes = int(max_mb * 1024 * 1024)
    total, sniffed = 0, False
    try:
        with dest.open("wb") as f:
            while chunk := await pdf.read(_UPLOAD_CHUNK):
                if not sniffed:
                    # Ingest format gate: PDF only today. The future ingest adapter
                    # (Docs/dev_notes/roadmap.md) accepts images / Office docs here and
                    # normalises them to a PDF instead of 415-ing.
                    if _PDF_MAGIC not in chunk[:1024]:
                        raise http_exc(status_code=415, detail="not a PDF (no %PDF- header)")
                    sniffed = True
                total += len(chunk)
                if total > max_bytes:
                    raise http_exc(status_code=413,
                                   detail=f"upload exceeds the {max_mb:g} MB limit")
                f.write(chunk)
        if not sniffed:
            raise http_exc(status_code=415, detail="empty upload")
    except BaseException:
        dest.unlink(missing_ok=True)   # don't leave a partial / oversized file in in/
        raise


def create_app(cfg=None, token=None, config_path="config.toml"):  # lazy: api extra only when serving
    from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile

    if cfg is None:                      # injectable for tests (skips the airgap seal)
        cfg = config_mod.load(config_path)
        if cfg.airgap:
            config_mod.enforce_airgap()
    if token is None:
        token = os.environ.get("FUSION_OCR_API_TOKEN", "")
    if not token:
        # Fail closed: never serve an unauthenticated API. Set FUSION_OCR_API_TOKEN.
        # (The watcher / CLI need no token — they don't go through HTTP.)
        raise RuntimeError(
            "FUSION_OCR_API_TOKEN is not set — refusing to start an unauthenticated API")

    def _require_auth(authorization: str = Header(default="")):
        # constant-time compare so a wrong token can't be timed out character by character
        if not secrets.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    jobs = JobStore(Path(cfg.out_dir) / "jobs.sqlite")
    in_dir = Path(cfg.in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)

    # app-level dependency -> every route requires the bearer token
    app = FastAPI(title="fusion-ocr", dependencies=[Depends(_require_auth)])

    @app.post("/jobs", status_code=202)
    async def submit(pdf: UploadFile):
        # Enqueue only: stream the upload into in/, register it queued, return immediately.
        # A worker (`fusion-ocr` watcher) drains the queue; the client polls GET /jobs/{sha}.
        # The request no longer blocks for the (slow) OCR run.
        dest = in_dir / _safe_name(pdf.filename)
        await _save_upload(pdf, dest, cfg.max_upload_mb, HTTPException)
        digest = sha256_of(dest)
        jobs.upsert_queued(digest, str(dest))
        row = jobs.get(digest)
        return {"sha256": digest, "status": row["status"] if row else "queued"}

    @app.get("/jobs")
    def list_jobs(status: str | None = None):
        # The 'out' feed: queue visibility / pull completed work (?status=done). Poll-based —
        # the only push-free option that also works on the sealed (airgap) tier.
        return {"jobs": [{"sha256": r["sha256"], "status": r["status"], "error": r["error"]}
                         for r in jobs.list(status)]}

    @app.get("/jobs/{sha256}")
    def job_status(sha256: str):
        if not _is_sha256(sha256):           # the path component feeds a filesystem path
            return {"sha256": sha256, "status": "unknown"}
        row = jobs.get(sha256)
        if not row:
            return {"sha256": sha256, "status": "unknown"}
        return {"sha256": sha256, "status": row["status"],
                "error": row["error"], "artifacts": storage.artifacts(cfg, sha256)}

    @app.get("/config")
    def get_config():
        # Surface every setting (secrets masked) so a consumer can see exactly how the
        # service is configured — the read half of the get/set contract.
        return {"settings": settings_mod.surface(cfg)}

    @app.patch("/config")
    def patch_config(updates: dict):
        # Configure the allowlisted settings in-process (affects subsequent jobs; not
        # written back to config.toml). Output-affecting changes re-key recipe_fingerprint,
        # so the next job reprocesses rather than reusing a stale cache.
        try:
            return settings_mod.apply(cfg, updates)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/config/save")
    def save_config():
        # Promote the current in-process config (including any PATCH /config tuning) to
        # disk. Explicit and opt-in: PATCH alone never persists, so a transient experiment
        # can't silently become the on-disk default. Writes a generated file (no comments).
        return {"saved": config_mod.save(cfg, config_path)}

    return app


def main() -> None:
    """`fusion-ocr-serve` — run the job API over HTTP. Host/port come from config
    (`api_host`/`api_port`); the app is built lazily, so the airgap seal and the
    FUSION_OCR_API_TOKEN fail-closed check happen as the worker imports it."""
    import uvicorn

    cfg = config_mod.load()
    where = "localhost only" if cfg.api_host in ("127.0.0.1", "localhost") else "LAN-reachable"
    print(f"[serve] http://{cfg.api_host}:{cfg.api_port}  ({where}; bearer token required)")
    # proxy_headers: behind a TLS-terminating reverse proxy (see Docs/deployment.md), trust
    # X-Forwarded-* only from forwarded_allow_ips so the app sees the real client IP/scheme.
    uvicorn.run("fusion_ocr.api:app", host=cfg.api_host, port=cfg.api_port,
                proxy_headers=True, forwarded_allow_ips=cfg.forwarded_allow_ips)


def __getattr__(name: str):
    # Lazy: importing this module stays side-effect-free (no config load, airgap socket
    # patch, or sqlite creation) until a server actually asks for `app`. `uvicorn
    # fusion_ocr.api:app` triggers this via getattr; the helpers stay importable for tests.
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
