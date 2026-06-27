"""HTTP job API (extra: api). The stable contract consumers (Giant, etc.) call —
identical whether the service runs on this desktop or later in a VPC, so moving it
is invisible to callers.

  POST /jobs   (multipart pdf)        -> {sha256, status}
  GET  /jobs/{sha256}                 -> {status, artifacts}

Run: uvicorn fusion_ocr.api:app
"""

from __future__ import annotations

from pathlib import Path

from . import config as config_mod
from .jobs import JobStore
from .pipeline import process, sha256_of


def _safe_name(filename: str | None) -> str:
    """Strip any directory components from a client-supplied upload name. Prevents a
    `filename="../../x"` from escaping in_dir (path traversal)."""
    name = Path(filename or "").name
    return name if name and name not in (".", "..") else "upload.pdf"


def _is_sha256(s: str) -> bool:
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s.lower())


def create_app():  # lazy so the api extra isn't needed unless you serve HTTP
    from fastapi import FastAPI, UploadFile

    cfg = config_mod.load()
    if cfg.airgap:
        config_mod.enforce_airgap()
    jobs = JobStore(Path(cfg.out_dir) / "jobs.sqlite")
    in_dir = Path(cfg.in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="fusion-ocr")

    @app.post("/jobs")
    async def submit(pdf: UploadFile):
        dest = in_dir / _safe_name(pdf.filename)
        dest.write_bytes(await pdf.read())
        digest = sha256_of(dest)
        newly = jobs.upsert_queued(digest, str(dest))
        if newly:
            jobs.set_status(digest, "running")
            try:
                process(dest, cfg)
                jobs.set_status(digest, "done")
            except Exception as exc:  # noqa: BLE001
                jobs.set_status(digest, "error", str(exc))
        row = jobs.get(digest)
        return {"sha256": digest, "status": row["status"] if row else "unknown"}

    @app.get("/jobs/{sha256}")
    def status(sha256: str):
        if not _is_sha256(sha256):           # the path component feeds a filesystem path
            return {"sha256": sha256, "status": "unknown"}
        row = jobs.get(sha256)
        if not row:
            return {"sha256": sha256, "status": "unknown"}
        work = Path(cfg.out_dir) / sha256
        artifacts = [p.name for p in work.iterdir()] if work.exists() else []
        return {"sha256": sha256, "status": row["status"],
                "error": row["error"], "artifacts": artifacts}

    return app


def __getattr__(name: str):
    # Lazy: importing this module stays side-effect-free (no config load, airgap socket
    # patch, or sqlite creation) until a server actually asks for `app`. `uvicorn
    # fusion_ocr.api:app` triggers this via getattr; the helpers stay importable for tests.
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
