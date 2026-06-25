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
        dest = in_dir / (pdf.filename or "upload.pdf")
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
        row = jobs.get(sha256)
        if not row:
            return {"sha256": sha256, "status": "unknown"}
        work = Path(cfg.out_dir) / sha256
        artifacts = [p.name for p in work.iterdir()] if work.exists() else []
        return {"sha256": sha256, "status": row["status"],
                "error": row["error"], "artifacts": artifacts}

    return app


app = create_app()
