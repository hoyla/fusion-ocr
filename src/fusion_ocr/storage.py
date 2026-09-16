"""Artifact storage seam.

Artifacts are content-addressed under `out/<sha256>/` — the same key shape an object store
uses. Everything that locates a job's output goes through here, so a future object-store
adapter (Garage on-estate, or S3-in-VPC for a less-sensitive tier) is a drop-in rather than
a refactor: only these functions change. For now it's the local filesystem.
"""

from __future__ import annotations

import re
from pathlib import Path

# The per-stage resume snapshots (`doc.00-triage.json` … `doc.09-render.json`) are the
# pipeline's cache, not a deliverable — they hold the same Document state as `doc.json`, one
# copy per stage. They live in the job dir for resume but are NOT artifacts to a consumer.
_SNAPSHOT = re.compile(r"^doc\.\d{2}-[^/]+\.json$")


def is_snapshot(name: str) -> bool:
    return bool(_SNAPSHOT.match(name))


def job_dir(cfg, digest: str) -> Path:
    """The content-addressed directory for a job's artifacts (and resume snapshots)."""
    return Path(cfg.out_dir) / digest


def artifacts(cfg, digest: str) -> list[str]:
    """Names of the artifacts produced for a job (empty if it hasn't produced any yet):
    the deliverables (`document.md`, `overlay.pdf`, `segment_index.json`), the final
    `doc.json`, and `source.pdf` for image inputs — never the resume snapshots."""
    d = job_dir(cfg, digest)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file() and not is_snapshot(p.name))


def artifact_path(cfg, digest: str, name: str) -> Path | None:
    """Filesystem path of ONE listed artifact, or None if `name` isn't one of this job's
    artifacts — the only way a remote consumer's `{name}` becomes a path, so a traversal
    or a snapshot name can never resolve."""
    return job_dir(cfg, digest) / name if name in artifacts(cfg, digest) else None
