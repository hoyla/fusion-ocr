"""Artifact storage seam.

Artifacts are content-addressed under `out/<sha256>/` — the same key shape an object store
uses. Everything that locates a job's output goes through here, so a future object-store
adapter (Garage on-estate, or S3-in-VPC for a less-sensitive tier) is a drop-in rather than
a refactor: only these two functions change. For now it's the local filesystem.
"""

from __future__ import annotations

from pathlib import Path


def job_dir(cfg, digest: str) -> Path:
    """The content-addressed directory for a job's artifacts (and resume snapshots)."""
    return Path(cfg.out_dir) / digest


def artifacts(cfg, digest: str) -> list[str]:
    """Names of the artifacts produced for a job (empty if it hasn't produced any yet)."""
    d = job_dir(cfg, digest)
    return sorted(p.name for p in d.iterdir()) if d.exists() else []
