"""Pipeline orchestration + the Stage contract.

Every stage is `Document in -> Document out`. The Document is serialised to
out/<sha256>/doc.json after each stage, so a crash or a deliberate re-run with a
refined prompt resumes from the last completed stage instead of redoing OCR.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from . import storage
from .config import Config
from .models import Document
from .stages.fusion import Fusion
from .stages.language import Language
from .stages.layout import Layout
from .stages.ocr_det import OcrDet
from .stages.render import Render
from .stages.table import Table
from .stages.table_fill import TableFill
from .stages.table_read import TableRead
from .stages.triage import Triage
from .stages.vlm_read import VlmRead


@runtime_checkable
class Stage(Protocol):
    name: str

    def run(self, doc: Document, cfg: Config) -> Document: ...


DEFAULT_PIPELINE: list[Stage] = [
    Triage(),
    Layout(),
    Table(),
    Language(),
    OcrDet(),
    VlmRead(),
    TableRead(),
    Fusion(),
    TableFill(),
    Render(),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recipe_fingerprint(cfg: Config, pipeline: list[Stage]) -> str:
    """Hash of everything that determines the OUTPUT — pipeline shape, reader model and
    endpoints, routing, the prompt text, and output-affecting flags. NOT in_dir / out_dir
    / airgap (they don't change content). A change here invalidates the cache, so a
    re-run after tuning a prompt or model reprocesses instead of silently reusing.

    Caveat: arbitrary CODE edits (e.g. a threshold constant inside a stage) are not
    captured — pass force=True after such a change."""
    from .vlm import prompts

    payload = {
        "pipeline": [s.name for s in pipeline],
        "vlm": {"model": cfg.vlm.model, "base_url": cfg.vlm.base_url,
                "escalation_model": cfg.vlm.escalation_model,
                "escalate_below": cfg.vlm.escalate_below,
                "escalation_base_url": cfg.vlm.escalation_base_url},
        "routes": cfg.routes,
        "flags": {"prefer_apple_vision": cfg.prefer_apple_vision,
                  "apple_vision_skip_vlm": cfg.apple_vision_skip_vlm,
                  "table_vlm_read": cfg.table_vlm_read,
                  "granularity": cfg.granularity,
                  "overlay_font": cfg.overlay_font,
                  "fuse_min_sim": cfg.fuse_min_sim,
                  "fuse_det_conf_trust": cfg.fuse_det_conf_trust},
        "prompts": {"transcribe": prompts.TRANSCRIBE, "typhoon": prompts.TYPHOON_OCR,
                    "table": prompts.TABLE},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _snapshot(work: Path, i: int, name: str) -> Path:
    return work / f"doc.{i:02d}-{name}.json"


def process(
    pdf_path: Path,
    cfg: Config,
    pipeline: list[Stage] | None = None,
    resume: bool = True,
    rerun_from: str | None = None,
    force: bool = False,
    digest: str | None = None,
) -> Document:
    """Run a PDF through the pipeline, resuming from cached per-stage snapshots.

    The cache key is the content hash AND the recipe fingerprint, so a re-run after
    changing a prompt / model / route reprocesses (the stale recipe is ignored) rather
    than being a silent no-op. ``rerun_from="vlm_read"`` re-runs from that stage onward,
    reusing the cached earlier stages — e.g. tune the VLM prompt without redoing OCR
    (per-stage snapshots make this correct even though stages like fusion aren't
    idempotent on their own output). ``force=True`` reprocesses from scratch.
    """
    pipeline = pipeline or DEFAULT_PIPELINE
    names = [s.name for s in pipeline]
    if rerun_from is not None and rerun_from not in names:
        raise ValueError(f"rerun_from={rerun_from!r} is not a stage in {names}")

    # Use the caller's digest when given, so the artifact dir (out/<digest>) can't drift
    # from the digest the job status was recorded under if the source path is overwritten
    # between the caller's hash and here. Falls back to hashing for direct callers.
    digest = digest or sha256_of(pdf_path)
    work = storage.job_dir(cfg, digest)
    work.mkdir(parents=True, exist_ok=True)
    recipe = recipe_fingerprint(cfg, pipeline)

    def _fresh() -> Document:
        return Document(source_path=str(pdf_path), sha256=digest)

    # Resume after the latest snapshot whose recipe still matches (else start fresh).
    start, doc = 0, _fresh()
    if resume and not force:
        for i in range(len(pipeline) - 1, -1, -1):
            snap = _snapshot(work, i, names[i])
            if snap.exists():
                cached = Document.from_json(snap.read_text())
                if cached.recipe == recipe:
                    start, doc = i + 1, cached
                    break

    # Explicit rerun: rewind to just before the requested stage, loading its pre-state.
    if rerun_from is not None:
        rf = names.index(rerun_from)
        if rf < start:
            prev = _snapshot(work, rf - 1, names[rf - 1]) if rf > 0 else None
            if prev is not None and prev.exists():
                start, doc = rf, Document.from_json(prev.read_text())
            else:
                start, doc = 0, _fresh()

    doc.recipe = recipe
    for i in range(start, len(pipeline)):
        doc = pipeline[i].run(doc, cfg)
        doc.stage_completed = pipeline[i].name
        doc.recipe = recipe
        _snapshot(work, i, names[i]).write_text(doc.to_json())
    (work / "doc.json").write_text(doc.to_json())   # latest state, for consumers
    return doc
