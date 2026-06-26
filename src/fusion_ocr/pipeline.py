"""Pipeline orchestration + the Stage contract.

Every stage is `Document in -> Document out`. The Document is serialised to
out/<sha256>/doc.json after each stage, so a crash or a deliberate re-run with a
refined prompt resumes from the last completed stage instead of redoing OCR.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

from .config import Config
from .models import Document
from .stages.fusion import Fusion
from .stages.language import Language
from .stages.layout import Layout
from .stages.ocr_det import OcrDet
from .stages.render import Render
from .stages.table import Table
from .stages.table_fill import TableFill
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


def process(
    pdf_path: Path,
    cfg: Config,
    pipeline: list[Stage] | None = None,
    resume: bool = True,
) -> Document:
    """Run a single PDF through the pipeline. Idempotent by content hash."""
    pipeline = pipeline or DEFAULT_PIPELINE
    digest = sha256_of(pdf_path)
    work = cfg.out_dir / digest
    work.mkdir(parents=True, exist_ok=True)
    doc_json = work / "doc.json"

    if resume and doc_json.exists():
        doc = Document.from_json(doc_json.read_text())
    else:
        doc = Document(source_path=str(pdf_path), sha256=digest)

    completed = doc.stage_completed
    seen_completed = completed is None
    for stage in pipeline:
        # Skip stages already done in a prior run (resume).
        if not seen_completed:
            if stage.name == completed:
                seen_completed = True
            continue
        doc = stage.run(doc, cfg)
        doc.stage_completed = stage.name
        doc_json.write_text(doc.to_json())

    return doc
