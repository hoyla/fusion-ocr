"""Stage 7 — Render outputs.

Emits three artifacts into out/<sha256>/:
  * overlay.pdf      — invisible text layer positioned by Segment boxes (PyMuPDF,
                       render mode 3). Searchable + highlightable.
  * <lang>.md        — structured per-language reading (tables intact).
  * segment_index.json — the id <-> box <-> text map that powers "show me this line
                       in situ" and is the provenance record.

WALKING SKELETON: always writes segment_index.json + a markdown stub. The overlay
PDF is delegated to overlay.pymupdf_overlay, which no-ops cleanly if PyMuPDF or the
source segments are absent.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..config import Config
from ..models import Document
from ..overlay.pymupdf_overlay import build_overlay


class Render:
    name = "render"

    def run(self, doc: Document, cfg: Config) -> Document:
        work = Path(cfg.out_dir) / doc.sha256

        index = {
            "source_path": doc.source_path,
            "sha256": doc.sha256,
            "languages": doc.languages,
            "segments": [
                {"page": s.page, **asdict(s)}
                for p in doc.pages
                for s in p.segments
            ],
        }
        index_path = work / "segment_index.json"
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
        doc.artifacts["segment_index"] = str(index_path)

        # Per-language markdown (stub joins best_text in reading order).
        lines = [s.best_text for p in doc.pages for s in p.segments if s.best_text]
        md_path = work / "document.md"
        md_path.write_text("\n\n".join(lines) if lines else "_(no text extracted yet)_\n")
        doc.artifacts["markdown"] = str(md_path)

        overlay_path = work / "overlay.pdf"
        if build_overlay(doc, overlay_path, granularity=cfg.granularity):
            doc.artifacts["overlay_pdf"] = str(overlay_path)

        return doc
