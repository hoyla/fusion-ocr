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
import re
from dataclasses import asdict
from pathlib import Path

from ..compose import _contains_centre, reading_key
from ..config import Config
from ..models import Document, Page
from ..overlay.pymupdf_overlay import build_overlay


class Render:
    name = "render"

    def run(self, doc: Document, cfg: Config) -> Document:
        work = Path(cfg.out_dir) / doc.sha256

        index = {
            "source_path": doc.source_path,
            "sha256": doc.sha256,
            "languages": doc.languages,
            "pages": [
                {"index": p.index, "script": p.script, "needs_ocr": p.needs_ocr,
                 "read_model": p.read_model, "rotation": p.rotation,
                 "regions": [
                     {"kind": r.kind, "reading_order": r.reading_order,
                      "source": r.source, "bbox": list(r.box.bbox),
                      **({"table_html": r.table_html,
                          "cells": [list(c.bbox) for c in r.cells],
                          "cell_confidence": _conf_counts(r.table_html)}
                         if r.kind == "table" else {})}
                     for r in p.regions
                 ]}
                for p in doc.pages
            ],
            "segments": [
                {"page": s.page, **asdict(s)}
                for p in doc.pages
                for s in p.segments
            ],
        }
        index_path = work / "segment_index.json"
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
        doc.artifacts["segment_index"] = str(index_path)

        # Markdown reading view (segments already in reading order). Table regions are
        # emitted as their filled HTML table at their position; everything else as text.
        parts = [t for t in (_page_markdown(p) for p in doc.pages) if t]
        md_path = work / "document.md"
        md_path.write_text("\n\n".join(parts) if parts else "_(no text extracted yet)_\n")
        doc.artifacts["markdown"] = str(md_path)

        overlay_path = work / "overlay.pdf"
        if build_overlay(doc, overlay_path, granularity=cfg.granularity,
                         font_path=cfg.overlay_font or None):
            doc.artifacts["overlay_pdf"] = str(overlay_path)

        return doc


def _extract_table(table_html: str) -> str:
    i, j = table_html.find("<table"), table_html.rfind("</table>")
    return table_html[i:j + 8] if i >= 0 and j >= 0 else table_html


def _conf_counts(table_html: str) -> dict:
    """Tally cell confidence so a consumer can gate on it (e.g. trust only `clean`)."""
    counts: dict[str, int] = {}
    for level in re.findall(r'data-confidence="(\w+)"', table_html):
        counts[level] = counts.get(level, 0) + 1
    return counts


def _page_markdown(page: Page) -> str:
    """Reading view for one page. If it has filled tables, emit each as its HTML table
    at its region position, suppress that table's loose line segments, and interleave
    the rest in reading order; otherwise fall back to the flat text view."""
    tables = [r for r in page.regions if r.kind == "table" and "<table" in r.table_html]
    # Only embed the deterministic grid when there's no VLM reading (OCR-only /
    # born-digital). When the VLM read the page its markdown already carries the table
    # (cleaner than cell-stuffing coarse segments), so prefer that.
    if not tables or page.vlm_reading.strip():
        return _page_markdown_flat(page)

    blocks: list[tuple] = []
    table_seg_ids: set[int] = set()
    for r in tables:
        for s in page.segments:
            if not s.superseded and _contains_centre(r.box, s.box):
                table_seg_ids.add(id(s))
        blocks.append(((r.reading_order, 0, 0.0), _extract_table(r.table_html)))
    for s in page.segments:
        if s.superseded or not s.best_text or id(s) in table_seg_ids:
            continue
        blocks.append((reading_key(s, page.regions), s.best_text))
    blocks.sort(key=lambda b: b[0])
    return "\n\n".join(text for _, text in blocks)


def _page_markdown_flat(page: Page) -> str:
    seg_text = [s.best_text for s in page.segments if s.best_text and not s.superseded]
    has_textlayer = any(s.source == "textlayer" and not s.superseded
                        for s in page.segments)
    if has_textlayer and seg_text:          # mixed: verbatim text layer + OCR
        return "\n".join(seg_text)
    if page.vlm_reading.strip():             # pure-VLM page: clean continuous reading
        return page.vlm_reading.strip()
    return "\n".join(seg_text)
