"""Stage 09 — Render outputs (the final pipeline stage).

Emits the three deliverables into out/<sha256>/ (see Docs/outputs.md):
  * overlay.pdf      — invisible text layer positioned by Segment boxes (PyMuPDF,
                       render mode 3). Searchable + highlightable.
  * document.md      — structured reading view, tables intact (one file, with
                       per-language sections; the ungated VLM reading).
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
                          "table_engine": r.table_engine,      # find_tables | table_structure
                          "table_read_by": r.table_read_by,    # VLM model, if read
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
        body = "\n\n".join(parts) if parts else "_(no text extracted yet)_\n"
        md_path = work / "document.md"
        md_path.write_text(_provenance_note(doc) + body)
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


def _provenance_note(doc: Document) -> str:
    """Honest header for the reading view. document.md is a READING aid; the overlay and
    segment index are the gated, provenance-bearing artifacts. The VLM caveat appears
    only when a vision-language model actually read a page — born-digital / Apple-Vision
    output is exact text (or ink-gated OCR), not a model transcription."""
    vlm_used = any(p.read_model and p.read_model != "apple_vision" for p in doc.pages)
    if not vlm_used:
        return ""
    return (
        "> _Reading view. Pages read by a vision-language model are a transcription and "
        "may contain model inferences on degraded or handwritten text. The searchable "
        "`overlay.pdf` and `segment_index.json` are the ink-gated, provenance-bearing "
        "record — every segment backs to a detected box and its source._\n\n")


def _page_markdown(page: Page) -> str:
    """Reading view for one page. If it has filled tables, emit each as its HTML table
    at its region position, suppress that table's loose line segments, and interleave
    the rest in reading order; otherwise fall back to the flat text view."""
    tables = [r for r in page.regions
              if r.kind == "table" and (r.table_vlm or "<table" in r.table_html)]
    has_vlm_table = any(r.table_vlm for r in tables)
    # Take the block path (each table placed at its region position, loose lines
    # interleaved) when a focused table read exists, OR when there's no page reading to
    # fall back on. If the page got a full reading and we have no focused table read,
    # that reading already carries the table — keep it whole (better flow than
    # reassembling from segments).
    if not tables or (page.vlm_reading.strip() and not has_vlm_table):
        return _page_markdown_flat(page)

    blocks: list[tuple] = []
    table_seg_ids: set[int] = set()
    for r in tables:
        for s in page.segments:
            if not s.superseded and _contains_centre(r.box, s.box):
                table_seg_ids.add(id(s))
        # focused VLM table read (clean content) preferred; else the deterministic grid
        blocks.append(((r.reading_order, 0, 0.0), r.table_vlm or _extract_table(r.table_html)))
    for s in page.segments:
        if s.superseded or not s.best_text or id(s) in table_seg_ids:
            continue
        blocks.append((reading_key(
            s, page.regions, page.rotation, page.width, page.height), s.best_text))
    blocks.sort(key=lambda b: b[0])
    return "\n\n".join(text for _, text in blocks)


def _page_markdown_flat(page: Page) -> str:
    # Reading VIEW. When a VLM read the page, its clean continuous reading IS the reading aid —
    # it covers the whole page (including any digital text it saw), and the EXACT text layer is
    # preserved untouched in the gated artifacts (segment_index / overlay), so nothing is
    # discarded. Reassembling from segments here would inherit any per-line fusion garble — and
    # a stray page-number text fragment must not flip a scan onto the "mixed" segment path.
    if page.vlm_reading.strip():
        return page.vlm_reading.strip()
    # No VLM reading: born-digital (exact text layer) or a non-VLM OCR tier — the composed
    # segments (verbatim text layer + OCR of the image areas) are the reading.
    seg_text = [s.best_text for s in page.segments if s.best_text and not s.superseded]
    return "\n".join(seg_text)
