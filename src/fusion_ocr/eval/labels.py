"""Hand-labelled eval — the counterpart to the born-digital harness for pages that
carry no machine-readable truth (degraded scans, handwriting).

The born-digital path ([harness.py]) gets ground truth for free: a digital page already
holds its correct text, so we render it, OCR it, and score the two. Scans and handwriting
have no such layer — the only ground truth is a human reading the page. This module scores
the pipeline's output against those human transcripts.

A side benefit: because a person transcribes in TRUE visual reading order, these labels are
also a reading-order oracle — something the born-digital text layer (content-stream order)
cannot give us, so CER/WER mean here what they can't mean there.

Manifest (`eval_labels/labelset.json`):

    {"labels": [
        {"id": "mandelson-note-handwritten",
         "pdf": "samples/.../HA_Volume_II_part_I.pdf",   # relative to where you run the eval
         "pages": [183, 184],                             # 0-BASED; or "page": 183 for one page
         "transcript": "mandelson-note-handwritten.txt",  # relative to the manifest's folder
         "note": "free-text reminder of what this page is"}
    ]}

A document that spans pages (a 2-page letter, a multi-page form) is one label with a
`pages` list; the transcript covers the whole span and the recovered text is concatenated
across those pages in order before scoring.

The transcript file holds the correct reading of the page. An empty transcript means
"not labelled yet" and is reported as TODO rather than scored, so the scaffold runs (and
tells you what's left to do) before you've filled anything in.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

from ..config import Config
from .harness import recovered_text
from .metrics import normalize, score


@dataclasses.dataclass
class Label:
    id: str
    pdf: Path
    pages: list[int]   # 0-based page indices — a label can span pages (e.g. a 2-page letter)
    transcript: Path   # resolved absolute path to the .txt
    note: str = ""

    def reference(self) -> str:
        """The human transcript, or '' if the file is missing/empty (= not yet labelled)."""
        return self.transcript.read_text(encoding="utf-8") if self.transcript.exists() else ""


def load_labelset(manifest_path) -> list[Label]:
    """Parse a labelset manifest. `pdf` paths are kept relative (resolved against the cwd
    you run the eval from, like the born-digital CLI); `transcript` paths resolve against
    the manifest's own folder so the label files travel with it. A label is one page
    (`"page": 3`) or several (`"pages": [3, 4]`) when one document spans pages — the
    transcript then covers the whole span, in reading order."""
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    labels = []
    for e in data.get("labels", []):
        pages = e["pages"] if "pages" in e else [e["page"]]
        labels.append(Label(
            id=e["id"],
            pdf=Path(e["pdf"]),
            pages=[int(p) for p in pages],
            transcript=(base / e["transcript"]).resolve(),
            note=e.get("note", ""),
        ))
    return labels


def _pdf_text(pdf_path) -> str:
    """All searchable text embedded in a PDF, pages joined — i.e. what a reader's
    find/search would actually hit (used for both overlay.pdf and the source PDF). ''
    if the file is absent (so a missing overlay scores as empty rather than raising)."""
    import fitz
    p = Path(pdf_path)
    if not p.exists():
        return ""
    with fitz.open(str(p)) as d:
        return " ".join(pg.get_text() for pg in d)


def _extract_pages(src, page_indices: list[int], dst) -> None:
    """Copy the given pages into a fresh PDF, preserving content as-is (no flattening), so
    the pipeline sees exactly what production would — a real scan stays a scan, and any
    mixed digital/scan content composes normally."""
    import fitz
    with fitz.open(src) as d:
        out = fitz.open()
        for pi in page_indices:
            out.insert_pdf(d, from_page=pi, to_page=pi)
        out.save(str(dst))
        out.close()


def evaluate_labelset(manifest_path, cfg: Config, tmp_root=None, no_vlm: bool = False) -> list[dict]:
    """Score every labelled page in a manifest. Each result carries `status`: "scored"
    (with the metric fields from score()) or "unlabelled" (transcript still empty).
    ``no_vlm=True`` runs the deterministic engine only (no reader) — the recovered text is
    pure PaddleOCR / Apple Vision recognition.

    Each scored result also carries a nested ``searchable`` score and ``searchable_via``:
    the text a reader's find/search would actually hit in the OUTPUT PDF, measured against
    the same human transcript. That's ``overlay.pdf`` when one was built (it carries the
    source text layer *plus* the OCR overlay), otherwise the source PDF itself — whose text
    layer is still searchable (born-digital pages, and mixed pages whose exact text layer
    already covers the content, so no overlay is added and search hits aren't doubled).
    ``searchable_via`` records which: ``"overlay"``, ``"text_layer"``, or ``"none"`` (nothing
    searchable — a genuine miss). The reading view (``document.md``) and the searchable text
    diverge where fusion can't anchor a line to a box; the gap is text we recovered but a
    reader can't find."""
    from ..pipeline import deterministic_pipeline, process

    labels = load_labelset(manifest_path)
    tmp_root = Path(tmp_root or tempfile.mkdtemp(prefix="fusion_label_eval_"))
    eval_cfg = dataclasses.replace(cfg, out_dir=tmp_root / "out")
    pipeline = deterministic_pipeline() if no_vlm else None

    results: list[dict] = []
    for lab in labels:
        ref = lab.reference()
        base = {"id": lab.id, "pdf": str(lab.pdf), "pages": lab.pages}
        if not normalize(ref):
            results.append({**base, "status": "unlabelled"})
            continue
        page_pdf = tmp_root / f"{lab.id}.pdf"
        _extract_pages(lab.pdf, lab.pages, page_pdf)
        doc = process(page_pdf, eval_cfg, pipeline=pipeline)
        hyp = "\n".join(recovered_text(p) for p in doc.pages)   # concat across the span
        res = {**base, "status": "scored", **score(ref, hyp)}

        # Searchability: score the text find() would hit in the OUTPUT PDF, not just the
        # reading view. That's overlay.pdf when one was built (it's the source PDF + the OCR
        # overlay), otherwise the source PDF itself — whose text layer stays searchable for
        # born-digital pages and for mixed pages whose exact text layer already covers the
        # content (the OCR is superseded, so no overlay is added and search hits aren't
        # doubled). The gap to the reading score is text recovered but not findable.
        overlay_pdf = doc.artifacts.get("overlay_pdf")
        searchable = _pdf_text(overlay_pdf) if overlay_pdf else _pdf_text(page_pdf)
        res["searchable_via"] = ("overlay" if overlay_pdf else "text_layer") \
            if normalize(searchable) else "none"
        res["searchable"] = score(ref, searchable)
        results.append(res)
    return results
