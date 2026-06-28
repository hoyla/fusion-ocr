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
         "page": 183,                                     # 0-BASED page index
         "transcript": "mandelson-note-handwritten.txt",  # relative to the manifest's folder
         "note": "free-text reminder of what this page is"}
    ]}

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
    page: int          # 0-based page index
    transcript: Path   # resolved absolute path to the .txt
    note: str = ""

    def reference(self) -> str:
        """The human transcript, or '' if the file is missing/empty (= not yet labelled)."""
        return self.transcript.read_text(encoding="utf-8") if self.transcript.exists() else ""


def load_labelset(manifest_path) -> list[Label]:
    """Parse a labelset manifest. `pdf` paths are kept relative (resolved against the cwd
    you run the eval from, like the born-digital CLI); `transcript` paths resolve against
    the manifest's own folder so the label files travel with it."""
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    labels = []
    for e in data.get("labels", []):
        labels.append(Label(
            id=e["id"],
            pdf=Path(e["pdf"]),
            page=int(e["page"]),
            transcript=(base / e["transcript"]).resolve(),
            note=e.get("note", ""),
        ))
    return labels


def _extract_page(src, page_index: int, dst) -> None:
    """Copy one page into a fresh 1-page PDF, preserving its content as-is (no flattening),
    so the pipeline sees exactly what production would — a real scan stays a scan, and any
    mixed digital/scan content composes normally."""
    import fitz
    with fitz.open(src) as d:
        out = fitz.open()
        out.insert_pdf(d, from_page=page_index, to_page=page_index)
        out.save(str(dst))
        out.close()


def evaluate_labelset(manifest_path, cfg: Config, tmp_root=None) -> list[dict]:
    """Score every labelled page in a manifest. Each result carries `status`: "scored"
    (with the metric fields from score()) or "unlabelled" (transcript still empty)."""
    from ..pipeline import process

    labels = load_labelset(manifest_path)
    tmp_root = Path(tmp_root or tempfile.mkdtemp(prefix="fusion_label_eval_"))
    eval_cfg = dataclasses.replace(cfg, out_dir=tmp_root / "out")

    results: list[dict] = []
    for lab in labels:
        ref = lab.reference()
        base = {"id": lab.id, "pdf": str(lab.pdf), "page": lab.page}
        if not normalize(ref):
            results.append({**base, "status": "unlabelled"})
            continue
        page_pdf = tmp_root / f"{lab.id}.pdf"
        _extract_page(lab.pdf, lab.page, page_pdf)
        doc = process(page_pdf, eval_cfg)
        hyp = recovered_text(doc.pages[0]) if doc.pages else ""
        results.append({**base, "status": "scored", **score(ref, hyp)})
    return results
