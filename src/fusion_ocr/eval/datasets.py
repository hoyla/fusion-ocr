"""Loaders for the 3rd-party OCR benchmark (samples/file_tests_3rdparty_01) — turn an
image + its annotation into (image_path, reference_text) so the eval can score the pipeline
against external ground truth, via the image-ingest adapter.

Only the GOLD, in-domain document sources are wired:
  - SROIE  (invoice/, printed receipts) — annotation `ocr_boxes: [{points, text}]`
  - FUNSD  (form/,    scanned forms)     — annotation `form: [{text, box, words, ...}]`
The reference is the concatenation of the annotated line texts in file order — a recognition
+ reading-order ground truth. Entity / key-value labels are deliberately ignored: that's
downstream analysis, not this tool's job (principle 1). IAM (document/) is NOT wired — its
bundled annotations are an OCR engine's *output* (per-line confidence), not human
transcriptions, so scoring our OCR against them would be circular. Total-Text (real_life/)
is scene text, out of the document domain.

Note: FUNSD annotation order isn't strict visual reading order, so on forms trust the
order-INSENSITIVE word recall / precision over CER/WER (the usual caveat).
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

from ..config import Config
from .harness import recovered_text
from .metrics import normalize, score

_ROOT = Path("samples/file_tests_3rdparty_01/archive")


def sroie_reference(ann_path) -> str:
    d = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    return "\n".join(b.get("text", "") for b in d.get("ocr_boxes", []))


def funsd_reference(ann_path) -> str:
    d = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    return "\n".join(i.get("text", "") for i in d.get("form", []) if i.get("text"))


# source -> (category subdir, reference extractor)
_SOURCES = {
    "sroie": ("invoice", sroie_reference),
    "funsd": ("form", funsd_reference),
}


def iter_pairs(source: str, split: str = "test", root=_ROOT, limit=None):
    """(image_path, reference_text) pairs for a dataset source/split, paired by file stem."""
    if source not in _SOURCES:
        raise ValueError(f"unknown source {source!r}; known: {sorted(_SOURCES)}")
    subdir, ref_fn = _SOURCES[source]
    base = Path(root) / subdir / split
    images = {p.stem: p for p in sorted((base / "images").glob("*")) if p.is_file()}
    anns = {p.stem: p for p in (base / "annotations").glob("*.json")}
    pairs = []
    for stem in sorted(images):
        if stem in anns:
            pairs.append((images[stem], ref_fn(anns[stem])))
            if limit and len(pairs) >= limit:
                break
    return pairs


def evaluate_dataset(source: str, cfg: Config, split: str = "test", limit: int = 20,
                     no_vlm: bool = False, root=_ROOT) -> list[dict]:
    """Score the pipeline on a sample of a benchmark source: ingest each image to a PDF,
    process it, and score the recovered text against the annotation. `no_vlm=True` measures
    the deterministic engine alone."""
    from .. import ingest
    from ..pipeline import deterministic_pipeline, process

    pairs = iter_pairs(source, split=split, root=root, limit=limit)
    tmp_root = Path(tempfile.mkdtemp(prefix=f"fusion_ds_{source}_"))
    eval_cfg = dataclasses.replace(cfg, out_dir=tmp_root / "out")
    pipeline = deterministic_pipeline() if no_vlm else None

    results = []
    for i, (img, ref) in enumerate(pairs):
        if not normalize(ref):
            continue   # no usable ground truth for this item
        pdf, _ = ingest.to_pdf(img, tmp_root / "derived")
        doc = process(pdf, eval_cfg, pipeline=pipeline, digest=f"{source}_{i:04d}")
        hyp = "\n".join(recovered_text(p) for p in doc.pages)
        results.append({"id": img.stem, "source": source, **score(ref, hyp)})
    return results
