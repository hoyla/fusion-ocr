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

PACKAGING NOTE: in samples/file_tests_3rdparty_01 the FUNSD `form/` train/test/val folders
were split INDEPENDENTLY for images vs annotations, so an image and its annotation usually sit
in *different* split folders (within one split folder the names don't line up). Pairing is by
stem ACROSS all splits (see _annotation_index), which resolves it — 200/203 FUNSD stems match.
SROIE's splits are aligned, so the cross-split lookup is a harmless no-op for it.
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


def _annotation_index(category: Path) -> dict:
    """stem -> annotation path, indexed across ALL splits. Some packagings split images and
    annotations INDEPENDENTLY into train/test/val, so an image's annotation can sit in a
    different split folder (FUNSD here); a global stem index pairs them correctly regardless.
    SROIE's splits happen to be aligned, so this is a no-op for it."""
    return {a.stem: a for a in category.glob("*/annotations/*.json")}


def iter_pairs(source: str, split: str = "test", root=_ROOT, limit=None):
    """(image_path, reference_text) pairs for a dataset source/split. Images come from the
    chosen split; each is paired with its annotation BY STEM, looked up across all splits (so
    an independently-split packaging still pairs correctly)."""
    if source not in _SOURCES:
        raise ValueError(f"unknown source {source!r}; known: {sorted(_SOURCES)}")
    subdir, ref_fn = _SOURCES[source]
    category = Path(root) / subdir
    anns = _annotation_index(category)
    pairs = []
    for img in sorted(p for p in (category / split / "images").glob("*") if p.is_file()):
        ann = anns.get(img.stem)
        if ann is not None:
            pairs.append((img, ref_fn(ann)))
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
