"""Loaders for the 3rd-party OCR benchmark (samples/file_tests_3rdparty_01) — turn an
image + its annotation into (image_path, reference_text) so the eval can score the pipeline
against external ground truth, via the image-ingest adapter.

Only the GOLD, in-domain document sources are wired:
  - SROIE  (invoice/, printed receipts) — annotation `ocr_boxes: [{points, text}]`
  - FUNSD  (form/,    scanned forms)     — annotation `form: [{text, box, words, ...}]`
The reference is the concatenation of the annotated line texts. Entity / key-value labels are
deliberately ignored: that's downstream analysis, not this tool's job (principle 1). IAM
(document/) is NOT wired — its bundled annotations are an OCR engine's *output* (per-line
confidence), not human transcriptions, so scoring our OCR against them would be circular.
Total-Text (real_life/) is scene text, out of the document domain.

READING ORDER. SROIE receipts are single-column, so annotation order is already reading order.
FUNSD forms are 2-D (label/value pairs, multiple regions) and their annotations are stored in
*annotation* order, NOT reading order — a naive concatenation makes CER/WER uninterpretable
(can't tell mis-ordering from misrecognition). FUNSD carries a per-line `box`, though, so
`funsd_reference` reconstructs reading order GEOMETRICALLY (`_reading_order`): cluster lines
into y-bands ~one line-height tall, then left-to-right within a band. This is a *constructed*
oracle — a principled proxy for human reading order, which FUNSD doesn't annotate, not a hand
transcription; it's defensible because top-to-bottom / left-to-right within a row is the
reading convention forms follow. With it CER folds in reading order honestly and the
recall-vs-CER split is interpretable (high recall + high CER = recognised but mis-ordered).
The pipeline output it scores against is independent of this sort: the VLM reads from document
understanding, and the deterministic path orders by PP-DocLayoutV2's learned head — so a low
CER against this reference is genuine agreement, not the same sort compared to itself.
(`funsd_reference_annotation_order` keeps the raw order for comparison / regression context.)

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


def funsd_reference_annotation_order(ann_path) -> str:
    """Annotated line texts in the JSON's own (annotation) order — NOT reading order. Kept
    for comparison; `funsd_reference` (reading order) is what the eval scores against."""
    d = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    return "\n".join(i.get("text", "") for i in d.get("form", []) if i.get("text"))


def _reading_order(lines: list[dict]) -> list[dict]:
    """Sort boxed lines into human reading order: cluster into y-bands ~one line-height tall
    (so a row of side-by-side fields stays together), then left-to-right within each band.
    Each line is a dict with a `box` = [x0, y0, x1, y1]. A geometric proxy for reading order
    (see module docstring), used where the source doesn't annotate it (FUNSD). Lines with no
    box can't be placed geometrically; they're kept (never dropped) in their original order,
    after the boxed lines — real FUNSD always has boxes, so this only guards degenerate input."""
    boxed = [ln for ln in lines if ln.get("box")]
    unboxed = [ln for ln in lines if not ln.get("box")]
    if not boxed:
        return unboxed
    heights = sorted(ln["box"][3] - ln["box"][1] for ln in boxed)
    band = max(heights[len(heights) // 2], 1)              # median line height
    ycenter = lambda ln: (ln["box"][1] + ln["box"][3]) / 2  # noqa: E731
    boxed.sort(key=ycenter)
    bands: list[list[dict]] = []
    for ln in boxed:
        if bands and ycenter(ln) - ycenter(bands[-1][0]) <= band * 0.6:
            bands[-1].append(ln)
        else:
            bands.append([ln])
    ordered: list[dict] = []
    for b in bands:
        b.sort(key=lambda ln: ln["box"][0])
        ordered.extend(b)
    return ordered + unboxed


def funsd_reference(ann_path) -> str:
    """FUNSD reference in reconstructed reading order (see module docstring)."""
    d = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    lines = [i for i in d.get("form", []) if i.get("text")]
    return "\n".join(i["text"] for i in _reading_order(lines))


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
