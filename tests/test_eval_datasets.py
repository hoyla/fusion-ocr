"""3rd-party benchmark loaders: reference extraction (SROIE / FUNSD) and stem pairing.
These are pure JSON/path logic — no images, models, or the real (gitignored) dataset."""

from __future__ import annotations

import json

import pytest

from fusion_ocr.eval import datasets


def test_sroie_reference_joins_ocr_box_text_in_order(tmp_path):
    ann = tmp_path / "X1.json"
    ann.write_text(json.dumps({
        "file_id": "X1",
        "entities": {"total": "RM9.00"},          # ignored: KV is downstream scope
        "ocr_boxes": [{"points": [], "text": "ACME STORE"},
                      {"points": [], "text": "TOTAL RM9.00"}],
    }))
    assert datasets.sroie_reference(ann) == "ACME STORE\nTOTAL RM9.00"


def test_sroie_scored_caseless_funsd_is_not():
    # SROIE's GT is all-uppercase, so it must be scored case-insensitively; FUNSD keeps real
    # case and must not be. Guards the registry that drives metrics.score(caseless=...).
    assert "sroie" in datasets._CASELESS_REF
    assert "funsd" not in datasets._CASELESS_REF


def test_funsd_reference_joins_form_text_and_skips_empty(tmp_path):
    # No boxes -> can't order geometrically, falls back to annotation order (lossless).
    ann = tmp_path / "f1.json"
    ann.write_text(json.dumps({
        "form": [{"text": "Date:", "label": "question"},
                 {"text": "", "label": "other"},        # empty -> skipped
                 {"text": "12/2024", "label": "answer"}],
    }))
    assert datasets.funsd_reference(ann) == "Date:\n12/2024"


def test_funsd_reference_reconstructs_reading_order_from_boxes(tmp_path):
    # Boxes given in scrambled annotation order; reference must come out in reading order:
    # top-to-bottom, and within a row left-to-right (the side-by-side label/value pair).
    # box = [x0, y0, x1, y1].
    ann = tmp_path / "f2.json"
    ann.write_text(json.dumps({
        "form": [
            {"text": "Value", "box": [300, 100, 360, 116]},   # row 1, right
            {"text": "Footer", "box": [80, 400, 140, 416]},   # row 3
            {"text": "Label:", "box": [80, 102, 140, 118]},   # row 1, left (y within a band)
            {"text": "Middle", "box": [80, 250, 160, 266]},   # row 2
        ],
    }))
    assert datasets.funsd_reference(ann) == "Label:\nValue\nMiddle\nFooter"
    # Annotation-order helper preserves the raw (scrambled) order for comparison.
    assert datasets.funsd_reference_annotation_order(ann) == "Value\nFooter\nLabel:\nMiddle"


def test_iter_pairs_matches_images_to_annotations_by_stem(tmp_path):
    base = tmp_path / "invoice" / "test"
    (base / "images").mkdir(parents=True)
    (base / "annotations").mkdir(parents=True)
    for stem in ("a", "b", "c"):
        (base / "images" / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        (base / "annotations" / f"{stem}.json").write_text(
            json.dumps({"ocr_boxes": [{"text": stem.upper()}]}))
    (base / "images" / "orphan.jpg").write_bytes(b"\xff\xd8\xff")   # no annotation -> dropped

    pairs = datasets.iter_pairs("sroie", split="test", root=tmp_path, limit=2)
    assert [p[0].stem for p in pairs] == ["a", "b"]    # sorted, limit honoured, orphan dropped
    assert pairs[0][1] == "A"


def test_iter_pairs_matches_annotation_in_a_different_split(tmp_path):
    # FUNSD packaging: images and annotations were split independently, so an image in `test`
    # has its annotation in `train`. Pairing is by stem across all splits.
    cat = tmp_path / "form"
    (cat / "test" / "images").mkdir(parents=True)
    (cat / "train" / "annotations").mkdir(parents=True)
    (cat / "test" / "images" / "doc1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (cat / "train" / "annotations" / "doc1.json").write_text(
        json.dumps({"form": [{"text": "HELLO"}]}))

    pairs = datasets.iter_pairs("funsd", split="test", root=tmp_path)
    assert [p[0].stem for p in pairs] == ["doc1"]      # found despite the cross-split layout
    assert pairs[0][1] == "HELLO"


def test_iter_pairs_unknown_source_raises(tmp_path):
    with pytest.raises(ValueError):
        datasets.iter_pairs("iam", root=tmp_path)      # IAM intentionally not a gold source
