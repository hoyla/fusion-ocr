"""Run the OCR eval. Two modes:

Born-digital (text layer = ground truth, automatic):
  python -m fusion_ocr.eval report.pdf --pages 10,11,12 --apple-vision

Hand-labelled (human transcripts = ground truth, for scans / handwriting):
  python -m fusion_ocr.eval --labels eval_labels/labelset.json

Engine selection (these compose, so you can A/B the same labels three ways):
  - default                  PaddleOCR geometry + the configured VLM reader (start the reader)
  - --no-vlm                 deterministic engine only, no reader — PaddleOCR recognition
  - --no-vlm --apple-vision  deterministic engine only, on-device — Apple Vision recognition

--no-vlm drops the VLM stages from the pipeline, so it measures the deterministic recogniser
in isolation without stopping the reader server. For the hand-labelled set — degraded scans
and handwriting — you usually want the full default (the VLM is the point). CER/WER are
order-sensitive; word recall/precision are order-insensitive recognition numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import config as config_mod
from .metrics import aggregate


_VIA = {"overlay": "ovl", "text_layer": "txt", "none": "MISS"}


def _searchable_cell(r: dict) -> str:
    """Right-hand block for one row: the searchable text (what find() hits in the output
    PDF) scored against the same reference, with how it's searchable — ovl=OCR overlay,
    txt=source text layer, MISS=nothing findable."""
    s = r.get("searchable")
    if not s:
        return ""
    via = _VIA.get(r.get("searchable_via"), "?")
    return f" | {s['cer']:6.3f} {s['word_recall']:6.3f} {via:>4}"


def _print_scorecard(results: list[dict], *, label_col: str) -> None:
    has_srch = any("searchable" in r for r in results)
    srch_hdr = f" | {'sCER':>6} {'sRcl':>6} {'via':>4}" if has_srch else ""
    hdr = f"{label_col:>24} {'CER':>7} {'WER':>7} {'recall':>7} {'prec':>6} {'refchars':>9}{srch_hdr}"
    print(hdr)
    print("  (CER/WER order-sensitive; recall/prec order-insensitive recognition" +
          ("; s* = searchable text find() hits in the output PDF — via ovl/txt/MISS)"
           if has_srch else ")"))
    scored = [r for r in results if r.get("status", "scored") == "scored"]
    for r in results:
        tag = r.get("tag", "")
        if r.get("status") == "unlabelled":
            print(f"{tag:>24} {'—':>7} {'—':>7} {'—':>7} {'—':>6} {'(no transcript yet)':>9}")
            continue
        srch = _searchable_cell(r) if has_srch else ""
        print(f"{tag:>24} {r['cer']:7.3f} {r['wer']:7.3f} "
              f"{r['word_recall']:7.3f} {r['word_precision']:6.3f} {r['ref_chars']:9}{srch}")
    if scored:
        agg = aggregate(scored)
        srch = ""
        if has_srch:
            s_scored = [r["searchable"] for r in scored if "searchable" in r]
            if s_scored:
                sa = aggregate(s_scored)
                srch = f" | {sa['cer']:6.3f} {sa['word_recall']:6.3f} {'':>4}"
        print(f"{'AGGREGATE':>24} {agg['cer']:7.3f} {agg['wer']:7.3f} "
              f"{agg['word_recall']:7.3f} {agg['word_precision']:6.3f} {agg['ref_chars']:9}{srch}"
              f"  ({agg['pages']} pages)")
    todo = [r for r in results if r.get("status") == "unlabelled"]
    if todo:
        print(f"\n{len(todo)} page(s) not yet labelled — fill in the transcript files:")
        for r in todo:
            print(f"  - {r['tag']}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="fusion-ocr-eval")
    ap.add_argument("pdfs", nargs="*",
                    help="born-digital PDFs (text layer = ground truth)")
    ap.add_argument("--labels", default=None,
                    help="hand-labelled manifest (e.g. eval_labels/labelset.json) — scores "
                         "against human transcripts instead of a born-digital text layer")
    ap.add_argument("--dataset", choices=["sroie", "funsd"], default=None,
                    help="score a sample of a 3rd-party benchmark (images via the ingest "
                         "adapter) against its annotations")
    ap.add_argument("--limit", type=int, default=20, help="dataset: items to sample")
    ap.add_argument("--split", default="test", help="dataset: train | val | test")
    ap.add_argument("--pages", default=None,
                    help="comma-separated page indices (born-digital mode; default: all)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--apple-vision", action="store_true",
                    help="force the on-device engine (no VLM server needed)")
    ap.add_argument("--no-vlm", action="store_true",
                    help="deterministic engine only — drop the VLM stages and score the "
                         "recogniser's own text (PaddleOCR, or Apple Vision with "
                         "--apple-vision). No reader server needed.")
    args = ap.parse_args()

    if not args.labels and not args.pdfs and not args.dataset:
        ap.error("give born-digital PDFs, --labels, or --dataset")

    cfg = config_mod.load(args.config)
    if args.apple_vision:
        cfg.prefer_apple_vision = True

    engine = ("Apple Vision" if args.apple_vision else "PaddleOCR") if args.no_vlm \
        else f"{'Apple Vision' if args.apple_vision else 'PaddleOCR'} + VLM ({cfg.vlm.model})"
    print(f"engine: {engine}")

    if args.labels:
        from .labels import evaluate_labelset
        results = evaluate_labelset(args.labels, cfg, no_vlm=args.no_vlm)
        if not results:
            print("no labels in the manifest")
            return
        for r in results:
            r["tag"] = r["id"][:24]
        _print_scorecard(results, label_col="label")
        return

    if args.dataset:
        from .datasets import evaluate_dataset
        results = evaluate_dataset(args.dataset, cfg, split=args.split,
                                   limit=args.limit, no_vlm=args.no_vlm)
        if not results:
            print(f"no scorable items in {args.dataset}/{args.split}")
            return
        for r in results:
            r["tag"] = r["id"][:24]
        _print_scorecard(results, label_col=args.dataset)
        return

    from .harness import evaluate
    pages = [int(x) for x in args.pages.split(",")] if args.pages else None
    results = evaluate([Path(p) for p in args.pdfs], cfg, pages=pages, dpi=args.dpi,
                       no_vlm=args.no_vlm)
    if not results:
        print("no scorable pages (need a born-digital text layer with enough text)")
        return
    for r in results:
        r["tag"] = f"{Path(r['pdf']).stem[:16]}:{r['page']}"
    _print_scorecard(results, label_col="page")


if __name__ == "__main__":
    main()
