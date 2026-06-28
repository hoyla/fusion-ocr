"""Run the OCR eval. Two modes:

Born-digital (text layer = ground truth, automatic):
  python -m fusion_ocr.eval report.pdf --pages 10,11,12 --apple-vision

Hand-labelled (human transcripts = ground truth, for scans / handwriting):
  python -m fusion_ocr.eval --labels eval_labels/labelset.json

--apple-vision runs server-free (the deterministic engine); omit it to use the configured
VLM. For the hand-labelled set — degraded scans and handwriting — you usually want the VLM,
so start the reader (e.g. the MLX server) and omit --apple-vision. CER/WER are
order-sensitive; word recall/precision are order-insensitive recognition numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import config as config_mod
from .metrics import aggregate


def _print_scorecard(results: list[dict], *, label_col: str) -> None:
    hdr = f"{label_col:>24} {'CER':>7} {'WER':>7} {'recall':>7} {'prec':>6} {'refchars':>9}"
    print(hdr)
    print("  (CER/WER are order-sensitive; recall/prec are order-insensitive recognition)")
    scored = [r for r in results if r.get("status", "scored") == "scored"]
    for r in results:
        tag = r.get("tag", "")
        if r.get("status") == "unlabelled":
            print(f"{tag:>24} {'—':>7} {'—':>7} {'—':>7} {'—':>6} {'(no transcript yet)':>9}")
            continue
        print(f"{tag:>24} {r['cer']:7.3f} {r['wer']:7.3f} "
              f"{r['word_recall']:7.3f} {r['word_precision']:6.3f} {r['ref_chars']:9}")
    if scored:
        agg = aggregate(scored)
        print(f"{'AGGREGATE':>24} {agg['cer']:7.3f} {agg['wer']:7.3f} "
              f"{agg['word_recall']:7.3f} {agg['word_precision']:6.3f} {agg['ref_chars']:9}"
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
    ap.add_argument("--pages", default=None,
                    help="comma-separated page indices (born-digital mode; default: all)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--apple-vision", action="store_true",
                    help="force the on-device engine (no VLM server needed)")
    args = ap.parse_args()

    if not args.labels and not args.pdfs:
        ap.error("give born-digital PDFs, or --labels for the hand-labelled set")

    cfg = config_mod.load(args.config)
    if args.apple_vision:
        cfg.prefer_apple_vision = True

    if args.labels:
        from .labels import evaluate_labelset
        results = evaluate_labelset(args.labels, cfg)
        if not results:
            print("no labels in the manifest")
            return
        for r in results:
            r["tag"] = r["id"][:24]
        _print_scorecard(results, label_col="label")
        return

    from .harness import evaluate
    pages = [int(x) for x in args.pages.split(",")] if args.pages else None
    results = evaluate([Path(p) for p in args.pdfs], cfg, pages=pages, dpi=args.dpi)
    if not results:
        print("no scorable pages (need a born-digital text layer with enough text)")
        return
    for r in results:
        r["tag"] = f"{Path(r['pdf']).stem[:16]}:{r['page']}"
    _print_scorecard(results, label_col="page")


if __name__ == "__main__":
    main()
