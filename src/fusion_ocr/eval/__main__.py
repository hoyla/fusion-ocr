"""Run the OCR eval against born-digital text-layer ground truth.

  python -m fusion_ocr.eval report.pdf --pages 10,11,12 --apple-vision

Prints a per-page CER/WER/insertion scorecard + a micro-averaged aggregate. --apple-vision
runs server-free (measures the deterministic engine); omit it to use the configured VLM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import config as config_mod
from .harness import evaluate
from .metrics import aggregate


def main() -> None:
    ap = argparse.ArgumentParser(prog="fusion-ocr-eval")
    ap.add_argument("pdfs", nargs="+", help="born-digital PDFs (text layer = ground truth)")
    ap.add_argument("--pages", default=None,
                    help="comma-separated page indices (default: all with enough text)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--apple-vision", action="store_true",
                    help="force the on-device engine (no VLM server needed)")
    args = ap.parse_args()

    cfg = config_mod.load(args.config)
    if args.apple_vision:
        cfg.prefer_apple_vision = True
    pages = [int(x) for x in args.pages.split(",")] if args.pages else None

    results = evaluate([Path(p) for p in args.pdfs], cfg, pages=pages, dpi=args.dpi)
    if not results:
        print("no scorable pages (need a born-digital text layer with enough text)")
        return

    hdr = f"{'page':>24} {'CER':>7} {'WER':>7} {'recall':>7} {'prec':>6} {'refchars':>9}"
    print(hdr)
    print("  (CER/WER are order-sensitive; recall/prec are order-insensitive recognition)")
    for r in results:
        tag = f"{Path(r['pdf']).stem[:16]}:{r['page']}"
        print(f"{tag:>24} {r['cer']:7.3f} {r['wer']:7.3f} "
              f"{r['word_recall']:7.3f} {r['word_precision']:6.3f} {r['ref_chars']:9}")
    agg = aggregate(results)
    print(f"{'AGGREGATE':>24} {agg['cer']:7.3f} {agg['wer']:7.3f} "
          f"{agg['word_recall']:7.3f} {agg['word_precision']:6.3f} {agg['ref_chars']:9}"
          f"  ({agg['pages']} pages)")


if __name__ == "__main__":
    main()
