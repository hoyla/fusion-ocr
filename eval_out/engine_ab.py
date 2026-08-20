"""Deterministic-engine A/B — rapidocr_eval_plan.md, executed 2026-08-20.

SCOPE CORRECTION discovered pre-run (the 7th "audit the defaults" catch): paddleocr has
been pinned at 3.7.0 since 2026-06-28 (constraints.txt), and 3.7.0's DEFAULT en models are
**PP-OCRv6_medium_det/rec** — confirmed live and by the desktop's ~/.paddlex cache. So
every July stream-A/B/E number labelled "PaddleOCR" was ALREADY PP-OCRv6_medium; the
plan's "null hypothesis: add a v6 row" was satisfied by accident. The informative arms are
therefore:

  paddle_v6m  — the current default (PP-OCRv6_medium via paddleocr, no override)
  paddle_v5s  — PP-OCRv5_server_det/rec via the new det_model/rec_model config knob
                (recovers the generation delta the docs believed was the baseline)
  rapid       — RapidOCR 3.9.x / onnxruntime CPU, whose bundled default is
                **PP-OCRv6_rec_small.onnx** (the ONNX port is on v6-small — so this arm
                conflates engine AND tier; if it is faster at equal quality that is still
                an adoption case per the plan's criteria, but say which tier won)

Pre-flight geometry check (plan requirement, PASSED 2026-08-20): on a FUNSD page rapid vs
paddle boxes match 25/25 lines at IoU>=0.5, same coords within ~1px, same text. The plan's
tiny/small Paddle tiers are NOT testable at the 3.7.0 pin (registry carries medium only) —
recorded as an env-pinned limitation, not skipped silently. RapidOCR per-script recogniser
gap (verification #1) documented in engines/rapid.py — non-Latin stays on PaddleOCR.

Method: seeded n=30 per dataset (seed 1, all splits pooled — the plan's own numbers),
deterministic_pipeline (no VLM), SROIE scored caseless (the certified convention). Speed =
`t_ocr_det` from Document.stage_seconds on identical pages. Durable CSV-append.

Run:  .venv/bin/python eval_out/engine_ab.py            # sweep
      .venv/bin/python eval_out/engine_ab.py --report   # per-arm tables + decision inputs
"""
import csv
import dataclasses
import random
import sys
import time
import traceback
from pathlib import Path

from fusion_ocr import config as cm, ingest
from fusion_ocr.eval.datasets import _CASELESS_REF, iter_pairs
from fusion_ocr.eval.harness import recovered_text
from fusion_ocr.eval.metrics import normalize, score
from fusion_ocr.pipeline import deterministic_pipeline, process

RES = Path("eval_out/engine_ab"); RES.mkdir(parents=True, exist_ok=True)
CSV_OUT = RES / "results.csv"
N_PER_DS, SEED = 30, 1
COLS = ["arm", "dataset", "id", "ref_chars", "ref_words", "word_overlap", "hyp_words",
        "word_recall", "word_precision", "cer", "insertion_rate", "t_ocr_det", "secs"]

ARMS = [
    ("paddle_v6m", {}),                                # current default = PP-OCRv6_medium
    ("paddle_v5s", {"det_model": "PP-OCRv5_server_det",
                    "rec_model": "PP-OCRv5_server_rec"}),
    ("rapid", {"prefer_rapidocr": True}),              # ONNX; bundled rec = PP-OCRv6_small
    # tier-confound resolver (2026-08-20 follow-up): same engine, REC model = v6 MEDIUM
    # (det stays small — word counts showed detection wasn't the gap). If this closes
    # recall to within the plan's bar, the x15+ speedup becomes an adoption case.
    ("rapid_medrec", {"prefer_rapidocr": True}),
]


def sampled_items(full: bool = False):
    out = []
    for ds in ("funsd", "sroie"):
        pool = [(ds, img, ref) for sp in ("train", "test", "val")
                for img, ref in iter_pairs(ds, split=sp)]
        if full:
            out += pool          # full-set confirmation mode (pre-adoption evidence)
        else:
            random.seed(SEED)
            out += random.sample(pool, N_PER_DS)
    return out


def run_sweep(full_arm: str | None = None):
    """Default: the seeded n=30 sample across all arms. `--full <arm>`: EVERY gold item
    for that one arm (resumable; the n=30 rows are a subset so they're reused)."""
    items = sampled_items(full=full_arm is not None)
    done = set()
    if CSV_OUT.exists():
        done = {(r["arm"], r["dataset"], r["id"]) for r in csv.DictReader(CSV_OUT.open())}
    newfile = not CSV_OUT.exists()
    fh = CSV_OUT.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
    if newfile:
        w.writeheader(); fh.flush()

    base = cm.load()
    for arm, overrides in ARMS:
        if full_arm is not None and arm != full_arm:
            continue
        from fusion_ocr.engines import rapid
        rapid.set_rec_tier("medium" if arm == "rapid_medrec" else None)
        cfg = dataclasses.replace(base, out_dir=RES / "out" / arm, **overrides)
        todo = [it for it in items if (arm, it[0], it[1].stem) not in done]
        print(f"== {arm}: {len(todo)}/{len(items)} to go", flush=True)
        for n, (ds, img, ref) in enumerate(todo):
            t0 = time.time()
            try:
                if not normalize(ref):
                    continue
                pdf, _ = ingest.to_pdf(img, RES / "derived")
                doc = process(pdf, cfg, pipeline=deterministic_pipeline(),
                              digest=f"{arm}_{ds}_{img.stem}")
                hyp = "\n".join(recovered_text(p) for p in doc.pages)
                s = score(ref, hyp, caseless=ds in _CASELESS_REF)
                w.writerow(dict(arm=arm, dataset=ds, id=img.stem,
                                ref_chars=s["ref_chars"], ref_words=s["ref_words"],
                                word_overlap=s["word_overlap"], hyp_words=s["hyp_words"],
                                word_recall=round(s["word_recall"], 4),
                                word_precision=round(s["word_precision"], 4),
                                cer=round(s["cer"], 4),
                                insertion_rate=round(s["insertion_rate"], 4),
                                t_ocr_det=doc.stage_seconds.get("ocr_det", ""),
                                secs=round(time.time() - t0, 1)))
                fh.flush()
            except Exception:
                print(f"  ERR {arm}/{ds}/{img.stem}:", flush=True); traceback.print_exc()
            if n % 10 == 0:
                print(f"  [{n + 1}/{len(todo)}] {ds}/{img.stem} "
                      f"({round(time.time() - t0, 1)}s)", flush=True)
    fh.close(); print("SWEEP DONE", flush=True)


def report():
    rows = list(csv.DictReader(CSV_OUT.open()))
    print("=== engine A/B (micro-avg; speed = mean t_ocr_det on identical pages) ===")
    print(f"{'arm':>11} {'ds':>6} {'n':>3} {'recall':>8} {'prec':>8} {'cer':>8} {'t_det':>7}")
    agg = {}
    for arm, _ in ARMS:
        for ds in ("funsd", "sroie"):
            sub = [r for r in rows if r["arm"] == arm and r["dataset"] == ds
                   and r["word_recall"]]
            if not sub:
                continue
            wn = sum(int(r["ref_words"]) for r in sub) or 1
            hw = sum(int(r["hyp_words"]) for r in sub) or 1
            ov = sum(int(r["word_overlap"]) for r in sub)
            cn = sum(int(r["ref_chars"]) for r in sub) or 1
            ts = [float(r["t_ocr_det"]) for r in sub if r["t_ocr_det"]]
            a = {"n": len(sub), "recall": ov / wn, "prec": ov / hw,
                 "cer": sum(float(r["cer"]) * int(r["ref_chars"]) for r in sub) / cn,
                 "t": sum(ts) / len(ts) if ts else float("nan")}
            agg[(arm, ds)] = a
            print(f"{arm:>11} {ds:>6} {a['n']:>3} {a['recall']:>8.4f} {a['prec']:>8.4f} "
                  f"{a['cer']:>8.4f} {a['t']:>6.2f}s")
    if ("paddle_v6m", "funsd") in agg and ("rapid", "funsd") in agg:
        print("\nDecision inputs (plan criteria: adopt iff faster AND recall within ~0.01"
              " + geometry equivalent [pre-verified]):")
        for rarm in ("rapid", "rapid_medrec"):
            for ds in ("funsd", "sroie"):
                a, b = agg.get(("paddle_v6m", ds)), agg.get((rarm, ds))
                if a and b:
                    print(f"  {rarm}/{ds}: Δrecall {b['recall']-a['recall']:+.4f}; "
                          f"speedup ×{a['t']/b['t']:.1f} ({a['t']:.2f}s → {b['t']:.2f}s)")
    print("\n(adoption is per-component and lands on the PR carrying this table + the "
          "manifest — Luke decides; layout/table stay PaddleOCR regardless, per the plan)")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif "--full" in sys.argv:
        run_sweep(full_arm=sys.argv[sys.argv.index("--full") + 1])
    else:
        run_sweep()
