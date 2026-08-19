"""Counterfactual cost of the PaddleOCR skip-VLM tier (`paddle_skip_vlm`) — zero VLM compute.

The knob ships DISABLED (0.0). This runner prices enabling it, on the 349 archived stream-A
pre-fusion snapshots (`eval_out/stream_a_vlm/out/<id>/doc.06-table_read.json`): for each
candidate threshold T, any page whose mean PaddleOCR confidence >= T has its `vlm_reading`
BLANKED before fusion (exactly what the skip does — det_text becomes the reading), everything
else fuses as archived. Both arms are scored against gold: gated word recall/precision + band
placement (the campaign's product metrics), plus the fraction of pages skipped (= VLM calls
saved — the whole point).

PRE-REGISTERED decision rule (before any number is produced): recommend enabling at the
HIGHEST T where, on BOTH corpora, micro-avg Δword_recall and Δband_placement vs the no-skip
baseline are each > -0.005 (stream E's benign bar; the noise floor is zero) AND pages-skipped
>= 20% (below that the saving isn't worth a new behaviour). Otherwise recommend it stays 0.0.
The report also lists the WORST-hit items per T — the skip's cost concentrates exactly where
det_text is weak, and a benign micro-average must not hide a catastrophic tail. Known scope
limit, stated up front: FUNSD/SROIE are scanned PRINT; the handwriting interaction (is Paddle
ever confidently wrong on cursive?) is not measurable here — the report prints the skip rate
this T would produce on any IAM rows found in `stream_b_iam/results.csv` as a canary, and the
enable PR must carry that caveat. Enabling lands as its own PR — Luke decides.

Candidates: T in {0.90, 0.95, 0.98}. Needs the CORSAIR doc.06 copy + the samples archive
(refs/annotations/images), like stream E1. Durable CSV-append.

Run:  .venv/bin/python eval_out/paddle_skip_cost.py            # sweep
      .venv/bin/python eval_out/paddle_skip_cost.py --report   # tables + recommendation
"""
import csv
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

from fusion_ocr import config as cm
from fusion_ocr.compose import reading_key
from fusion_ocr.eval.datasets import _CASELESS_REF, _ROOT, _annotation_index, iter_pairs
from fusion_ocr.eval.metrics import normalize, score
from fusion_ocr.eval.placement import gt_lines, placement_counts
from fusion_ocr.models import Document
from fusion_ocr.stages.fusion import Fusion

RES = Path("eval_out/paddle_skip_cost"); RES.mkdir(parents=True, exist_ok=True)
CSV_OUT = RES / "results.csv"
ARCHIVE = Path("eval_out/stream_a_vlm/out")
IAM_CSV = Path("eval_out/stream_b_iam/results.csv")

THRESHOLDS = [0.90, 0.95, 0.98]          # 0 = the no-skip baseline arm
BENIGN_BAR, MIN_SAVING = 0.005, 0.20     # pre-registered (see module docstring)

COLS = ["threshold", "dataset", "id", "skipped", "mean_conf",
        "ref_words", "word_overlap", "hyp_words",
        "word_recall", "word_precision", "placed_band", "plain", "gt_words", "secs"]


def gated_text(doc) -> str:
    parts = []
    for page in doc.pages:
        segs = [s for s in page.segments if s.best_text and not s.superseded]
        segs.sort(key=lambda s: reading_key(
            s, page.regions, page.rotation, page.width, page.height))
        parts.append("\n".join(s.best_text for s in segs))
    return "\n".join(parts)


def build_reference_maps():
    ann = {"funsd": _annotation_index(_ROOT / "form"),
           "sroie": _annotation_index(_ROOT / "invoice")}
    out = {}
    for ds in ("funsd", "sroie"):
        for sp in ("train", "test", "val"):
            for img, ref in iter_pairs(ds, split=sp):
                out[f"{ds}_{img.stem}"] = (ds, ref, ds in _CASELESS_REF,
                                           ann[ds].get(img.stem), img)
    return out


def mean_paddle_conf(page) -> float | None:
    confs = [s.det_conf for s in page.segments
             if s.source == "paddle" and s.det_conf is not None]
    return (sum(confs) / len(confs)) if confs else None


def run_sweep():
    refs = build_reference_maps()
    items = sorted(d.name for d in ARCHIVE.iterdir()
                   if d.is_dir() and (d / "doc.06-table_read.json").exists()
                   and d.name in refs)
    if not items:
        raise SystemExit(f"no snapshots under {ARCHIVE} — copy the CORSAIR archive first")
    print(f"{len(items)} archived items; thresholds {[0.0] + THRESHOLDS}", flush=True)
    done = set()
    if CSV_OUT.exists():
        done = {(r["threshold"], r["dataset"], r["id"])
                for r in csv.DictReader(CSV_OUT.open())}
    newfile = not CSV_OUT.exists()
    fh = CSV_OUT.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
    if newfile:
        w.writeheader(); fh.flush()

    cfg = dataclasses.replace(cm.load(), fuse_min_sim=0.34, fuse_det_conf_trust=0.80)
    for n, iid in enumerate(items):
        ds, ref, caseless, ann_path, img_path = refs[iid]
        if not normalize(ref):
            continue
        snap = ARCHIVE / iid / "doc.06-table_read.json"
        for t in [0.0] + THRESHOLDS:
            key = (f"{t}", ds, iid.split("_", 1)[1])
            if key in done:
                continue
            t0 = time.time()
            try:
                doc = Document.from_json(snap.read_text())
                skipped, conf = 0, None
                for page in doc.pages:
                    conf = mean_paddle_conf(page)
                    if t > 0 and conf is not None and conf >= t:
                        page.vlm_reading = ""       # the skip: det_text IS the reading
                        page.read_model = "paddle"
                        skipped = 1
                doc = Fusion().run(doc, cfg)
                s = score(ref, gated_text(doc), caseless=caseless)
                pb = {"placed": 0, "plain": 0, "total": 0}
                if ann_path and img_path and Path(img_path).exists():
                    lines = gt_lines(json.loads(Path(ann_path).read_text()), ds)
                    if lines:
                        W, H = Image.open(img_path).size
                        pb = placement_counts(doc.pages[0], lines, W, H,
                                              caseless=caseless, band=True)
                w.writerow(dict(threshold=t, dataset=ds, id=iid.split("_", 1)[1],
                                skipped=skipped,
                                mean_conf=round(conf, 4) if conf is not None else "",
                                ref_words=s["ref_words"], word_overlap=s["word_overlap"],
                                hyp_words=s["hyp_words"],
                                word_recall=round(s["word_recall"], 4),
                                word_precision=round(s["word_precision"], 4),
                                placed_band=pb["placed"], plain=pb["plain"],
                                gt_words=pb["total"], secs=round(time.time() - t0, 2)))
                fh.flush()
            except Exception:
                print(f"  ERR {iid} T={t}:", flush=True); traceback.print_exc()
        if n % 25 == 0:
            print(f"[{n + 1}/{len(items)}] {iid}", flush=True)
    fh.close(); print("SWEEP DONE", flush=True)


def micro(rows):
    wn = sum(int(r["ref_words"]) for r in rows) or 1
    hw = sum(int(r["hyp_words"]) for r in rows) or 1
    gt = sum(int(r["gt_words"]) for r in rows) or 1
    return {"n": len(rows),
            "recall": sum(int(r["word_overlap"]) for r in rows) / wn,
            "precision": sum(int(r["word_overlap"]) for r in rows) / hw,
            "band": sum(int(r["placed_band"]) for r in rows) / gt,
            "skip": sum(int(r["skipped"]) for r in rows) / (len(rows) or 1)}


def report():
    rows = list(csv.DictReader(CSV_OUT.open()))
    datasets = sorted({r["dataset"] for r in rows})
    base = {ds: micro([r for r in rows if r["dataset"] == ds and float(r["threshold"]) == 0])
            for ds in datasets}
    print("=== paddle_skip_vlm counterfactual cost (micro-avg vs no-skip baseline) ===")
    print(f"{'T':>5} {'ds':>6} {'n':>4} {'skip%':>6} {'Δrecall':>9} {'Δprec':>9} {'Δband':>9}")
    qualifying = []
    for t in THRESHOLDS:
        ok_all, sav_all = True, True
        for ds in datasets:
            m = micro([r for r in rows if r["dataset"] == ds and float(r["threshold"]) == t])
            if not m["n"]:
                continue
            dr, dp = m["recall"] - base[ds]["recall"], m["precision"] - base[ds]["precision"]
            db = m["band"] - base[ds]["band"]
            print(f"{t:>5} {ds:>6} {m['n']:>4} {m['skip'] * 100:>5.1f}% "
                  f"{dr:>+9.4f} {dp:>+9.4f} {db:>+9.4f}")
            ok_all &= (dr > -BENIGN_BAR and db > -BENIGN_BAR)
            sav_all &= (m["skip"] >= MIN_SAVING)
        if ok_all and sav_all:
            qualifying.append(t)
    for t in THRESHOLDS:                     # the tail a micro-average could hide
        tr = [r for r in rows if float(r["threshold"]) == t and r["skipped"] == "1"]
        br = {(r["dataset"], r["id"]): float(r["word_recall"])
              for r in rows if float(r["threshold"]) == 0}
        hit = sorted(((float(r["word_recall"]) - br[(r["dataset"], r["id"])], r)
                      for r in tr if (r["dataset"], r["id"]) in br), key=lambda x: x[0])[:5]
        if hit:
            print(f"\nworst skipped items at T={t} (Δrecall vs own baseline):")
            for d, r in hit:
                print(f"  {r['dataset']}/{r['id']}: {d:+.4f} (mean_conf {r['mean_conf']})")
    if IAM_CSV.exists():
        print("\n(IAM canary: stream_b results carry no det_conf column — the handwriting "
              "interaction stays an explicit caveat on any enable PR.)")
    print(f"\nPRE-REGISTERED RULE -> "
          + (f"recommend enabling at T={max(qualifying)}" if qualifying
             else "recommend STAYING DISABLED (no T clears both bars)")
          + f"  [bars: Δ > -{BENIGN_BAR} on recall+band per corpus; skip >= {MIN_SAVING:.0%}]"
          + "\n(the enable lands as its own PR carrying this table — Luke decides)")


if __name__ == "__main__":
    report() if "--report" in sys.argv else run_sweep()
