"""Counterfactual cost of the ENABLED Apple Vision skip-VLM tier (`apple_vision_skip_vlm`,
default 0.92) — the follow-up registered by the paddle_skip pricing (2026-08-19).

The Paddle skip was priced before enabling and stayed disabled: PaddleOCR is confidently
wrong on a real tail, so det_text-as-reading cost 4–8 recall points on clean print. The
Vision tier shipped ENABLED at 0.92 before any such pricing existed (routing.md: "benchmarked
before enabling" meant recognition quality, not the skip's cost). Same method, same bars:

  * the 349 archived stream-A pre-fusion snapshots (`stream_a_vlm/out/<id>/doc.06-*.json`)
    supply the layout regions and the VLM reading — the reading is engine-independent (same
    image), so it is reused; only the GEOMETRY track is re-run: the page is re-boxed with
    Apple Vision (OcrDet under prefer_apple_vision, 150 DPI, the production path);
  * for each threshold T, a page whose mean Vision confidence >= T has its `vlm_reading`
    blanked before fusion (exactly what the skip does); T=0 is the no-skip Vision baseline;
  * gated word recall / precision + band placement vs gold, per corpus (FUNSD 199, SROIE 150).

PRE-REGISTERED decision rule (written before any number is produced): the enabled default
T=0.92 is JUSTIFIED iff, on BOTH corpora, micro-avg Δword_recall and Δband_placement vs the
no-skip Vision baseline are each > -0.005 (stream E's benign bar; the noise floor is zero)
AND it skips >= 20% of pages (below that the saving isn't worth the behaviour). Otherwise the
recommendation is the HIGHEST T in {0.92, 0.95, 0.98} that clears both bars, or "disable
(set 0)" if none does — landing as its own PR, Luke decides. As with the Paddle pricing the
worst-hit skipped items are listed: a benign mean must not hide a catastrophic tail. Scope
limit stated up front: FUNSD/SROIE are scanned PRINT; the handwriting interaction is not
measurable here. Side output, for context only: how the Vision-routed pipeline (Vision
geometry + VLM reading) compares with the archived Paddle-routed run at T=0.

Run:  .venv/bin/python eval_out/vision_skip_cost.py            # sweep (durable CSV-append)
      .venv/bin/python eval_out/vision_skip_cost.py --report   # tables + recommendation
"""
import csv
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

from fusion_ocr import config as cm, ingest
from fusion_ocr.compose import reading_key
from fusion_ocr.eval.datasets import _CASELESS_REF, _ROOT, _annotation_index, iter_pairs
from fusion_ocr.eval.metrics import normalize, score
from fusion_ocr.eval.placement import gt_lines, placement_counts
from fusion_ocr.models import Document
from fusion_ocr.stages.fusion import Fusion
from fusion_ocr.stages.ocr_det import OcrDet

RES = Path("eval_out/vision_skip_cost"); RES.mkdir(parents=True, exist_ok=True)
CSV_OUT = RES / "results.csv"
ARCHIVE = Path("eval_out/stream_a_vlm/out")
PADDLE_CSV = Path("eval_out/paddle_skip_cost/results.csv")   # the Paddle-routed T=0 rows

THRESHOLDS = [0.92, 0.95, 0.98]          # 0 = the no-skip Vision baseline arm
BENIGN_BAR, MIN_SAVING = 0.005, 0.20     # pre-registered (see module docstring)

COLS = ["threshold", "dataset", "id", "skipped", "mean_conf", "n_boxes",
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


def mean_vision_conf(page) -> float | None:
    confs = [s.det_conf for s in page.segments
             if s.source == "vision" and s.det_conf is not None]
    return (sum(confs) / len(confs)) if confs else None


def rebox_with_vision(snap_path: Path, img_path: Path, cfg) -> Document:
    """The archived Document with its PaddleOCR boxes replaced by Apple Vision's — the
    production OcrDet path (prefer_apple_vision), on a derived PDF regenerated from the
    source image exactly as the archived run's ingest did (same page geometry)."""
    doc = Document.from_json(snap_path.read_text())
    pdf, _ = ingest.to_pdf(img_path, RES / "derived")
    doc.source_path = str(pdf)
    for page in doc.pages:
        page.segments = [s for s in page.segments if s.source == "textlayer"]
    return OcrDet().run(doc, cfg)


def run_sweep():
    refs = build_reference_maps()
    items = sorted(d.name for d in ARCHIVE.iterdir()
                   if d.is_dir() and (d / "doc.06-table_read.json").exists()
                   and d.name in refs)
    if not items:
        raise SystemExit(f"no snapshots under {ARCHIVE}")
    print(f"{len(items)} archived items; thresholds {[0.0] + THRESHOLDS}", flush=True)
    done = set()
    if CSV_OUT.exists():
        done = {(r["threshold"], r["dataset"], r["id"])
                for r in csv.DictReader(CSV_OUT.open())}
    newfile = not CSV_OUT.exists()
    fh = CSV_OUT.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
    if newfile:
        w.writeheader(); fh.flush()

    cfg = dataclasses.replace(cm.load(), prefer_apple_vision=True,
                              fuse_min_sim=0.34, fuse_det_conf_trust=0.80)
    for n, iid in enumerate(items):
        ds, ref, caseless, ann_path, img_path = refs[iid]
        if not normalize(ref):
            continue
        todo = [t for t in [0.0] + THRESHOLDS if (f"{t}", ds, iid.split("_", 1)[1]) not in done]
        if not todo:
            continue
        t0 = time.time()
        try:
            boxed = rebox_with_vision(ARCHIVE / iid / "doc.06-table_read.json", img_path, cfg)
        except Exception:
            print(f"  ERR rebox {iid}:", flush=True); traceback.print_exc(); continue
        boxed_json = boxed.to_json()
        for t in todo:
            try:
                doc = Document.from_json(boxed_json)
                skipped, conf, nb = 0, None, 0
                for page in boxed.pages:
                    conf = mean_vision_conf(page)
                    nb = sum(1 for s in page.segments if s.source == "vision")
                for page in doc.pages:
                    if t > 0 and conf is not None and conf >= t:
                        page.vlm_reading = ""       # the skip: det_text IS the reading
                        page.read_model = "apple_vision"
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
                                n_boxes=nb,
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
            print(f"[{n + 1}/{len(items)}] {iid} ({time.time() - t0:.1f}s)", flush=True)
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
    print("=== apple_vision_skip_vlm counterfactual cost (micro-avg vs the no-skip VISION "
          "baseline) ===")
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
    if PADDLE_CSV.exists():                  # context: Vision-routed vs Paddle-routed, no skip
        prow = [r for r in csv.DictReader(PADDLE_CSV.open()) if float(r["threshold"]) == 0]
        print("\ncontext — no-skip pipelines, Vision-routed vs the archived Paddle-routed run "
              "(same VLM reading; geometry track differs):")
        for ds in datasets:
            p = micro([r for r in prow if r["dataset"] == ds])
            v = base[ds]
            print(f"  {ds}: recall {v['recall']:.4f} vs {p['recall']:.4f} "
                  f"({v['recall'] - p['recall']:+.4f}); band {v['band']:.4f} vs {p['band']:.4f} "
                  f"({v['band'] - p['band']:+.4f}); precision {v['precision']:.4f} vs "
                  f"{p['precision']:.4f}")
    print(f"\nPRE-REGISTERED RULE -> "
          + ("the enabled default T=0.92 is JUSTIFIED" if 0.92 in qualifying
             else (f"recommend RAISING to T={max(qualifying)}" if qualifying
                   else "recommend DISABLING the Vision skip (set apple_vision_skip_vlm = 0)"))
          + f"  [bars: Δ > -{BENIGN_BAR} on recall+band per corpus; skip >= {MIN_SAVING:.0%}]"
          + "\n(any change lands as its own PR carrying this table — Luke decides)")


if __name__ == "__main__":
    report() if "--report" in sys.argv else run_sweep()
