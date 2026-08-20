"""Evidence-plan stream E1/E2 — threshold sensitivity of the fusion pair (pins 2026-07-12).

E1 (zero-VLM): re-run `Fusion()` in-process over the 349 archived stream-A pre-fusion
snapshots (`doc.06-table_read.json`), one-at-a-time sweeping the two config-exposed fusion
constants, and score the GATED product text + band placement per config:

    fuse_min_sim        in {0.238, 0.289, 0.34*, 0.391, 0.442}
    fuse_det_conf_trust in {0.56, 0.68, 0.80*, 0.92, 1.04}   (* = shared baseline;
                          1.04 exceeds any real confidence = a guard-OFF endpoint, not a
                          linear point)

`_MR_COVERAGE` / `_LARGE_IMAGE_FRAC` are NOT swept — flat by construction on these corpora
(no text layer; full-page images); see the pins ("Deviation 1"). Re-trigger: first
mixed-content gold corpus.

Snapshots are loaded FRESH from disk per (item, config) — fusion mutates the Document.
Response variables per corpus (SROIE caseless): gated word_recall + word_precision (D1's
gated-text definition = the `recovered_text` fallback branch applied unconditionally) and
BAND placement (strict logged too; char metrics logged, not headline).

Built-in integrity check: on the baseline config, (a) the UNGATED reading re-scored from the
snapshot must reproduce the committed `stream_a_vlm/results.csv` row, and (b) the re-fused
STRICT placement counts must reproduce the archived placed/plain/gt_words ints. Mismatch ->
STOP (archive/code drift, by measurement not assumption).

E2 (`--e2`, needs the MLX reader): seeded n=20 FUNSD (seed 1) through the FULL pipeline once
at baseline under the current default reader; keep the fresh doc.06 snapshots; fusion-only
sweep them across the 8 non-baseline configs. Check = direction of effect agrees with E1.

n: all 349; pre-authorized fallback to a seeded n=150 (seed 1) iff projected wall-clock
> ~3 h (the runner measures the first items and projects, printing the decision).

Escalation tripwires (diagnosis triggers, not pass bars — any firing is reported loudly and
the manifest verdict is left PENDING senior eyes): (a) integrity-check failure; (b)
word_recall and band placement moving in OPPOSITE directions at MATERIAL magnitude
(both |Δ| >= 0.005) anywhere in a sweep; (c) E2 direction of effect disagreeing with E1.

Durable/resumable CSV-append (D1 style). Run:
    .venv/bin/python eval_out/stream_e_sensitivity.py            # E1
    .venv/bin/python eval_out/stream_e_sensitivity.py --e2       # E2 (reader on :8080)
    .venv/bin/python eval_out/stream_e_sensitivity.py --report   # sweep tables + verdicts
"""
import csv
import dataclasses
import json
import random
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

RES = Path("eval_out/stream_e_sensitivity"); RES.mkdir(parents=True, exist_ok=True)
CSV_OUT = RES / "results.csv"
ARCHIVE = Path("eval_out/stream_a_vlm/out")          # the copied CORSAIR snapshots
STREAM_A_CSV = Path("eval_out/stream_a_vlm/results.csv")
E2_OUT = RES / "e2_out"

BASELINE = {"fuse_min_sim": 0.34, "fuse_det_conf_trust": 0.80}
SWEEPS = {"fuse_min_sim": [0.238, 0.289, 0.34, 0.391, 0.442],
          "fuse_det_conf_trust": [0.56, 0.68, 0.80, 0.92, 1.04]}
BENIGN_BAR, LOAD_BEARING_BAR = 0.005, 0.02           # pinned materiality bars
FALLBACK_N, FALLBACK_SEED, PROJECT_BUDGET_S = 150, 1, 3 * 3600
E2_N, E2_SEED = 20, 1

COLS = ["phase", "constant", "value", "dataset", "id",
        "ref_chars", "ref_words", "word_overlap", "hyp_words", "char_errors", "char_ins",
        "word_recall", "word_precision", "cer", "insertion_rate",
        "placed_band", "placed_strict", "plain", "gt_words", "segs", "secs"]


def configs():
    """(constant, value, cfg) for the 9 distinct configs; baseline appears once."""
    base = cm.load()
    base = dataclasses.replace(base, **BASELINE)     # pin the baseline explicitly
    yield ("baseline", BASELINE["fuse_min_sim"], base)
    for const, values in SWEEPS.items():
        for v in values:
            if v == BASELINE[const]:
                continue
            yield (const, v, dataclasses.replace(base, **{const: v}))


def gated_text(doc) -> str:
    """D1's gated-text definition: the `recovered_text` fallback branch, unconditionally —
    the text of segment_index.json / the overlay."""
    parts = []
    for page in doc.pages:
        segs = [s for s in page.segments if s.best_text and not s.superseded]
        segs.sort(key=lambda s: reading_key(
            s, page.regions, page.rotation, page.width, page.height))
        parts.append("\n".join(s.best_text for s in segs))
    return "\n".join(parts)


def build_reference_maps():
    """id ('<ds>_<stem>') -> (ds, ref, caseless, ann_path, img_path)."""
    ann = {"funsd": _annotation_index(_ROOT / "form"),
           "sroie": _annotation_index(_ROOT / "invoice")}
    out = {}
    for ds in ("funsd", "sroie"):
        for sp in ("train", "test", "val"):
            for img, ref in iter_pairs(ds, split=sp):
                ap = ann[ds].get(img.stem)
                out[f"{ds}_{img.stem}"] = (ds, ref, ds in _CASELESS_REF, ap, img)
    return out


def snapshot_path(item_dir: Path) -> Path:
    return item_dir / "doc.06-table_read.json"


def score_config(snap: Path, cfg, ds, ref, caseless, ann_path, img_path):
    """Load the pre-fusion snapshot FRESH, fuse under cfg, score gated text + placement."""
    doc = Document.from_json(snap.read_text())
    doc = Fusion().run(doc, cfg)
    s = score(ref, gated_text(doc), caseless=caseless)
    pb = ps = {"placed": 0, "plain": 0, "total": 0, "segs": 0}
    if ann_path and img_path and Path(img_path).exists():
        lines = gt_lines(json.loads(Path(ann_path).read_text()), ds)
        if lines:
            W, H = Image.open(img_path).size
            page = doc.pages[0]
            pb = placement_counts(page, lines, W, H, caseless=caseless, band=True)
            ps = placement_counts(page, lines, W, H, caseless=caseless)
    return doc, s, pb, ps


def integrity_check(items, refs, n=8):
    """Tripwire (a): the archive must be the committed stream-A run, by measurement.
    (a1) ungated re-score == results.csv row (4dp); (a2) baseline strict placement counts
    == archived ints. Checked on the first `n` items of each dataset."""
    committed = {r["dataset"] + "_" + r["id"]: r for r in csv.DictReader(STREAM_A_CSV.open())}
    base_cfg = next(c for name, _v, c in configs() if name == "baseline")
    checked = 0
    per_ds = {"funsd": 0, "sroie": 0}
    for iid in items:
        ds, ref, caseless, ann_path, img_path = refs[iid]
        if per_ds[ds] >= n or iid not in committed:
            continue
        row = committed[iid]
        doc = Document.from_json(snapshot_path(ARCHIVE / iid).read_text())
        ungated = "\n".join(p.vlm_reading for p in doc.pages)
        s = score(ref, ungated, caseless=caseless)
        for col in ("word_recall", "word_precision", "cer"):
            if abs(s[col] - float(row[col])) > 5e-5:
                return False, f"{iid}: ungated {col} {s[col]:.4f} != committed {row[col]}"
        _doc, _s, _pb, ps = score_config(snapshot_path(ARCHIVE / iid), base_cfg,
                                         ds, ref, caseless, ann_path, img_path)
        if (ps["placed"], ps["plain"], ps["total"]) != \
                (int(row["placed"]), int(row["plain_placed"]), int(row["gt_words"])):
            return False, (f"{iid}: strict placement {ps['placed']}/{ps['plain']}/{ps['total']}"
                           f" != committed {row['placed']}/{row['plain_placed']}/{row['gt_words']}")
        per_ds[ds] += 1; checked += 1
    return checked > 0, f"{checked} items reproduce the committed stream-A columns"


def run_items(phase, items, refs, snap_root, done, writer, fh):
    cfgs = list(configs())
    t_start, n_done = time.time(), 0
    for iid in items:
        ds, ref, caseless, ann_path, img_path = refs[iid]
        if not normalize(ref):
            continue
        snap = snapshot_path(snap_root / iid)
        if not snap.exists():
            print(f"  MISSING snapshot {snap}", flush=True); continue
        for const, val, cfg in cfgs:
            key = (phase, const, f"{val}", ds, iid.split("_", 1)[1])
            if key in done:
                continue
            t0 = time.time()
            try:
                _doc, s, pb, ps = score_config(snap, cfg, ds, ref, caseless, ann_path, img_path)
                writer.writerow(dict(
                    phase=phase, constant=const, value=val, dataset=ds,
                    id=iid.split("_", 1)[1], ref_chars=s["ref_chars"],
                    ref_words=s["ref_words"], word_overlap=s["word_overlap"],
                    hyp_words=s["hyp_words"], char_errors=s["char_errors"],
                    char_ins=s["char_ins"], word_recall=round(s["word_recall"], 4),
                    word_precision=round(s["word_precision"], 4), cer=round(s["cer"], 4),
                    insertion_rate=round(s["insertion_rate"], 4),
                    placed_band=pb["placed"], placed_strict=ps["placed"], plain=pb["plain"],
                    gt_words=pb["total"], segs=pb["segs"], secs=round(time.time() - t0, 2)))
                fh.flush()
            except Exception:
                print(f"  ERR {iid} {const}={val}:", flush=True); traceback.print_exc()
        n_done += 1
        if n_done == 5 and phase == "e1":
            per_item = (time.time() - t_start) / n_done
            proj = per_item * len(items)
            print(f"[projection] {per_item:.1f}s/item x {len(items)} = {proj/3600:.2f}h", flush=True)
            if proj > PROJECT_BUDGET_S and len(items) > FALLBACK_N:
                random.seed(FALLBACK_SEED)
                keep = set(random.sample(sorted(items), FALLBACK_N))
                items[:] = [i for i in items if i in keep]
                print(f"[projection] > {PROJECT_BUDGET_S/3600:.0f}h -> pre-authorized "
                      f"fallback to seeded n={FALLBACK_N} (seed {FALLBACK_SEED})", flush=True)
        if n_done % 20 == 0:
            print(f"[{n_done}/{len(items)}] {iid}", flush=True)


def main_e1():
    refs = build_reference_maps()
    items = sorted(d.name for d in ARCHIVE.iterdir()
                   if d.is_dir() and snapshot_path(d).exists() and d.name in refs)
    if not items:
        raise SystemExit(f"no snapshots under {ARCHIVE} — copy doc.06-table_read.json from "
                         "the CORSAIR archive first (see evidence_plan.md §E pins)")
    print(f"E1: {len(items)} archived items with references", flush=True)
    ok, msg = integrity_check(items, refs)
    print(f"integrity check: {'PASS' if ok else 'TRIPWIRE (a) FIRED'} — {msg}", flush=True)
    if not ok:
        raise SystemExit("STOP: archive does not reproduce the committed stream-A columns. "
                         "Diagnose drift before any sweep number is produced.")
    done = set()
    if CSV_OUT.exists():
        done = {(r["phase"], r["constant"], r["value"], r["dataset"], r["id"])
                for r in csv.DictReader(CSV_OUT.open())}
    newfile = not CSV_OUT.exists()
    fh = CSV_OUT.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
    if newfile:
        w.writeheader(); fh.flush()
    run_items("e1", items, refs, ARCHIVE, done, w, fh)
    fh.close(); print("E1 DONE", flush=True)


def main_e2():
    from fusion_ocr import ingest
    from fusion_ocr.pipeline import process
    refs = build_reference_maps()
    allf = [(img, ref) for sp in ("train", "test", "val")
            for img, ref in iter_pairs("funsd", split=sp)]
    random.seed(E2_SEED)
    sample = random.sample(allf, E2_N)
    cfg = next(c for name, _v, c in configs() if name == "baseline")
    cfg = dataclasses.replace(cfg, out_dir=E2_OUT)
    for img, _ref in sample:                          # baseline full-pipeline reads
        iid = f"funsd_{img.stem}"
        if snapshot_path(E2_OUT / iid).exists():
            continue
        pdf, _ = ingest.to_pdf(img, RES / "derived")
        print(f"E2 read {iid}", flush=True)
        process(pdf, cfg, digest=iid)
    items = [f"funsd_{img.stem}" for img, _ in sample]
    done = set()
    if CSV_OUT.exists():
        done = {(r["phase"], r["constant"], r["value"], r["dataset"], r["id"])
                for r in csv.DictReader(CSV_OUT.open())}
    newfile = not CSV_OUT.exists()
    fh = CSV_OUT.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
    if newfile:
        w.writeheader(); fh.flush()
    run_items("e2", items, refs, E2_OUT, done, w, fh)
    fh.close(); print("E2 DONE", flush=True)


# ---- reporting: sweep tables, verdicts, tripwires ----
def micro(rows):
    wn = sum(int(r["ref_words"]) for r in rows) or 1
    hw = sum(int(r["hyp_words"]) for r in rows) or 1
    ov = sum(int(r["word_overlap"]) for r in rows)
    gt = sum(int(r["gt_words"]) for r in rows) or 1
    return {"n": len(rows), "recall": ov / wn, "precision": ov / hw,
            "band": sum(int(r["placed_band"]) for r in rows) / gt,
            "strict": sum(int(r["placed_strict"]) for r in rows) / gt}


def sweep_table(rows, phase):
    """{constant: {value: {dataset: micro}}} with the shared baseline filled in."""
    base = [r for r in rows if r["phase"] == phase and r["constant"] == "baseline"]
    out = {}
    for const, values in SWEEPS.items():
        out[const] = {}
        for v in values:
            sel = base if v == BASELINE[const] else \
                [r for r in rows if r["phase"] == phase and r["constant"] == const
                 and float(r["value"]) == v]
            out[const][v] = {ds: micro([r for r in sel if r["dataset"] == ds])
                             for ds in ("funsd", "sroie")
                             if any(r["dataset"] == ds for r in sel)}
    return out


def verdicts(table):
    """Pinned bars on |Δ recall| and |Δ band| vs baseline; plus tripwire (b)."""
    out, trips = {}, []
    for const, byval in table.items():
        base = byval[BASELINE[const]]
        deltas = []
        for v, byds in byval.items():
            if v == BASELINE[const]:
                continue
            for ds, m in byds.items():
                if ds not in base or not m["n"]:
                    continue
                dr, db = m["recall"] - base[ds]["recall"], m["band"] - base[ds]["band"]
                deltas += [abs(dr), abs(db)]
                if abs(dr) >= BENIGN_BAR and abs(db) >= BENIGN_BAR and dr * db < 0:
                    trips.append(f"(b) {const}={v} on {ds}: recall {dr:+.4f} vs band {db:+.4f}"
                                 " move in opposite directions at material magnitude")
        mx = max(deltas) if deltas else 0.0
        out[const] = ("load-bearing" if mx >= LOAD_BEARING_BAR
                      else "benign" if mx < BENIGN_BAR else "mild", mx)
    return out, trips


def main_report():
    rows = list(csv.DictReader(CSV_OUT.open()))
    for phase in ("e1", "e2"):
        if not any(r["phase"] == phase for r in rows):
            print(f"[{phase}: no rows yet]"); continue
        table = sweep_table(rows, phase)
        vd, trips = verdicts(table)
        print(f"\n=== {phase.upper()} sweep (micro-avg; baseline * ) ===")
        for const, byval in table.items():
            print(f"\n{const}  ->  {vd[const][0]} (max |Δ| {vd[const][1]:.4f}; "
                  f"bars: <{BENIGN_BAR} benign, >={LOAD_BEARING_BAR} load-bearing)")
            print(f"  {'value':>7} {'ds':>6} {'n':>4} {'recall':>8} {'prec':>8} "
                  f"{'band':>8} {'strict':>8}")
            for v, byds in byval.items():
                mark = "*" if v == BASELINE[const] else " "
                for ds, m in byds.items():
                    print(f"  {v:>6}{mark} {ds:>6} {m['n']:>4} {m['recall']:>8.4f} "
                          f"{m['precision']:>8.4f} {m['band']:>8.4f} {m['strict']:>8.4f}")
        for t in trips:
            print(f"\nTRIPWIRE {t}")
        if trips:
            print("-> manifest verdict PENDING senior eyes (pins: diagnose before verdict)")
    print("\n(write the manifest from these tables: manifests/stream_e_sensitivity_<date>.md)")


if __name__ == "__main__":
    if "--report" in sys.argv:
        main_report()
    elif "--e2" in sys.argv:
        main_e2()
    else:
        main_e1()
