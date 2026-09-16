"""Engine A/B — the FULL-SET verdict (closes manifests/engine_ab_2026-08-20.md).

The n=30 A/B (engine_ab.py) found rapid_medrec (RapidOCR, PP-OCRv6 *medium* rec ONNX) met
all three pre-registered adoption criteria (rapidocr_eval_plan.md §Decision criteria):
(a) measurably faster, (b) recall within ~0.01 + reading order preserved on the labelled set,
(c) box geometry equivalent. The full-set confirmation needed a paddle_v6m full run on the
SAME runner/scoring; that ran on the desktop (2026-08-20, ~8 h) and its 1,112 rows are merged
here by key. Method pins honoured: the desktop run contributes QUALITY columns only (Luke,
PR #45 — speed is same-machine only; the speed evidence stays the laptop n=30 table); the
labelled-set reading-order check that criterion (b) also requires — never run for
rapid_medrec (the labelset was in its false-alarm window) — is run here, deterministic-only,
both arms, appended as dataset="labels" rows.

Three modes, all durable / idempotent:
  --merge <desktop_results.csv>   keyed merge of the desktop rows into results.csv
                                  (shared keys must be byte-identical; only new keys appended)
  --labels                        run the 5 labelled pages for paddle_v6m + rapid_medrec (no VLM)
  --report                        full-set micro tables, PAIRED per-item deltas with a bootstrap
                                  CI, the n=30-vs-full check, the labelled rows, and the verdict
                                  lines against the pre-registered criteria (markdown)
"""
import csv
import dataclasses
import hashlib
import statistics
import sys
import time
from pathlib import Path

RES = Path("eval_out/engine_ab")
CSV_OUT = RES / "results.csv"
COLS = ["arm", "dataset", "id", "ref_chars", "ref_words", "word_overlap", "hyp_words",
        "word_recall", "word_precision", "cer", "insertion_rate", "t_ocr_det", "secs"]
ARMS = {"paddle_v6m": {}, "rapid_medrec": {"prefer_rapidocr": True}}
RECALL_BAR = 0.01          # plan: "recall within ~0.01"
TAIL = 0.05                # per-item |Δrecall| beyond this = a big disagreement


def _rows():
    return list(csv.DictReader(CSV_OUT.open()))


def merge(desktop_csv: str) -> None:
    src = Path(desktop_csv)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    have = _rows()
    keyed = {(r["arm"], r["dataset"], r["id"]): r for r in have}
    new, same = [], 0
    for r in csv.DictReader(src.open()):
        k = (r["arm"], r["dataset"], r["id"])
        if k in keyed:
            assert keyed[k] == r, f"shared key differs between machines: {k}"
            same += 1
        else:
            new.append(r)
    with CSV_OUT.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        for r in new:
            w.writerow(r)
    arms = sorted({(r["arm"], r["dataset"]) for r in new})
    print(f"merged {len(new)} new rows from {src.name} (sha256 {digest[:16]}…); "
          f"{same} shared keys identical; new rows from {arms}")


def labels() -> None:
    from fusion_ocr import config as cm
    from fusion_ocr.engines import rapid
    from fusion_ocr.eval.labels import evaluate_labelset

    done = {(r["arm"], r["dataset"], r["id"]) for r in _rows()}
    base = cm.load()
    fh = CSV_OUT.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    from fusion_ocr.eval.labels import load_labelset
    label_ids = [lab.id for lab in load_labelset("eval_labels/labelset.json")]
    for arm, overrides in ARMS.items():
        if all((arm, "labels", lid) in done for lid in label_ids):
            print(f"== {arm}: labelled rows already present, skipping", flush=True)
            continue
        rapid.set_rec_tier("medium" if arm == "rapid_medrec" else None)
        cfg = dataclasses.replace(base, out_dir=RES / "out" / arm / "labels", **overrides)
        work = RES / "out" / arm / "labels_work"      # under eval_out, never /tmp
        work.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rows = evaluate_labelset("eval_labels/labelset.json", cfg, tmp_root=work, no_vlm=True)
        for r in rows:
            if r.get("status") != "scored" or (arm, "labels", r["id"]) in done:
                continue
            w.writerow(dict(arm=arm, dataset="labels", id=r["id"],
                            ref_chars=r["ref_chars"], ref_words=r["ref_words"],
                            word_overlap=r["word_overlap"], hyp_words=r["hyp_words"],
                            word_recall=round(r["word_recall"], 4),
                            word_precision=round(r["word_precision"], 4),
                            cer=round(r["cer"], 4),
                            insertion_rate=round(r["insertion_rate"], 4),
                            t_ocr_det="", secs=round(time.time() - t0, 1)))
            fh.flush()
        print(f"== {arm}: {len(rows)} labelled rows in {time.time() - t0:.0f}s", flush=True)
    fh.close()
    print("LABELS DONE", flush=True)


def _micro(sub):
    wn = sum(int(r["ref_words"]) for r in sub) or 1
    hw = sum(int(r["hyp_words"]) for r in sub) or 1
    ov = sum(int(r["word_overlap"]) for r in sub)
    cn = sum(int(r["ref_chars"]) for r in sub) or 1
    return {"n": len(sub), "recall": ov / wn, "prec": ov / hw,
            "cer": sum(float(r["cer"]) * int(r["ref_chars"]) for r in sub) / cn}


def _paired(a_rows, b_rows):
    """Per-item deltas (b - a) on identical items, keyed by (dataset, id)."""
    a = {(r["dataset"], r["id"]): r for r in a_rows}
    b = {(r["dataset"], r["id"]): r for r in b_rows}
    keys = sorted(set(a) & set(b))
    return keys, [float(b[k]["word_recall"]) - float(a[k]["word_recall"]) for k in keys], \
        [float(b[k]["word_precision"]) - float(a[k]["word_precision"]) for k in keys]


def _boot_ci(xs, iters=10_000, seed=1):
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(iters)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _q(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def report() -> None:
    rows = _rows()
    by = lambda arm, ds: [r for r in rows if r["arm"] == arm and r["dataset"] == ds and r["word_recall"]]  # noqa: E731

    print("## Full-set quality (same runner + scoring; desktop rows = quality columns only)\n")
    print("| arm | ds | n | recall | precision | CER |")
    print("| --- | --- | --- | --- | --- | --- |")
    agg = {}
    for ds in ("funsd", "sroie"):
        for arm in ARMS:
            a = _micro(by(arm, ds))
            agg[(arm, ds)] = a
            print(f"| {arm} | {ds} | {a['n']} | {a['recall']:.4f} | {a['prec']:.4f} | {a['cer']:.4f} |")
    print()
    for ds in ("funsd", "sroie"):
        p, r = agg[("paddle_v6m", ds)], agg[("rapid_medrec", ds)]
        print(f"- {ds}: Δrecall **{r['recall'] - p['recall']:+.4f}**, Δprecision "
              f"{r['prec'] - p['prec']:+.4f}, ΔCER {r['cer'] - p['cer']:+.4f} (n={p['n']})")

    print("\n## Paired per-item deltas, rapid_medrec − paddle_v6m (identical items)\n")
    print("| ds | n | mean Δrecall | 95% bootstrap CI | median | p10 | p90 | min | max "
          "| within ±0.01 | < −0.05 | > +0.05 | mean Δprec |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    pooled_r, pooled_p = [], []
    for ds in ("funsd", "sroie", "pooled"):
        if ds == "pooled":
            dr, dp = pooled_r, pooled_p
        else:
            _, dr, dp = _paired(by("paddle_v6m", ds), by("rapid_medrec", ds))
            pooled_r += dr
            pooled_p += dp
        lo, hi = _boot_ci(dr)
        n = len(dr)
        print(f"| {ds} | {n} | {statistics.fmean(dr):+.4f} | [{lo:+.4f}, {hi:+.4f}] "
              f"| {statistics.median(dr):+.4f} | {_q(dr, 0.10):+.3f} | {_q(dr, 0.90):+.3f} "
              f"| {min(dr):+.3f} | {max(dr):+.3f} "
              f"| {sum(abs(x) <= 0.01 for x in dr) / n:.0%} "
              f"| {sum(x < -TAIL for x in dr)} | {sum(x > TAIL for x in dr)} "
              f"| {statistics.fmean(dp):+.4f} |")

    print("\n## Was the seeded n=30 representative? (its deltas vs the full set)\n")
    import random
    from fusion_ocr.eval.datasets import iter_pairs
    for ds in ("funsd", "sroie"):
        pool = [img.stem for sp in ("train", "test", "val") for img, _ in iter_pairs(ds, split=sp)]
        random.seed(1)
        sample = set(random.sample(pool, 30))
        keys, dr, _ = _paired(by("paddle_v6m", ds), by("rapid_medrec", ds))
        sub = [d for k, d in zip(keys, dr) if k[1] in sample]
        rest = [d for k, d in zip(keys, dr) if k[1] not in sample]
        print(f"- {ds}: n=30 mean Δrecall {statistics.fmean(sub):+.4f} (n={len(sub)}) vs the other "
              f"{len(rest)} items {statistics.fmean(rest):+.4f} — full {statistics.fmean(dr):+.4f}")

    lab = {arm: {r["id"]: r for r in by(arm, "labels")} for arm in ARMS}
    if lab["paddle_v6m"] and lab["rapid_medrec"]:
        print("\n## Labelled set, deterministic-only (criterion (b): reading order preserved)\n")
        print("| label | paddle CER | rapid CER | ΔCER | paddle recall | rapid recall | Δrecall |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        worst = []
        for lid in sorted(set(lab["paddle_v6m"]) & set(lab["rapid_medrec"])):
            p, r = lab["paddle_v6m"][lid], lab["rapid_medrec"][lid]
            dc = float(r["cer"]) - float(p["cer"])
            dr = float(r["word_recall"]) - float(p["word_recall"])
            worst.append((dc, dr, lid))
            print(f"| {lid} | {float(p['cer']):.3f} | {float(r['cer']):.3f} | {dc:+.3f} "
                  f"| {float(p['word_recall']):.3f} | {float(r['word_recall']):.3f} | {dr:+.3f} |")
        bad = [(dc, dr, lid) for dc, dr, lid in worst if dc > RECALL_BAR or dr < -RECALL_BAR]
        print()
        print("- CER is order-SENSITIVE: a CER rise without a recall fall is reading-order "
              "damage; equal CER and recall means order and recognition both held.")
        print(f"- Labels breaching the ~0.01 bar on CER or recall: "
              f"{', '.join(f'{lid} (ΔCER {dc:+.3f}, Δrecall {dr:+.3f})' for dc, dr, lid in bad) or 'none'}")

    print("\n## Verdict lines against the pre-registered criteria\n")
    for ds in ("funsd", "sroie"):
        d = agg[("rapid_medrec", ds)]["recall"] - agg[("paddle_v6m", ds)]["recall"]
        print(f"- (b) recall within ~0.01 — {ds}: {d:+.4f} → "
              f"{'MET' if d >= -RECALL_BAR else 'NOT met (outside the bar)'}")
    print("- (a) faster: same-machine only — the laptop n=30 table (×4.5 FUNSD / ×7.1 SROIE) "
          "stands; the desktop t_ocr_det is NOT compared (method pin).")
    print("- (c) geometry equivalent: pre-verified (25/25 lines IoU ≥ 0.5, ~1 px).")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge(sys.argv[sys.argv.index("--merge") + 1])
    elif "--labels" in sys.argv:
        labels()
    elif "--report" in sys.argv:
        report()
    else:
        print(__doc__)
