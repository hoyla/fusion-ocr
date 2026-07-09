"""Evidence-plan stream F — model A/B: Qwen3.5-9B vs Qwen3-VL-8B at n>=50.

The generalist default switched to Qwen3.5-9B on 2026-06-30 on a Δ0.005 / n=4 signal — inside the
noise the campaign hadn't yet measured. Stream G later put that noise floor at ZERO, so a delta at
n>=50 is now a real result; this runner produces it and replaces the anecdote. Config-only: both
builds are 4-bit, cached, and served by the one MLX server (per-request by the model field).

Scope (matches the pin "the labelled set + seeded FUNSD n=50"): the 5 hand-labelled hard pages
(incl. the Mandelson handwriting — the case that motivated the switch) + a seeded FUNSD n=50
(seed=7, the campaign's FUNSD convention). BOTH models see the identical items. Scores the ungated
reading (`recovered_text`): word recall / precision / CER / WER / insertion + `t_vlm_read`.

The 4-bit-vs-8-bit QUANT arm is intentionally NOT here — it is BLOCKED under airgap (only 4-bit
Qwen3.5-9B is cached; the 8-bit build `mlx-community/Qwen3.5-9B-MLX-8bit` must be pulled outside
the seal first). See `manifests/stream_f_model_ab_2026-07-09.md`.

Durable/resumable CSV-append; model is the OUTER loop (one server reload). Needs MLX on :8080.
Run:  .venv/bin/python eval_out/stream_f_model_ab.py
"""
import csv
import dataclasses
import random
import time
import traceback
from pathlib import Path

from fusion_ocr import config as cm, ingest
from fusion_ocr.eval.datasets import iter_pairs
from fusion_ocr.eval.harness import recovered_text
from fusion_ocr.eval.labels import (_extract_pages, _render_pages_image_only,
                                     load_labelset)
from fusion_ocr.eval.metrics import normalize, score
from fusion_ocr.pipeline import process

RES = Path("eval_out/stream_f_model_ab"); RES.mkdir(parents=True, exist_ok=True)
DERIVED = RES / "derived"; DERIVED.mkdir(exist_ok=True)
CSV = RES / "results.csv"
# rounded rates for eyeballing + raw counts so the aggregate is a true micro-average
# (sum errors / sum length — the campaign convention, matching metrics.aggregate)
COLS = ["model", "kind", "id", "ref_chars", "ref_words", "word_overlap", "hyp_words",
        "char_errors", "char_ins", "word_errors", "word_recall", "word_precision",
        "cer", "wer", "insertion_rate", "t_vlm", "secs"]

MODELS = [
    ("q35_9b_4bit", "mlx-community/Qwen3.5-9B-MLX-4bit"),      # current default
    ("q3vl_8b_4bit", "mlx-community/Qwen3-VL-8B-Instruct-4bit"),  # rollback candidate (model-gen A/B)
    ("q35_9b_8bit", "mlx-community/Qwen3.5-9B-MLX-8bit"),      # quant A/B (vs the 4-bit default)
    ("q36_35ba3b_4bit", "mlx-community/Qwen3.6-35B-A3B-4bit"),  # new-gen arm (MoE, qwen3_5_moe, load-smoke-tested)
]

LABELSET = "eval_labels/labelset.json"
FUNSD_N, FUNSD_SEED = 50, 7   # seed 7 = the campaign's FUNSD convention (noise_floor.py)


# ---- build the shared item list (model-independent); prebuild the derived PDFs once ----
def build_items():
    items = []   # (kind, id, ref, caseless, pdf_path)
    for lab in load_labelset(LABELSET):
        ref = lab.reference()
        if not normalize(ref):
            continue
        pdf = DERIVED / f"label_{lab.id}.pdf"
        if not pdf.exists():
            (_render_pages_image_only if lab.render else _extract_pages)(lab.pdf, lab.pages, pdf)
        items.append(("label", lab.id, ref, False, pdf))

    allf = [(img, ref) for sp in ("train", "test", "val") for img, ref in iter_pairs("funsd", split=sp)]
    random.seed(FUNSD_SEED)
    for img, ref in random.sample(allf, FUNSD_N):
        if not normalize(ref):
            continue
        pdf, _ = ingest.to_pdf(img, DERIVED)
        items.append(("funsd", img.stem, ref, False, pdf))
    return items


ITEMS = build_items()
print(f"items/model: {sum(1 for i in ITEMS if i[0]=='label')} labelled + "
      f"{sum(1 for i in ITEMS if i[0]=='funsd')} funsd = {len(ITEMS)}", flush=True)

done = set()
if CSV.exists():
    done = {(r["model"], r["kind"], r["id"]) for r in csv.DictReader(CSV.open())}
newfile = not CSV.exists()
fh = CSV.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
if newfile:
    w.writeheader(); fh.flush()

base_cfg = cm.load()
for tag, model_id in MODELS:
    cfg = dataclasses.replace(base_cfg, vlm=dataclasses.replace(base_cfg.vlm, model=model_id))
    cfg = dataclasses.replace(cfg, out_dir=RES / "out" / tag)
    todo = [it for it in ITEMS if (tag, it[0], it[1]) not in done]
    print(f"\n=== {tag} ({model_id}) — {len(todo)}/{len(ITEMS)} to go ===", flush=True)
    for n, (kind, iid, ref, caseless, pdf) in enumerate(todo):
        t0 = time.time()
        try:
            doc = process(pdf, cfg, digest=f"{tag}_{kind}_{iid}")
            hyp = "\n".join(recovered_text(p) for p in doc.pages)
            s = score(ref, hyp, caseless=caseless)
            w.writerow(dict(model=tag, kind=kind, id=iid, ref_chars=s["ref_chars"],
                            ref_words=s["ref_words"], word_overlap=s["word_overlap"],
                            hyp_words=s["hyp_words"], char_errors=s["char_errors"],
                            char_ins=s["char_ins"], word_errors=s["word_errors"],
                            word_recall=round(s["word_recall"], 4),
                            word_precision=round(s["word_precision"], 4),
                            cer=round(s["cer"], 4), wer=round(s["wer"], 4),
                            insertion_rate=round(s["insertion_rate"], 4),
                            t_vlm=round(doc.stage_seconds.get("vlm_read", 0.0), 2),
                            secs=round(time.time() - t0, 1)))
            fh.flush()
        except Exception:
            print(f"  ERR {tag}/{kind}/{iid}:", flush=True); traceback.print_exc()
        if n % 10 == 0:
            print(f"  [{n+1}/{len(todo)}] {kind}/{iid} ({round(time.time()-t0,1)}s)", flush=True)
fh.close()

# ---- aggregate: micro-avg per model per set + A-vs-B deltas (noise floor = 0) ----
rows = list(csv.DictReader(CSV.open()))
for r in rows:
    for k in ("ref_chars", "ref_words", "word_overlap", "hyp_words", "char_errors",
              "char_ins", "word_errors"):
        r[k] = int(r[k])
    r["t_vlm"] = float(r["t_vlm"])


import statistics as _st


def agg(subset):
    # micro-average: total errors / total length (metrics.aggregate convention). ALSO report
    # median + a runaway count (hyp >5x ref chars) because char-level micro-avg is corruptible
    # by a single repetition-loop outlier (q3vl_8b hit one: a FUNSD form -> 262k chars of '.').
    cn = sum(r["ref_chars"] for r in subset) or 1
    wn = sum(r["ref_words"] for r in subset) or 1
    hw = sum(r["hyp_words"] for r in subset) or 1
    ov = sum(r["word_overlap"] for r in subset)
    m = len(subset) or 1
    runaway = [r for r in subset if r["char_ins"] > 5 * r["ref_chars"]]
    return {"n": len(subset),
            "recall": ov / wn, "precision": ov / hw,
            "cer": sum(r["char_errors"] for r in subset) / cn,
            "insertion": sum(r["char_ins"] for r in subset) / cn,
            "med_cer": _st.median([float(r["cer"]) for r in subset]) if subset else 0.0,
            "runaway": len(runaway),
            "t_vlm": sum(r["t_vlm"] for r in subset) / m}


print("\n=== STREAM F MODEL A/B SUMMARY (micro-avg; mean t_vlm; noise floor = 0) ===")
hdr = f"{'model':16} {'set':6} {'n':>3} {'recall':>7} {'prec':>7} {'cer':>8} {'medCER':>7} " \
      f"{'ins':>8} {'run':>3} {'t_vlm':>7}"
print(hdr)
report = {}
for tag, _ in MODELS:
    tagrows = [r for r in rows if r["model"] == tag]
    if not tagrows:
        continue   # not yet run (e.g. an arm added but not executed)
    report[tag] = {}
    for setname, pred in (("label", lambda r: r["kind"] == "label"),
                          ("funsd", lambda r: r["kind"] == "funsd"),
                          ("all", lambda r: True)):
        sub = [r for r in tagrows if pred(r)]
        a = agg(sub); report[tag][setname] = a
        print(f"{tag:16} {setname:6} {a['n']:>3} {a['recall']:>7.4f} {a['precision']:>7.4f} "
              f"{a['cer']:>8.4f} {a['med_cer']:>7.4f} {a['insertion']:>8.4f} {a['runaway']:>3} "
              f"{a['t_vlm']:>7.2f}")

base = MODELS[0][0]   # default = Qwen3.5-9B-4bit; compare every other arm against it
for tag, _ in MODELS[1:]:
    if tag not in report:
        continue
    print(f"\nΔ (default {base} − {tag}), per set [medCER = outlier-robust]:")
    for setname in ("label", "funsd", "all"):
        A, B = report[base][setname], report[tag][setname]
        print(f"  {setname:6}: Δrecall={A['recall']-B['recall']:+.4f}  "
              f"ΔmedCER={A['med_cer']-B['med_cer']:+.4f}  Δt_vlm={A['t_vlm']-B['t_vlm']:+.2f}s  "
              f"(runaway {base}/{tag} = {A['runaway']}/{B['runaway']})")
print("\n(noise floor is 0 → any delta is real, but the keep/switch call weighs quality vs "
      "speed/memory + robustness and is Luke's — this run replaces the n=4 anecdote, it does "
      "not flip the default. Micro CER/ins is corruptible by a single repetition-loop outlier "
      "→ medCER + runaway count are the honest signals.)")
print("DONE", flush=True)
