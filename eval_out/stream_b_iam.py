"""Evidence-plan stream B — IAM handwriting beyond n=1.

The headline handwriting capability was proven on ONE letter (the Mandelson note, recall ~0.95).
IAM's 1,539 pages sat unusable because the bundled per-image JSON is an OCR engine's output
(circular). With the FKI human transcriptions sourced (ascii/lines.txt, 2026-07-09) the
`datasets.iam_*` adapter now pairs every image to its human reading, so we can measure at scale.

Scope (pinned): seeded n=100 across splits (IAM pages are dense — sample first, extend if the
signal warrants). Per item we score BOTH engines against the human transcription, exactly like
stream A: the deterministic path (PaddleOCR, `deterministic_pipeline`) and the full VLM pipeline
(current default reader — Qwen3.5-9B-4bit). Case is significant (IAM keeps real case), so NOT
caseless. IAM is single-column, so line order = reading order (the reference is honest for CER).

CROP TO THE HANDWRITING (important — the n=1 smoke caught this): an IAM 'form' also shows a printed
header + the printed prompt the writer copied, above the handwriting. The lines.txt reference is the
handwriting only, so OCR'ing the whole page double-counts the text (printed prompt + handwriting →
CER > 1.0, and recall is inflated because the printed prompt gives every word for free). We crop
each image to the union of its handwritten line boxes (`datasets.iam_hw_bbox`) so the eval measures
HANDWRITING recognition, not the printed prompt. Verified by eye (a01-020u).

Interpretation guard (from the plan): IAM is CLEAN, ruled English handwriting — a FLOOR for the
handwriting claim, not proof on degraded FOI material.

Durable/resumable CSV-append; needs MLX on :8080 for the VLM column.
Run:  .venv/bin/python eval_out/stream_b_iam.py
"""
import csv
import dataclasses
import random
import time
import traceback
from pathlib import Path

from PIL import Image

from fusion_ocr import config as cm, ingest
from fusion_ocr.eval.datasets import iam_hw_bbox, iter_pairs
from fusion_ocr.eval.harness import recovered_text
from fusion_ocr.eval.metrics import normalize, score
from fusion_ocr.pipeline import deterministic_pipeline, process

_HW = iam_hw_bbox()      # form_id -> handwriting bbox (original-image px)
_CROP_MARGIN = 40


def hw_crop(img_path, dst_dir):
    """Crop an IAM form image to its handwriting region (+margin), save, return the crop path.
    Falls back to the full image if the form has no line boxes (shouldn't happen for real IAM)."""
    im = Image.open(img_path)
    box = _HW.get(img_path.stem)
    if box:
        W, H = im.size
        im = im.crop((max(0, box[0] - _CROP_MARGIN), max(0, box[1] - _CROP_MARGIN),
                      min(W, box[2] + _CROP_MARGIN), min(H, box[3] + _CROP_MARGIN)))
    dst = Path(dst_dir) / f"{img_path.stem}_hw.png"
    im.save(dst)
    return dst

RES = Path("eval_out/stream_b_iam")
DERIVED = RES / "derived"
CSV = RES / "results.csv"
N, SEED = 100, 1
COLS = ["id", "split", "ref_chars", "ref_words",
        "det_recall", "det_cer", "det_ins", "det_overlap", "det_hypwords",
        "vlm_recall", "vlm_cer", "vlm_ins", "vlm_overlap", "vlm_hypwords",
        "t_vlm", "secs"]


def micro(rows, pfx):
    import statistics as st
    wn = sum(int(r["ref_words"]) for r in rows) or 1
    hw = sum(int(r[f"{pfx}_hypwords"]) for r in rows) or 1
    ov = sum(int(r[f"{pfx}_overlap"]) for r in rows)
    return dict(recall=ov / wn, precision=ov / hw,
                med_cer=st.median(float(r[f"{pfx}_cer"]) for r in rows) if rows else 0,
                mean_ins=sum(float(r[f"{pfx}_ins"]) for r in rows) / (len(rows) or 1))


def main():
    RES.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(exist_ok=True)
    # seeded n=100 across all splits (fixed set)
    allp = [(sp, img, ref) for sp in ("train", "test", "val")
            for img, ref in iter_pairs("iam", split=sp)]
    random.seed(SEED)
    work = random.sample(allp, N)
    print(f"IAM universe {len(allp)} -> seeded n={len(work)} (seed={SEED})", flush=True)

    done = {r["id"] for r in csv.DictReader(CSV.open())} if CSV.exists() else set()
    newfile = not CSV.exists()
    fh = CSV.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
    if newfile:
        w.writeheader(); fh.flush()

    cfg = cm.load()
    det_cfg = dataclasses.replace(cfg, out_dir=RES / "out_det")
    vlm_cfg = dataclasses.replace(cfg, out_dir=RES / "out_vlm")
    todo = [t for t in work if t[1].stem not in done]
    print(f"resuming: {len(done)} done, {len(todo)} to go", flush=True)

    for n, (split, img, ref) in enumerate(todo):
        t0 = time.time()
        try:
            if not normalize(ref):
                continue
            crop = hw_crop(img, DERIVED)             # isolate handwriting (drop printed prompt)
            pdf, _ = ingest.to_pdf(crop, DERIVED)
            det_doc = process(pdf, det_cfg, pipeline=deterministic_pipeline(),
                              digest=f"iam_det_{img.stem}")
            det_hyp = "\n".join(recovered_text(p) for p in det_doc.pages)
            ds = score(ref, det_hyp, caseless=False)
            t_vlm0 = time.time()
            vlm_doc = process(pdf, vlm_cfg, digest=f"iam_vlm_{img.stem}")
            t_vlm = vlm_doc.stage_seconds.get("vlm_read", round(time.time() - t_vlm0, 2))
            vlm_hyp = "\n".join(recovered_text(p) for p in vlm_doc.pages)
            vs = score(ref, vlm_hyp, caseless=False)
            w.writerow(dict(
                id=img.stem, split=split, ref_chars=ds["ref_chars"], ref_words=ds["ref_words"],
                det_recall=round(ds["word_recall"], 4), det_cer=round(ds["cer"], 4),
                det_ins=round(ds["insertion_rate"], 4), det_overlap=ds["word_overlap"],
                det_hypwords=ds["hyp_words"],
                vlm_recall=round(vs["word_recall"], 4), vlm_cer=round(vs["cer"], 4),
                vlm_ins=round(vs["insertion_rate"], 4), vlm_overlap=vs["word_overlap"],
                vlm_hypwords=vs["hyp_words"],
                t_vlm=round(t_vlm, 2), secs=round(time.time() - t0, 1)))
            fh.flush()
        except Exception:
            print(f"  ERR iam/{img.stem}:", flush=True); traceback.print_exc()
        if n % 10 == 0:
            print(f"[{n+1}/{len(todo)}] iam/{img.stem} ({round(time.time()-t0,1)}s)", flush=True)
    fh.close()

    rows = list(csv.DictReader(CSV.open()))
    print(f"\n=== STREAM B IAM SUMMARY (n={len(rows)}; micro-avg; case-sensitive) ===")
    for pfx, name in (("det", "deterministic (PaddleOCR)"), ("vlm", "VLM (Qwen3.5-9B-4bit)")):
        m = micro(rows, pfx)
        print(f"  {name:28} recall={m['recall']:.4f} precision={m['precision']:.4f} "
              f"medCER={m['med_cer']:.4f} meanINS={m['mean_ins']:.4f}")
    print("\n(Mandelson n=1 was VLM recall ~0.95; IAM is clean ruled English handwriting = a FLOOR.)")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
