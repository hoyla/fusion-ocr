"""Evidence-plan stream A — 150-DPI re-check. The shipped ocr_det default (200->150 DPI, PR #16)
was validated on just 5 pseudo-GT pages; this is the real test on human-GT gold. Deterministic
A/B (PaddleOCR, --no-vlm) at 150 vs 200 DPI on seeded n=50 FUNSD + n=50 SROIE.

Criterion (the noise floor is now 0, so any nonzero delta is 'real'; the question is MAGNITUDE
and SIGN): keep 150 if it is at least recognition-equivalent to 200 (it also bought a shared
raster cache + speed). Result 2026-07-08: 150 >= 200 on both sets (FUNSD +0.011 recall, SROIE
+0.0015) -> keep 150, confirmed on gold. Manifest: eval_out/manifests/dpi_recheck_2026-07-08.md.

DPI is a stage constant, NOT in the recipe fingerprint -> both arms would collide in the resume
cache. Each arm uses a distinct out_dir + digest to force genuine fresh processing (verified:
79/100 items differ between DPIs).

Run:  .venv/bin/python eval_out/dpi_recheck.py
"""
import csv, dataclasses, random, time
from pathlib import Path
from fusion_ocr import config as cm, ingest
from fusion_ocr.pipeline import process
from fusion_ocr.stages.triage import Triage
from fusion_ocr.stages.layout import Layout
from fusion_ocr.stages.table import Table
from fusion_ocr.stages.language import Language
from fusion_ocr.stages.ocr_det import OcrDet
from fusion_ocr.stages.fusion import Fusion
from fusion_ocr.stages.table_fill import TableFill
from fusion_ocr.stages.render import Render
from fusion_ocr.eval.harness import recovered_text
from fusion_ocr.eval.metrics import score, normalize
from fusion_ocr.eval.datasets import iter_pairs, _CASELESS_REF

RES = Path("eval_out/dpi_recheck"); RES.mkdir(parents=True, exist_ok=True)
CSV = RES / "results.csv"


def det_pipeline(dpi):
    return [Triage(), Layout(), Table(), Language(), OcrDet(dpi=dpi), Fusion(), TableFill(), Render()]


def sample(ds, n):
    allp = [(sp, img, ref) for sp in ("train", "test", "val") for img, ref in iter_pairs(ds, split=sp)]
    random.seed(3)
    return random.sample(allp, n)


WORK = [("funsd", sp, img, ref) for sp, img, ref in sample("funsd", 50)] + \
       [("sroie", sp, img, ref) for sp, img, ref in sample("sroie", 50)]

done = set()
if CSV.exists():
    done = {(int(r["dpi"]), r["dataset"], r["id"]) for r in csv.DictReader(CSV.open())}
newfile = not CSV.exists()
fh = CSV.open("a", newline="")
w = csv.DictWriter(fh, fieldnames=["dpi", "dataset", "id", "ref_chars", "recall", "cer", "secs"])
if newfile:
    w.writeheader(); fh.flush()

base = cm.load()
todo = [(dpi, ds, sp, img, ref) for dpi in (150, 200) for (ds, sp, img, ref) in WORK
        if (dpi, ds, img.stem) not in done]
print(f"resuming: {len(done)} done, {len(todo)} to go", flush=True)

for n, (dpi, ds, sp, img, ref) in enumerate(todo):
    t0 = time.time()
    if not normalize(ref):
        continue
    cfg = dataclasses.replace(base, out_dir=RES / f"out_{dpi}")
    pdf, _ = ingest.to_pdf(img, RES / "derived")
    doc = process(pdf, cfg, pipeline=det_pipeline(dpi), digest=f"d{dpi}_{ds}_{img.stem}")
    hyp = "\n".join(recovered_text(p) for p in doc.pages)
    s = score(ref, hyp, caseless=ds in _CASELESS_REF)
    w.writerow(dict(dpi=dpi, dataset=ds, id=img.stem, ref_chars=s["ref_chars"],
                    recall=round(s["word_recall"], 5), cer=round(s["cer"], 5),
                    secs=round(time.time() - t0, 1)))
    fh.flush()
    if n % 20 == 0:
        print(f"[{n+1}/{len(todo)}] dpi{dpi} {ds}/{img.stem} ({round(time.time()-t0,1)}s)", flush=True)
fh.close()
print("DONE", flush=True)
