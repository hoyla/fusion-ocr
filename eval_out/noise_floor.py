"""Evidence-plan stream G — noise floor. Run the IDENTICAL config 3x on a fixed seeded-30 FUNSD
set (full VLM pipeline, temperature already 0.0) to measure residual MLX/decode/env variance.
A delta inside this floor is not a result (the standing gate for every future 'X beats Y',
incl. the RapidOCR A/B).

CRUCIAL: each run uses a distinct out_dir + digest, so the recipe-fingerprint resume cache does
NOT return an identical cached doc (which would falsely show zero variance). The 3 runs are
genuinely independent re-processings of the same inputs under the same config.

Run:  .venv/bin/python eval_out/noise_floor.py
"""
import csv, json, random, time
from pathlib import Path
import dataclasses
from PIL import Image
from fusion_ocr import config as cm, ingest
from fusion_ocr.pipeline import process
from fusion_ocr.eval.harness import recovered_text
from fusion_ocr.eval.metrics import score
from fusion_ocr.eval.datasets import iter_pairs, _annotation_index, _ROOT
from fusion_ocr.eval.placement import gt_lines, placement_counts

RES = Path("eval_out/noise_floor"); RES.mkdir(parents=True, exist_ok=True)
CSV = RES / "results.csv"
N_RUNS = 3

# fixed seeded-30 FUNSD across splits (same set every run)
allf = [(sp, img, ref) for sp in ("train", "test", "val") for img, ref in iter_pairs("funsd", split=sp)]
random.seed(7)
SET = random.sample(allf, 30)
ANN = _annotation_index(_ROOT / "form")

done = set()
if CSV.exists():
    done = {(int(r["run"]), r["id"]) for r in csv.DictReader(CSV.open())}
newfile = not CSV.exists()
fh = CSV.open("a", newline="")
w = csv.DictWriter(fh, fieldnames=["run", "id", "ref_chars", "recall", "cer", "insertion",
                                   "placed", "band_placed", "gt_words", "secs"])
if newfile:
    w.writeheader(); fh.flush()

base = cm.load()
todo = [(r, sp, img, ref) for r in range(1, N_RUNS + 1) for (sp, img, ref) in SET
        if (r, img.stem) not in done]
print(f"resuming: {len(done)} done, {len(todo)} to go ({N_RUNS} runs x {len(SET)} items)", flush=True)

for n, (run, sp, img, ref) in enumerate(todo):
    t0 = time.time()
    cfg = dataclasses.replace(base, out_dir=RES / f"out_run{run}")
    pdf, _ = ingest.to_pdf(img, RES / "derived")
    doc = process(pdf, cfg, digest=f"nf{run}_{img.stem}")   # distinct digest => fresh, no cache hit
    page = doc.pages[0]
    hyp = "\n".join(recovered_text(p) for p in doc.pages)
    s = score(ref, hyp)
    lines = gt_lines(json.loads(Path(ANN[img.stem]).read_text()), "funsd")
    W, H = Image.open(img).size
    strict = placement_counts(page, lines, W, H)
    band = placement_counts(page, lines, W, H, band=True)
    w.writerow(dict(run=run, id=img.stem, ref_chars=s["ref_chars"],
                    recall=round(s["word_recall"], 5), cer=round(s["cer"], 5),
                    insertion=round(s["insertion_rate"], 5),
                    placed=strict["placed"], band_placed=band["placed"], gt_words=strict["total"],
                    secs=round(time.time() - t0, 1)))
    fh.flush()
    if n % 10 == 0:
        print(f"[{n+1}/{len(todo)}] run{run} {img.stem} ({round(time.time()-t0,1)}s)", flush=True)
fh.close()
print("DONE", flush=True)
