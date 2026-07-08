"""Evidence-plan stream A, deterministic rows — FULL gold-set runs (FUNSD 199 + SROIE 973,
all splits), PaddleOCR and Apple Vision, --no-vlm equivalent. Durable + resumable in the
ocrq_full.py pattern: appends one CSV row per item under eval_out/stream_a/, skips
(dataset, split, engine, id) tuples already present. Per-item scores include insertion_rate
(evidence_plan stream D1). Selection protocol: ALL items in every split, no exclusions
(items with empty ground truth are logged with ref_chars=0 and skipped from scoring).

Run:  .venv/bin/python eval_out/stream_a.py
"""
import csv, dataclasses, time, traceback
from pathlib import Path
from fusion_ocr import config as cm, ingest
from fusion_ocr.pipeline import deterministic_pipeline, process
from fusion_ocr.eval.harness import recovered_text
from fusion_ocr.eval.metrics import score, normalize
from fusion_ocr.eval.datasets import iter_pairs

RES = Path("eval_out/stream_a"); RES.mkdir(parents=True, exist_ok=True)
CSV = RES / "results.csv"
COLS = ["dataset", "split", "engine", "id", "ref_chars", "word_recall", "word_precision",
        "cer", "wer", "insertion_rate", "secs"]

done = set()
if CSV.exists():
    done = {(r["dataset"], r["split"], r["engine"], r["id"]) for r in csv.DictReader(CSV.open())}
newfile = not CSV.exists()
fh = CSV.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
if newfile:
    w.writeheader(); fh.flush()

base = cm.load()
JOBS = [(ds, split, eng)
        for ds in ("funsd", "sroie")
        for split in ("train", "test", "val")
        for eng in ("paddle", "apple_vision")]

for ds, split, eng in JOBS:
    cfg = dataclasses.replace(base, out_dir=RES / "out" / f"{ds}_{split}_{eng}",
                              prefer_apple_vision=(eng == "apple_vision"))
    pairs = iter_pairs(ds, split=split)
    todo = [(i, img, ref) for i, (img, ref) in enumerate(pairs)
            if (ds, split, eng, img.stem) not in done]
    print(f"== {ds}/{split}/{eng}: {len(pairs)} items, {len(todo)} to do", flush=True)
    for n, (i, img, ref) in enumerate(todo):
        t0 = time.time()
        try:
            if not normalize(ref):
                w.writerow(dict(dataset=ds, split=split, engine=eng, id=img.stem, ref_chars=0,
                                word_recall="", word_precision="", cer="", wer="",
                                insertion_rate="", secs=0)); fh.flush()
                continue
            pdf, _ = ingest.to_pdf(img, RES / "derived")
            doc = process(pdf, cfg, pipeline=deterministic_pipeline(),
                          digest=f"{ds}_{split}_{eng}_{i:04d}")
            hyp = "\n".join(recovered_text(p) for p in doc.pages)
            s = score(ref, hyp)
            w.writerow(dict(dataset=ds, split=split, engine=eng, id=img.stem,
                            ref_chars=s["ref_chars"], word_recall=round(s["word_recall"], 4),
                            word_precision=round(s["word_precision"], 4),
                            cer=round(s["cer"], 4), wer=round(s["wer"], 4),
                            insertion_rate=round(s["insertion_rate"], 4),
                            secs=round(time.time() - t0, 1)))
            fh.flush()
        except Exception:
            print(f"  ERR {ds}/{split}/{eng} {img.stem}:", flush=True)
            traceback.print_exc()
        if n % 25 == 0:
            print(f"  [{n + 1}/{len(todo)}] {img.stem} ({round(time.time() - t0, 1)}s)", flush=True)
fh.close()
print("DONE", flush=True)
