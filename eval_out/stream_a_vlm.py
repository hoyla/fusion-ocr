"""Evidence-plan stream A — VLM PASS (full pipeline: PaddleOCR geometry + VLM reader).
Completes the questions the deterministic pass left open AND produces the headline FUSED
box-placement number (P1). Needs the MLX reader up (config vlm.base_url).

Scope (VLM is ~40-90s/page, so not the full SROIE): FUNSD all 199 (matches the deterministic
full set for a direct VLM-vs-det comparison), SROIE a fixed-seed 150 across splits.

Per item we score TWO artifacts (evidence-plan standing rule 3 — score the product, not just
the VLM):
  - text: score(ref, recovered_text) — recovered_text returns the UNGATED vlm_reading when
    present, so this is comparable to the historic n=16 VLM numbers (CER 0.15) and answers Q1 +
    the receipt VLM claim. Carries insertion_rate (stream D hallucination proxy, ungated).
  - placement: placement_counts on page.segments — the GATED fused output (overlay/segment_index),
    the real click-a-claim P1 number that the deterministic pass could only floor.

Durable + resumable: one CSV row per item under eval_out/stream_a_vlm/, skips done ids.
Run:  .venv/bin/python eval_out/stream_a_vlm.py
"""
import csv, json, random, time, traceback
from pathlib import Path
from PIL import Image
from fusion_ocr import config as cm, ingest
from fusion_ocr.pipeline import process
from fusion_ocr.eval.harness import recovered_text
from fusion_ocr.eval.metrics import score, normalize
from fusion_ocr.eval.datasets import (iter_pairs, _annotation_index, _ROOT, _SOURCES,
                                      _CASELESS_REF)
from fusion_ocr.eval.placement import gt_lines, placement_counts

RES = Path("eval_out/stream_a_vlm"); RES.mkdir(parents=True, exist_ok=True)
CSV = RES / "results.csv"
COLS = ["dataset", "split", "id", "ref_chars", "word_recall", "word_precision", "cer", "wer",
        "insertion_rate", "placed", "plain_placed", "gt_words", "secs"]

# work list: FUNSD all (199), SROIE fixed-seed 150 across splits
def sroie_sample():
    items = []
    for split in ("train", "test", "val"):
        for img, ref in iter_pairs("sroie", split=split):
            items.append((split, img, ref))
    random.seed(1)
    return random.sample(items, 150)

WORK = []
for split in ("train", "test", "val"):
    for img, ref in iter_pairs("funsd", split=split):
        WORK.append(("funsd", split, img, ref))
for split, img, ref in sroie_sample():
    WORK.append(("sroie", split, img, ref))

ANN = {"funsd": _annotation_index(_ROOT / "form"),
       "sroie": _annotation_index(_ROOT / "invoice")}

done = set()
if CSV.exists():
    done = {(r["dataset"], r["id"]) for r in csv.DictReader(CSV.open())}
newfile = not CSV.exists()
fh = CSV.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
if newfile:
    w.writeheader(); fh.flush()

cfg = cm.load()
import dataclasses
cfg = dataclasses.replace(cfg, out_dir=RES / "out")
todo = [t for t in WORK if (t[0], t[2].stem) not in done]
print(f"resuming: {len(done)} done, {len(todo)} to go (of {len(WORK)})", flush=True)

for n, (ds, split, img, ref) in enumerate(todo):
    t0 = time.time()
    try:
        if not normalize(ref):
            continue
        caseless = ds in _CASELESS_REF
        pdf, _ = ingest.to_pdf(img, RES / "derived")
        doc = process(pdf, cfg, digest=f"{ds}_{img.stem}")
        page = doc.pages[0]
        hyp = "\n".join(recovered_text(p) for p in doc.pages)
        s = score(ref, hyp, caseless=caseless)
        ap = ANN[ds].get(img.stem)
        pc = {"placed": 0, "plain": 0, "total": 0}
        if ap is not None:
            lines = gt_lines(json.loads(Path(ap).read_text()), ds)
            if lines:
                W, H = Image.open(img).size
                pc = placement_counts(page, lines, W, H, caseless=caseless)
        w.writerow(dict(dataset=ds, split=split, id=img.stem, ref_chars=s["ref_chars"],
                        word_recall=round(s["word_recall"], 4),
                        word_precision=round(s["word_precision"], 4),
                        cer=round(s["cer"], 4), wer=round(s["wer"], 4),
                        insertion_rate=round(s["insertion_rate"], 4),
                        placed=pc["placed"], plain_placed=pc["plain"], gt_words=pc["total"],
                        secs=round(time.time() - t0, 1)))
        fh.flush()
    except Exception:
        print(f"  ERR {ds}/{img.stem}:", flush=True); traceback.print_exc()
    if n % 10 == 0:
        print(f"[{n+1}/{len(todo)}] {ds}/{img.stem} ({round(time.time()-t0,1)}s)", flush=True)
fh.close()
print("DONE", flush=True)
