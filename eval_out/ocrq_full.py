"""Full OCR-Quality run (all 1000) through our engine — DURABLE rerun (writes under
eval_out/, not the session tmp scratchpad that ate the first run). Resumable: appends one CSV
row per page and skips indices already in the CSV. Comparative only (agreement with the 72B
ocr_text, formatting normalised out); human_score 1=best rates the 72B's quality, so trust
agreement most on score-1. Saves our per-page output for later Claude-Vision divergence triage.
Run:  .venv/bin/python eval_out/ocrq_full.py eval_out/ocrq_full
"""
import sys, re, csv, time, dataclasses
from pathlib import Path
import pandas as pd
from fusion_ocr import config as cm, ingest
from fusion_ocr.pipeline import process
from fusion_ocr.eval.harness import recovered_text
from fusion_ocr.eval.metrics import score, normalize

RES = Path(sys.argv[1]); OURS = RES / "ours"; CSV = RES / "results.csv"
OURS.mkdir(parents=True, exist_ok=True)
DF = pd.read_parquet("samples/OCR-Quality/OCR-Quality.parquet").set_index("index")

def strip_fmt(t):
    t = re.sub(r"\\[\(\)\[\]]", " ", t); t = re.sub(r"[|#*`_>]", " ", t)
    t = re.sub(r":?-{2,}:?", " ", t); return normalize(t)

done = {int(r["idx"]) for r in csv.DictReader(CSV.open())} if CSV.exists() else set()
cols = ["idx", "source", "lang", "human_score", "ref_chars", "our_chars", "agree_recall",
        "agree_cer", "prec", "secs", "t_ocr_det", "t_vlm_read", "t_language", "t_layout",
        "t_table_read", "t_fusion", "t_total"]
newfile = not CSV.exists()
fh = CSV.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=cols)
if newfile:
    w.writeheader(); fh.flush()

cfg = cm.load()
cfg = dataclasses.replace(cfg, out_dir=RES / "out")
derived = RES / "derived"; derived.mkdir(exist_ok=True)
todo = [i for i in DF.index.tolist() if i not in done]
print(f"resuming: {len(done)} done, {len(todo)} to go", flush=True)
for n, i in enumerate(todo):
    t0 = time.time(); src = "?"
    try:
        ref = DF.loc[i, "ocr_text"]; src = str(DF.loc[i, "source"]); hs = int(DF.loc[i, "human_score"])
        lang = src.split("-")[0]
        pdf, _ = ingest.to_pdf(Path(f"samples/OCR-Quality/pics/{i}.png"), derived)
        doc = process(pdf, cfg, digest=f"ocrq_{i:04d}")
        hyp = "\n".join(recovered_text(p) for p in doc.pages)
        s = score(strip_fmt(ref), strip_fmt(hyp)); ss = doc.stage_seconds
        (OURS / f"{i}.txt").write_text(hyp)
        w.writerow(dict(idx=i, source=src, lang=lang, human_score=hs,
            ref_chars=s["ref_chars"], our_chars=len(normalize(hyp)),
            agree_recall=round(s["word_recall"], 4), agree_cer=round(s["cer"], 4),
            prec=round(s["word_precision"], 4), secs=round(time.time() - t0, 1),
            t_ocr_det=ss.get("ocr_det", 0), t_vlm_read=ss.get("vlm_read", 0),
            t_language=ss.get("language", 0), t_layout=ss.get("layout", 0),
            t_table_read=ss.get("table_read", 0), t_fusion=ss.get("fusion", 0),
            t_total=round(sum(ss.values()), 3)))
        fh.flush()
    except Exception as e:
        w.writerow(dict(idx=i, source=src, lang="?", human_score=-1, ref_chars=-1, our_chars=-1,
            agree_recall=-1, agree_cer=-1, prec=-1, secs=round(time.time() - t0, 1),
            t_ocr_det=0, t_vlm_read=0, t_language=0, t_layout=0, t_table_read=0, t_fusion=0,
            t_total=0)); fh.flush()
        print(f"  ERR idx {i}: {type(e).__name__}: {e}", flush=True)
    if n % 10 == 0:
        print(f"[{n+1}/{len(todo)}] idx {i} ({round(time.time() - t0, 1)}s)", flush=True)
fh.close()
print("DONE", flush=True)
