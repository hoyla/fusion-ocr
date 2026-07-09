"""Evidence-plan stream D1 — the gated column (P2: the ink-gate's benefit AND its cost).

Pure re-scoring of the 349 archived stream-A VLM docs (NO VLM compute). Docs live at
`<CORSAIR>/eval_out/stream_a_vlm/out/<dataset>_<id>/doc.json` (CORSAIR must be mounted); the
committed `stream_a_vlm/results.csv` supplies the item list + the archive-integrity check.

For each item, score TWO hypotheses against the same gold reference (SROIE caseless both sides,
`datasets._CASELESS_REF`):

  - UNGATED = `"\n".join(recovered_text(p) for p in doc.pages)` — what `document.md` carries when
    VLM-read (== `page.vlm_reading` for VLM-read pages; falls back to the segment reassembly only
    where the guards discarded the read). This is the exact hyp `stream_a_vlm.py` scored, so its
    per-item numbers must REPRODUCE the committed results.csv (word_recall/insertion_rate/cer/
    ref_chars) — a built-in check that this archive is the same run.
  - GATED   = non-superseded segments' `best_text` in `compose.reading_key` order, joined "\n" —
    the fallback branch of `harness.recovered_text` applied UNCONDITIONALLY. This is the text of
    `segment_index.json` / the overlay (the same filter `placement.py` uses): the gated product.

Reports, per item and micro-averaged (per corpus + overall), for BOTH columns:
  insertion_rate (char-level — the pre-registered P2 proxy) · word_recall · word_precision
  (1 - precision = the word-level hallucination companion).
  Gate BENEFIT = ungated - gated insertion; gate COST = ungated - gated recall.
  Publishing the benefit without the cost flatters the gate — both are in the manifest.

Escalation tripwires (DIAGNOSIS triggers, NOT pass bars — D1 is a first measurement):
  (b) aggregate gated insertion NOT lower than ungated; (c) gate recall cost > 0.05 absolute on
  either corpus. If either fires -> STOP and get senior eyes before writing a manifest verdict
  (the strict-vs-band placement confound is the precedent). This runner only FLAGS them.

Durable/resumable CSV-append. D1 needs NO reader.
Run:  .venv/bin/python eval_out/insertion_gate.py
"""
import csv
import json
from datetime import date
from pathlib import Path

from fusion_ocr.compose import reading_key
from fusion_ocr.eval.datasets import _CASELESS_REF, iter_pairs
from fusion_ocr.eval.harness import recovered_text
from fusion_ocr.eval.metrics import score, word_tokens
from fusion_ocr.models import Document

ARCH = Path("/Volumes/CORSAIR/Work_Corsair/fusion-ocr/eval_out/stream_a_vlm")
SRC_CSV = ARCH / "results.csv"          # committed stream-A run: item list + reproduce check
OUT = ARCH / "out"
RES = Path("eval_out/insertion_gate"); RES.mkdir(parents=True, exist_ok=True)
CSV = RES / "results.csv"

# per-item raw counts (enough to micro-average both columns without re-running) + eyeball rates
COLS = ["dataset", "split", "id", "reproduced", "vlm_empty", "ref_chars", "ref_words",
        "u_ins", "u_recall", "u_prec", "u_char_ins", "u_overlap", "u_hyp_words",
        "g_ins", "g_recall", "g_prec", "g_char_ins", "g_overlap", "g_hyp_words"]


def gated_text(doc) -> str:
    """The gated product: non-superseded segments' best_text in reading order, over pages.
    Mirrors harness.recovered_text's fallback branch, applied unconditionally."""
    parts = []
    for page in doc.pages:
        segs = [s for s in page.segments if s.best_text and not s.superseded]
        segs.sort(key=lambda s: reading_key(
            s, page.regions, page.rotation, page.width, page.height))
        parts.append("\n".join(s.best_text for s in segs))
    return "\n".join(parts)


def ungated_text(doc) -> str:
    return "\n".join(recovered_text(p) for p in doc.pages)


# reference map by (dataset, stem) across all splits — same refs stream_a_vlm.py used
refmap = {}
for ds in ("funsd", "sroie"):
    for split in ("train", "test", "val"):
        for img, ref in iter_pairs(ds, split=split):
            refmap[(ds, img.stem)] = ref

# item list + committed columns for the reproduce check
src = {(r["dataset"], r["id"]): r for r in csv.DictReader(SRC_CSV.open())}

done = set()
if CSV.exists():
    done = {(r["dataset"], r["id"]) for r in csv.DictReader(CSV.open())}
newfile = not CSV.exists()
fh = CSV.open("a", newline=""); w = csv.DictWriter(fh, fieldnames=COLS)
if newfile:
    w.writeheader(); fh.flush()

todo = [k for k in src if k not in done]
print(f"resuming: {len(done)} done, {len(todo)} to go (of {len(src)})", flush=True)

for n, (ds, rid) in enumerate(todo):
    docp = OUT / f"{ds}_{rid}" / "doc.json"
    if not docp.exists():
        print(f"  MISSING {docp}", flush=True); continue
    doc = Document.from_json(docp.read_text())
    caseless = ds in _CASELESS_REF
    ref = refmap[(ds, rid)]
    ung, gat = ungated_text(doc), gated_text(doc)
    su = score(ref, ung, caseless=caseless)
    sg = score(ref, gat, caseless=caseless)

    r = src[(ds, rid)]
    reproduced = (round(su["word_recall"], 4) == float(r["word_recall"])
                  and round(su["insertion_rate"], 4) == float(r["insertion_rate"])
                  and round(su["cer"], 4) == float(r["cer"])
                  and su["ref_chars"] == int(r["ref_chars"]))
    vlm_empty = not any(p.vlm_reading.strip() for p in doc.pages)
    # true (unfloored) hyp token counts for honest precision / invented-word accounting
    u_hyp = len(word_tokens(ung.casefold() if caseless else ung))
    g_hyp = len(word_tokens(gat.casefold() if caseless else gat))

    w.writerow(dict(
        dataset=ds, split=r["split"], id=rid, reproduced=int(reproduced),
        vlm_empty=int(vlm_empty), ref_chars=su["ref_chars"], ref_words=su["ref_words"],
        u_ins=round(su["insertion_rate"], 4), u_recall=round(su["word_recall"], 4),
        u_prec=round(su["word_precision"], 4), u_char_ins=su["char_ins"],
        u_overlap=su["word_overlap"], u_hyp_words=u_hyp,
        g_ins=round(sg["insertion_rate"], 4), g_recall=round(sg["word_recall"], 4),
        g_prec=round(sg["word_precision"], 4), g_char_ins=sg["char_ins"],
        g_overlap=sg["word_overlap"], g_hyp_words=g_hyp))
    fh.flush()
    if n % 25 == 0:
        print(f"[{n+1}/{len(todo)}] {ds}/{rid} u_ins={su['insertion_rate']:.3f} "
              f"g_ins={sg['insertion_rate']:.3f}", flush=True)
fh.close()

# ---- aggregate (micro-avg per corpus + overall) + tripwire evaluation ----
rows = list(csv.DictReader(CSV.open()))
for r in rows:
    for k in ("ref_chars", "ref_words", "u_char_ins", "u_overlap", "u_hyp_words",
              "g_char_ins", "g_overlap", "g_hyp_words", "reproduced", "vlm_empty"):
        r[k] = int(r[k])


def agg(subset):
    cn = sum(r["ref_chars"] for r in subset) or 1
    wn = sum(r["ref_words"] for r in subset) or 1
    uh = sum(r["u_hyp_words"] for r in subset) or 1
    gh = sum(r["g_hyp_words"] for r in subset) or 1
    return {
        "n": len(subset),
        "u_ins": sum(r["u_char_ins"] for r in subset) / cn,
        "g_ins": sum(r["g_char_ins"] for r in subset) / cn,
        "u_recall": sum(r["u_overlap"] for r in subset) / wn,
        "g_recall": sum(r["g_overlap"] for r in subset) / wn,
        "u_prec": sum(r["u_overlap"] for r in subset) / uh,
        "g_prec": sum(r["g_overlap"] for r in subset) / gh,
        "u_inv_words": sum(r["u_hyp_words"] - r["u_overlap"] for r in subset),
        "g_inv_words": sum(r["g_hyp_words"] - r["g_overlap"] for r in subset),
    }


summary = {"date": str(date.today()), "n_items": len(rows),
           "reproduced": sum(r["reproduced"] for r in rows),
           "vlm_empty": sum(r["vlm_empty"] for r in rows), "corpora": {}, "tripwires": {}}
for ds in ("funsd", "sroie"):
    a = agg([r for r in rows if r["dataset"] == ds])
    a["benefit_ins"] = a["u_ins"] - a["g_ins"]      # gate benefit (char insertion reduced)
    a["cost_recall"] = a["u_recall"] - a["g_recall"]  # gate cost (true words dropped)
    summary["corpora"][ds] = a
summary["corpora"]["overall"] = agg(rows)
summary["corpora"]["overall"]["benefit_ins"] = (
    summary["corpora"]["overall"]["u_ins"] - summary["corpora"]["overall"]["g_ins"])
summary["corpora"]["overall"]["cost_recall"] = (
    summary["corpora"]["overall"]["u_recall"] - summary["corpora"]["overall"]["g_recall"])

# tripwires (b) and (c) — (a) and (d) belong to D2 / D3
tw_b = {ds: summary["corpora"][ds]["g_ins"] >= summary["corpora"][ds]["u_ins"]
        for ds in ("funsd", "sroie", "overall")}
tw_c = {ds: summary["corpora"][ds]["cost_recall"] > 0.05 for ds in ("funsd", "sroie")}
summary["tripwires"] = {"b_gated_not_lower": tw_b, "c_recall_cost_gt_0.05": tw_c}

(RES / "summary.json").write_text(json.dumps(summary, indent=2))

print("\n=== D1 SUMMARY ===")
print(f"items={summary['n_items']} reproduced={summary['reproduced']}/{summary['n_items']} "
      f"vlm_empty={summary['vlm_empty']}")
hdr = f"{'corpus':8} {'n':>4} {'u_ins':>7} {'g_ins':>7} {'benefit':>8} " \
      f"{'u_rec':>7} {'g_rec':>7} {'cost':>7} {'u_prec':>7} {'g_prec':>7}"
print(hdr)
for ds in ("funsd", "sroie", "overall"):
    a = summary["corpora"][ds]
    print(f"{ds:8} {a['n']:>4} {a['u_ins']:>7.4f} {a['g_ins']:>7.4f} "
          f"{a.get('benefit_ins',0):>8.4f} {a['u_recall']:>7.4f} {a['g_recall']:>7.4f} "
          f"{a.get('cost_recall',0):>7.4f} {a['u_prec']:>7.4f} {a['g_prec']:>7.4f}")
print("\nTRIPWIRES (diagnosis triggers, not pass bars):")
print(f"  (b) gated insertion NOT lower than ungated: {tw_b}")
print(f"  (c) recall cost > 0.05 (per corpus): {tw_c}")
if summary["reproduced"] != summary["n_items"]:
    print("\n!! ARCHIVE INTEGRITY: some items did NOT reproduce results.csv — STOP, investigate.")
if any(tw_b.values()) or any(tw_c.values()):
    print("\n!! TRIPWIRE FIRED — stop and diagnose before a manifest verdict (senior eyes).")
print("DONE", flush=True)
