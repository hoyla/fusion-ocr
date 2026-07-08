"""Triage the OCR-Quality 1000-run per evidence_plan.md — a RANKING / triage / calibration
signal, NOT an accuracy scorecard (judge-approved != true). Aggregates agreement by
human_score/lang, pre-filters the noise (blank pages, 72B refusals/duplication), and emits the
genuine-divergence shortlist + the bottom decile for the hand-label queue.
Run:  .venv/bin/python eval_out/ocrq_triage.py eval_out/ocrq_full
"""
import sys, re, csv, statistics as st
from pathlib import Path
from collections import Counter
import pandas as pd

RES = Path(sys.argv[1])
DF = pd.read_parquet("samples/OCR-Quality/OCR-Quality.parquet").set_index("index")

def classify(idx, our_chars):
    ref = str(DF.loc[idx, "ocr_text"])
    if our_chars == 0 and re.search(r"blank|completely white|no text|no content", ref, re.I):
        return "blank/degenerate"
    if re.search(r"provided image|i (cannot|am unable)|unable to (extract|read)", ref, re.I) and len(ref) < 300:
        return "72b-refusal"
    lines = [l.strip() for l in ref.splitlines() if len(l.strip()) > 15]
    if lines and (len(lines) - len(set(lines))) >= 0.30 * len(lines):
        return "72b-duplication"
    return "ADJUDICATE"      # genuine divergence -> Claude-Vision adjudication

rows = [r for r in csv.DictReader((RES / "results.csv").open()) if int(r["human_score"]) >= 0]
for r in rows:
    r["agree_recall"] = float(r["agree_recall"]); r["human_score"] = int(r["human_score"])
    r["our_chars"] = int(r["our_chars"]); r["idx"] = int(r["idx"]); r["ref_chars"] = int(r["ref_chars"])

print(f"== {len(rows)} pages ==  (agreement = triage/calibration signal, NOT accuracy)")
print("\nby human_score (72B's rated quality):")
for s in sorted(set(r["human_score"] for r in rows)):
    g = [r["agree_recall"] for r in rows if r["human_score"] == s]
    print(f"  score {s}: n={len(g):4d}  mean_agree={st.mean(g):.3f}")
print("\nby language:")
for lang in sorted(set(r["lang"] for r in rows)):
    g = [r["agree_recall"] for r in rows if r["lang"] == lang]
    print(f"  {lang:4s}: n={len(g):4d}  mean_agree={st.mean(g):.3f}")

# The informative slice: LOW agreement on SCORE-1 (good reference, yet we diverge) -> our-error
# or 72B-artifact candidates. Classify to strip the noise before Claude-Vision adjudication.
susp = [r for r in rows if r["human_score"] == 1 and r["agree_recall"] < 0.60]
for r in susp:
    r["cls"] = classify(r["idx"], r["our_chars"])
print(f"\n== score-1 divergences (agree<0.60): {len(susp)} ==")
print("  ", Counter(r["cls"] for r in susp).most_common())
adj = sorted([r for r in susp if r["cls"] == "ADJUDICATE"], key=lambda r: r["agree_recall"])
print(f"\n== ADJUDICATE shortlist (score-1, genuine divergence): {len(adj)} — worst 25 ==")
print(f"  {'idx':>4} {'agree':>6} {'refchars':>8}  source")
for r in adj[:25]:
    print(f"  {r['idx']:>4} {r['agree_recall']:>6.3f} {r['ref_chars']:>8}  {r['source']}")

# Bottom decile overall -> hand-label queue (most informative items, per evidence_plan A/B/D3)
botN = max(1, len(rows) // 10)
bottom = sorted(rows, key=lambda r: r["agree_recall"])[:botN]
Path(RES / "hand_label_queue.csv").write_text(
    "idx,source,human_score,agree_recall\n" +
    "\n".join(f"{r['idx']},{r['source']},{r['human_score']},{r['agree_recall']:.3f}" for r in bottom))
Path(RES / "adjudicate.csv").write_text(
    "idx,source,agree_recall,ref_chars\n" +
    "\n".join(f"{r['idx']},{r['source']},{r['agree_recall']:.3f},{r['ref_chars']}" for r in adj))
print(f"\nwrote hand_label_queue.csv (bottom decile, n={botN}) + adjudicate.csv (n={len(adj)})")
