"""Evidence-plan stream E3 — `escalate_below` rescue analysis (decision rule pre-registered).

The escalation feature (confidence-gated fallthrough to a stronger reader) ships disabled
(`escalate_below = 0.0`). E3's pinned rule (evidence_plan.md, 2026-07-12 pins) decides
keep-or-delete on EVIDENCE, registered before anyone looked at the numbers:

  - From stream F `results.csv` (both models saw the identical 55 items): take the worst
    decile (6 items) of `q36_35ba3b_4bit` by word_recall; count items where `q35_9b_4bit`
    scores >= +0.05 word_recall higher (a "rescue"). Symmetrically, q35's worst decile vs q36.
  - Optional, 2 live reads: the stream-A refusal pages (funsd/80707440_7443,
    sroie/X51005719863) through Qwen3.6; "rescued" = a guard-surviving reading with
    word_recall >= the archived det-fallback row + 0.05.
  - Rule: decile rescues < 2 in BOTH directions AND refusal rescues = 0 -> recommend DELETE
    the escalation plumbing. Refusal rescues >= 1 with decile rescues < 2 -> evidence for a
    simpler retry-on-refusal design option, build nothing in E. Decile rescues >= 2 either
    direction -> keep-disabled + document.

E produces evidence + a recommendation only — the delete/keep lands as a separate small PR
carrying this table, and Luke decides on that PR.

Zero-VLM for the decile table (pure re-analysis of the committed CSV). The 2 refusal reads
need the samples corpus + the MLX reader; when either is absent they are reported as PENDING
and the rule outcome is stated conditionally.

Run:  .venv/bin/python eval_out/stream_e_escalation.py
"""
import csv
from pathlib import Path

CSV_IN = Path("eval_out/stream_f_model_ab/results.csv")
OUT = Path("eval_out/stream_e_sensitivity")
OUT.mkdir(parents=True, exist_ok=True)
TABLE = OUT / "e3_rescue_table.md"

RESCUE_MARGIN = 0.05          # pinned
DECILE_N = 6                  # pinned: worst decile of 55 = 6 items
A, B = "q36_35ba3b_4bit", "q35_9b_4bit"   # current default vs rollback

rows = list(csv.DictReader(CSV_IN.open()))
by_model = {}
for r in rows:
    by_model.setdefault(r["model"], {})[(r["kind"], r["id"])] = float(r["word_recall"])

for m in (A, B):
    if m not in by_model:
        raise SystemExit(f"model {m} missing from {CSV_IN}")

shared = sorted(set(by_model[A]) & set(by_model[B]))
assert len(shared) == 55, f"expected the 55 shared stream-F items, got {len(shared)}"


def worst_decile_rescues(worst_of: str, rescuer: str):
    """Worst DECILE_N items of `worst_of` by word_recall; rescues by `rescuer`."""
    ranked = sorted(shared, key=lambda k: by_model[worst_of][k])[:DECILE_N]
    out = []
    for k in ranked:
        w, r = by_model[worst_of][k], by_model[rescuer][k]
        out.append((k, w, r, r >= w + RESCUE_MARGIN))
    return out


lines = ["# Stream E3 — escalate_below rescue analysis",
         "",
         f"Universe: the {len(shared)} items scored by BOTH `{A}` and `{B}` in stream F",
         f"(`{CSV_IN}`). Rescue = the other model scores >= +{RESCUE_MARGIN} word_recall on an",
         "item in this model's worst decile. Rule + interpretation are pinned in",
         "evidence_plan.md §E3 (2026-07-12) — registered before these numbers were computed.",
         ""]
totals = {}
for worst_of, rescuer in ((A, B), (B, A)):
    tab = worst_decile_rescues(worst_of, rescuer)
    n_resc = sum(1 for *_x, resc in tab if resc)
    totals[worst_of] = n_resc
    lines += [f"## Worst decile of `{worst_of}` — rescues by `{rescuer}`: **{n_resc}**", "",
              "| item | " + worst_of + " | " + rescuer + " | rescued |",
              "| --- | --- | --- | --- |"]
    for (kind, iid), w, r, resc in tab:
        lines.append(f"| {kind}/{iid} | {w:.4f} | {r:.4f} | {'YES' if resc else 'no'} |")
    lines.append("")

both_lt2 = all(v < 2 for v in totals.values())
lines += ["## Rule evaluation", "",
          f"- decile rescues: {A} worst-decile rescued-by-{B} = {totals[A]}; "
          f"{B} worst-decile rescued-by-{A} = {totals[B]}",
          f"- decile condition (`< 2 in BOTH directions`): {'MET' if both_lt2 else 'NOT met'}"]
if not both_lt2:
    lines += ["", "**Outcome (unconditional): KEEP-DISABLED + document** — decile rescues >= 2 "
              "in at least one direction; threshold calibration becomes a follow-up (senior "
              "eyes). Refusal reads are moot for the rule."]
else:
    lines += ["- refusal-rescue reads (funsd/80707440_7443, sroie/X51005719863): **PENDING** — "
              "needs the samples corpus + MLX reader on this machine.",
              "", "**Conditional outcome:** refusal rescues = 0 -> recommend **DELETE** the "
              "escalation plumbing; >= 1 -> surface retry-on-refusal as a design option, build "
              "nothing in E. Either way the decision lands on its own PR (Luke decides)."]
TABLE.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwritten: {TABLE}")
